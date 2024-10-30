import argparse
import os
import pprint
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, checkpoint_module
from timm.models.vision_transformer import (
    # Block,
    PatchEmbed,
    # _init_vit_weights,
    trunc_normal_,
    lecun_normal_,
    VisionTransformer,
    vit_base_patch32_224,
    DropPath,
    # Attention,
    Mlp,
    _cfg,
)
from einops import rearrange
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MoeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, pretraining_tp=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        # self.act_fn = nn.Identity()
        # self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.pretraining_tp = pretraining_tp

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0) 
            # print(self.up_proj.weight.size(), self.down_proj.weight.size())
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=-1)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            # down_proj = self.linear(x)

        return down_proj
    
import math
def _capacity(gates, capacity_factor=1, min_capacity=0):
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity

class ECMoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape    
        # print(bsz, seq_len, h)    
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            # scores = logits.softmax(dim=-1)
            scores = torch.sigmoid(logits)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        # topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        # select top-k tokens
        capacity = _capacity(scores, self.top_k)
        topk_score , topk_ids = torch.topk(scores.T, k=capacity, dim=-1, sorted=False) # (n_experts, capacity)
        
        return topk_ids, topk_score

class SparseECMoeBlock(nn.Module):
    """
    ***Expert-choice MoE Block***
    A mixed expert module containing shared experts.
    for deepk, maybe sharing experts and gates, or only sharing gates, defaults is not sharing anything
    """
    def __init__(self, embed_dim, mlp_ratio=4, num_experts=16, num_experts_per_tok=2, pretraining_tp=1, n_shared_experts = 2, deepk=1, expert_hidden_dim=None, num_sub_tokens=1, ffn_type="glu", actless=False):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        assert num_experts % deepk == 0, "num_experts must be divisible by deepk"
        assert num_experts_per_tok % deepk == 0, "num_experts_per_tok must be divisible by deepk"
        self.num_sub_tokens = num_sub_tokens
        expert_hidden_dim = expert_hidden_dim if expert_hidden_dim is not None else mlp_ratio * embed_dim
        if num_sub_tokens > 1:
            self.token_split = nn.Linear(embed_dim, embed_dim)
            self.token_merge = nn.Linear(embed_dim, embed_dim)
            embed_dim = embed_dim // num_sub_tokens
            self.embed_dim = embed_dim

        if ffn_type == "ori":
            self.experts = nn.ModuleList([Mlp(in_features=embed_dim, hidden_features=expert_hidden_dim, act_layer=nn.Identity if actless else nn.GELU) for i in range(num_experts)])
        elif ffn_type == "glu":
            self.experts = nn.ModuleList([MoeMLP(hidden_size = embed_dim, intermediate_size = expert_hidden_dim, pretraining_tp=pretraining_tp) for i in range(num_experts)])
        else:
            raise NotImplementedError(f"mlp type {ffn_type} is not implemented")
        
        # self.experts = nn.ModuleList([mlp(hidden_size = embed_dim, intermediate_size = expert_hidden_dim, pretraining_tp=pretraining_tp) for i in range(num_experts)])
        # self.gate = ECMoEGate(embed_dim=embed_dim, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.gates = nn.ModuleList([ECMoEGate(embed_dim=embed_dim, num_experts=num_experts//deepk, num_experts_per_tok=num_experts_per_tok//deepk) for i in range(deepk)])
        self.n_shared_experts = n_shared_experts
        self.deepk = deepk
        
        if self.n_shared_experts is not None and self.n_shared_experts > 0:
            intermediate_size =  expert_hidden_dim * self.n_shared_experts // mlp_ratio
            self.shared_experts = MoeMLP(hidden_size = embed_dim, intermediate_size = intermediate_size, pretraining_tp=pretraining_tp) if ffn_type == "glu" else Mlp(in_features=embed_dim, hidden_features=intermediate_size, act_layer=nn.Identity if actless else nn.GELU)
    

    def forward(self, hidden_states):
        if self.num_sub_tokens > 1:
            hidden_states = self.token_split(hidden_states)
            hidden_states = rearrange(hidden_states, 'b l (s d) -> b (l s) d', s=self.num_sub_tokens)
            hidden_states = hidden_states.float()

        identity = hidden_states
        orig_shape = hidden_states.shape
        # if self.training:
        if self.deepk == 1:
            # hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0) # (64*256*2, 384) remove because actiavtion is not fixed to 2
            topk_ids, topk_score = self.gates[0](hidden_states) # (n_experts, capacity)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            y = torch.zeros_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts): 
                y[topk_ids[i]] = y[topk_ids[i]] + expert(hidden_states[topk_ids[i]]).float() * topk_score[i].view(-1, 1)
            # y = SparseECMoeBlock.parallel_expert_forward(self.experts, hidden_states, topk_ids, topk_score)
            y =  y.view(*orig_shape)
        else:
            for i, gate in enumerate(self.gates):
                topk_ids, topk_score = gate(hidden_states)
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                y = hidden_states.clone()
                for j, expert in enumerate(self.experts[i*len(self.experts)//self.deepk:(i+1)*len(self.experts)//self.deepk]): 
                    y[topk_ids[j]] = y[topk_ids[j]] + expert(hidden_states[topk_ids[j]]).float() * topk_score[j].view(-1, 1)
                y =  y.view(*orig_shape)
                hidden_states = y

        if self.n_shared_experts is not None and self.n_shared_experts > 0:
            y = y + self.shared_experts(identity)

        if self.num_sub_tokens > 1:
            y = rearrange(y, 'b (l s) d -> b l (s d)', s=self.num_sub_tokens)
            y = self.token_merge(y)
        return y


class FusedMoELinear(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim):
        super(FusedMoELinear, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = nn.Parameter(torch.empty(num_experts, output_dim, input_dim))
        self.bias = nn.Parameter(torch.empty(num_experts, output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.input_dim
        bound = 1 / math.sqrt(fan_in)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: (num_experts, sequence_length, input_dim)
        # weight: (num_experts, output_dim, input_dim)
        # y = torch.einsum('esi,eoi->eso', x, self.weight) + self.bias.unsqueeze(1)
        # y = torch.bmm(x, self.weight.transpose(1, 2)) + self.bias.unsqueeze(1)
        y = torch.baddbmm(self.bias.unsqueeze(1), x, self.weight.transpose(1, 2))
        return y

# from torch_xla.experimental.custom_kernel import _histogram, GMM
# class FusedMoELinear(nn.Module):
#     def __init__(self, num_experts, input_dim, output_dim):
#         super(FusedMoELinear, self).__init__()
#         print("with gmm fusing!")
#         self.num_experts = num_experts
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         # 权重和偏置参数
#         self.weight = nn.Parameter(torch.empty(num_experts, output_dim, input_dim))
#         self.bias = nn.Parameter(torch.empty(num_experts, output_dim))

#         self.reset_parameters()

#     def reset_parameters(self):
#         fan_in = self.input_dim
#         bound = 1 / math.sqrt(fan_in)
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         init.uniform_(self.bias, -bound, bound)

#     def forward(self, x):
#         """
#         前向传播实现：
#         x: [num_experts, sequence_length, input_dim]
#         返回: [num_experts, sequence_length, output_dim]
#         """
#         num_experts, sequence_length, input_dim = x.shape

#         # 重塑输入张量为 [num_experts * sequence_length, input_dim]
#         x_flat = x.reshape(num_experts * sequence_length, input_dim)

#         # 转置权重张量为 [num_experts, input_dim, output_dim]
#         weight_flat = self.weight.transpose(1, 2)  # [num_experts, input_dim, output_dim]

#         # 创建分组大小张量 [num_experts]，每个元素为 sequence_length
#         group_sizes = torch.full(
#             (num_experts,), sequence_length, dtype=torch.int32, device=x.device
#         )

#         # 使用 torch.ops.xla.gmm 进行分组矩阵乘法
#         # 假设 torch.ops.xla.gmm 的签名为 (x_flat, weight_flat, group_sizes) -> y_flat
#         # y_flat = torch.ops.xla.gmm(x_flat, weight_flat, group_sizes)
#         y_flat = GMM.apply(x_flat, weight_flat, group_sizes)

#         # 重塑输出张量为 [num_experts, sequence_length, output_dim]
#         y = y_flat.reshape(num_experts, sequence_length, self.output_dim)

#         # 添加偏置，偏置形状为 [num_experts, output_dim]，通过 unsqueeze(1) 变为 [num_experts, 1, output_dim]
#         y = y + self.bias.unsqueeze(1)

#         return y

class FusedSparseECMoeBlock(nn.Module):
    """
    使用FusedMoELinear的稀疏专家选择MoE块
    """
    def __init__(self, embed_dim, mlp_ratio=4, num_experts=16, num_experts_per_tok=2,
                 pretraining_tp=2, n_shared_experts=2, deepk=1, expert_hidden_dim=None,
                 num_sub_tokens=1, ffn_type="glu", actless=False):
        super().__init__()
        print("using fused kernel moe!")
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.ffn_type = ffn_type

        # Assertions for deprecated parameters
        assert deepk == 1, "deepk is deprecated and must be 1"
        assert num_sub_tokens == 1, "num_sub_tokens is deprecated and must be 1"
        
        expert_hidden_dim = expert_hidden_dim if expert_hidden_dim is not None else int(mlp_ratio * embed_dim)
        
        self.gate = ECMoEGate(embed_dim=embed_dim, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        
        if ffn_type == "glu":
            self.fused_up = FusedMoELinear(num_experts, embed_dim, expert_hidden_dim)
            self.fused_gate = FusedMoELinear(num_experts, embed_dim, expert_hidden_dim)
            self.fused_down = FusedMoELinear(num_experts, expert_hidden_dim, embed_dim)
            self.act_fn = nn.Identity() if actless else nn.SiLU()
        elif ffn_type == "ori":
            self.fused_up = FusedMoELinear(num_experts, embed_dim, expert_hidden_dim)
            self.fused_down = FusedMoELinear(num_experts, expert_hidden_dim, embed_dim)
            self.act_fn = nn.Identity() if actless else nn.SiLU()
        else:
            raise NotImplementedError(f"不支持的ffn类型: {ffn_type}")
        
        self.n_shared_experts = n_shared_experts
        if self.n_shared_experts > 0:
            intermediate_size = expert_hidden_dim * self.n_shared_experts // mlp_ratio
            if ffn_type == "glu":
                self.shared_experts = MoeMLP(
                    hidden_size=embed_dim,
                    intermediate_size=intermediate_size,
                    pretraining_tp=pretraining_tp,
                    act_layer=nn.Identity if actless else nn.GELU
                )
            else:
                self.shared_experts = Mlp(
                    in_features=embed_dim,
                    hidden_features=intermediate_size,
                    act_layer=nn.Identity if actless else nn.GELU
                )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        
        # 获取每个token的专家分配
        topk_ids, topk_score = self.gate(hidden_states)  # (n_experts, capacity)
        
        # 重塑hidden_states以适应FusedMoELinear的输入
        hidden_states = hidden_states.view(-1, self.embed_dim)
        
        # 为每个专家准备输入
        expert_inputs = []
        for i in range(self.num_experts):
            expert_input = hidden_states[topk_ids[i]]
            expert_inputs.append(expert_input)
        
        # 将专家输入堆叠成一个张量
        expert_inputs = torch.stack(expert_inputs, dim=0)  # (num_experts, max_tokens, embed_dim)
        
        if self.ffn_type == "glu":
            up = self.fused_up(expert_inputs)
            gate = self.fused_gate(expert_inputs)
            expert_outputs = self.act_fn(gate) * up
            expert_outputs = self.fused_down(expert_outputs)
        else:  # ori
            expert_outputs = self.fused_up(expert_inputs)
            expert_outputs = self.act_fn(expert_outputs)
            expert_outputs = self.fused_down(expert_outputs)
        
        # 重建hidden_states
        output = torch.zeros_like(hidden_states)
        for i in range(self.num_experts):
            output[topk_ids[i]] += expert_outputs[i].to(output.dtype) * topk_score[i].unsqueeze(-1)
        
        output = output.view(*orig_shape)
        
        if self.n_shared_experts > 0:
            output = output + self.shared_experts(identity)
        
        return output

class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4, actless=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = FusedSparseECMoeBlock(embed_dim=dim, mlp_ratio=1, num_experts=8, num_experts_per_tok=2, n_shared_experts=1, actless=actless, expert_hidden_dim=512, ffn_type='ori')
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 


class vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = Block,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,
                dpr_constant=True,init_scale=1e-4,
                mlp_ratio_clstk = 4.0,**kwargs):
        super().__init__()
        
        self.dropout_rate = drop_rate

            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale, actless=False if i<2 or i>=depth-2 else True)
                # act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale, actless=False)
            for i in range(depth)])
        

        
            
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = x + self.pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
            
        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):

        x = self.forward_features(x)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        return x

def vit_medium_patch32(pretrained=False, img_size=224, pretrained_21k = False, **kwargs):
    model = vit_models(
        patch_size=32, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Block, **kwargs)
    model.default_cfg = _cfg()
    return model 
# def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
#     """ ViT weight initialization
#     * When called without n, head_bias, jax_impl args it will behave exactly the same
#       as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
#     * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
#     """
#     if isinstance(module, nn.Linear):
#         if name.startswith('head'):
#             nn.init.zeros_(module.weight)
#             nn.init.constant_(module.bias, head_bias)
#         elif name.startswith('pre_logits'):
#             lecun_normal_(module.weight)
#             nn.init.zeros_(module.bias)
#         else:
#             if jax_impl:
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     if 'mlp' in name:
#                         nn.init.normal_(module.bias, std=1e-6)
#                     else:
#                         nn.init.zeros_(module.bias)
#             else:
#                 trunc_normal_(module.weight, std=.02)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#     elif jax_impl and isinstance(module, nn.Conv2d):
#         # NOTE conv was left to pytorch default in my original init
#         lecun_normal_(module.weight)
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)
#     elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
#         nn.init.zeros_(module.bias)
#         nn.init.ones_(module.weight)

from utils import (
    FakeImageNetDataset,
    SmoothedValue,
    get_warmup_cosine_scheduler,
    save_ckpt,
    load_ckpt,
)


def build_datasets(cfg, device):
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()

    assert cfg.batch_size % world_size == 0
    local_batch_size = cfg.batch_size // world_size

    if not cfg.fake_data:
        xm.master_print(f"loading images from directory: {cfg.data_dir}")
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(cfg.image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(cfg.data_dir, "train"), train_transform)
        val_transform = T.Compose(
            [
                T.Resize((cfg.image_size * 256) // 224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(cfg.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        val_dataset = torchvision.datasets.ImageFolder(os.path.join(cfg.data_dir, "val"), val_transform)
    else:
        xm.master_print("loading fake images")
        train_dataset = FakeImageNetDataset(cfg.image_size, 1281167)
        val_dataset = FakeImageNetDataset(cfg.image_size, 50000)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, drop_last=True, shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader = pl.MpDeviceLoader(train_loader, device)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, drop_last=True, shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=local_batch_size,
        sampler=val_sampler,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = pl.MpDeviceLoader(val_loader, device)
    return (
        train_dataset,
        train_loader,
        train_sampler,
        val_dataset,
        val_loader,
        val_sampler,
    )


# class FSDPViTModel(nn.Module):
#     """
#     To train large models that cannot fit into a single TPU, one should use nested
#     FSDP (wrapping sub-modules with inner FSDP when building the entire model).
#     This class provides an example for nested FSDP.
#     """

#     def __init__(
#         self,
#         image_size,
#         patch_size,
#         embed_dim,
#         num_heads,
#         num_blocks,
#         mlp_ratio,
#         pos_dropout,
#         mlp_dropout,
#         att_dropout,
#         num_classes,
#         grad_ckpt_wrap,
#         fsdp_wrap,
#     ):
#         super().__init__()

#         # image patch and positional embedding
#         self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
#         _init_vit_weights(self.patch_embed)
#         num_patches = self.patch_embed.num_patches
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#         trunc_normal_(self.pos_embed, std=0.02)
#         self.pos_drop = nn.Dropout(pos_dropout)

#         # vision transformer blocks
#         blocks = []
#         for idx in range(num_blocks):
#             block = Block(  # using the ViT block from the timm library
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=True,
#                 # drop=mlp_dropout,
#                 proj_drop=mlp_dropout,
#                 attn_drop=att_dropout,
#             )
#             _init_vit_weights(block)  # note: init module weights BEFORE wrapping with FSDP
#             # note: to use gradient checkpointing, wrap the module with gradient checkpointing
#             # wrapper BEFORE wrapping it with FSDP
#             block = fsdp_wrap(grad_ckpt_wrap(block))
#             blocks.append(block)
#             xm.master_print(f"built ViT block {idx}")
#         self.blocks = nn.Sequential(*blocks)

#         # classifier
#         self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
#         _init_vit_weights(self.norm)
#         self.head = nn.Linear(embed_dim, num_classes)

#     def forward(self, image):
#         x = self.patch_embed(image) + self.pos_embed
#         x = self.pos_drop(x)
#         x = self.blocks(x)
#         # here we use average pooling over image sequence (instead of using [CLS])
#         # as in https://arxiv.org/abs/2106.04560
#         logits = self.head(torch.mean(self.norm(x), dim=1))
#         return logits


def build_fsdp_vit_model(cfg, device):
    """
    Create a ViT model with nested FSDP and gradient checkpointing
    """
    # model = vit_base_patch32_224()
    model = vit_medium_patch32()
    # model = VisionTransformer(
    #     img_size = 224,
    #     patch_size = 16,
    #     embed_dim = 512,
    #     depth = 24,
    #     num_heads = 8
    # )
    if cfg.run_without_fsdp:
        return model.to(device)

    def fsdp_wrap(module):
        # note: to implement ZeRO-3, set `cfg.reshard_after_forward` to True
        # FSDP can directly wrap a module on CPU in https://github.com/pytorch/xla/pull/3992
        # so one doesn't need to cast the module into XLA devices first.
        return FSDP(
            module if cfg.shard_on_cpu else module.to(device),
            reshard_after_forward=cfg.reshard_after_forward,
            flatten_parameters=cfg.flatten_parameters,
        )

    # model = FSDPViTModel(
    #     image_size=cfg.image_size,
    #     patch_size=cfg.patch_size,
    #     embed_dim=cfg.embed_dim,
    #     num_heads=cfg.num_heads,
    #     num_blocks=cfg.num_blocks,
    #     mlp_ratio=cfg.mlp_ratio,
    #     pos_dropout=cfg.pos_dropout,
    #     mlp_dropout=cfg.mlp_dropout,
    #     att_dropout=cfg.att_dropout,
    #     num_classes=cfg.num_classes,
    #     grad_ckpt_wrap=checkpoint_module if cfg.grad_ckpt else (lambda x: x),
    #     fsdp_wrap=fsdp_wrap,
    # )
    # note: always wrap the base model with an outer (root) FSDP
    # (we don't need to apply gradient checkpointing to the base model)

    if cfg.use_nested_fsdp:
        grad_ckpt_wrap = checkpoint_module if cfg.grad_ckpt else (lambda x: x)
        for submodule_name, submodule in model.named_children():
            if sum(p.numel() for p in submodule.parameters()) == 0:
                continue
            m_fsdp = fsdp_wrap(grad_ckpt_wrap(getattr(model, submodule_name)))
            setattr(model, submodule_name, m_fsdp)
    model = fsdp_wrap(model)
    return model


def run_logging(epoch, step, smoothed_loss, smoothed_time, loss, lr, device):
    loss_value = loss.item()
    reduced_loss = xm.mesh_reduce("loss_value", loss_value, sum)
    reduced_loss /= xm.xrt_world_size()
    smoothed_loss.update(reduced_loss, batch_size=1)
    xm.master_print(
        f"epoch {epoch} step {(step + 1)}, lr: {lr:.4f}, "
        f"loss: {smoothed_loss.avg:.4f}, "
        f"sec/iter: {smoothed_time.avg:.4f}, "
        f"TPU memory: {xm.get_memory_info(device)}"
    )


def train(cfg):
    torch.manual_seed(cfg.seed)
    batch_size = cfg.batch_size
    num_epochs = cfg.num_epochs
    device = xm.xla_device()
    rank = xm.get_local_ordinal()

    # build datasets
    train_dataset, train_loader, train_sampler, _, val_loader, _ = build_datasets(cfg, device)
    xm.rendezvous("loaded dataset")
    xm.master_print(f"\n=== dataset ===\n{pprint.pformat(train_dataset)}\n")

    # build model and loss
    model = build_fsdp_vit_model(cfg, device)
    loss_fn = torch.nn.CrossEntropyLoss()
    xm.rendezvous("loaded model")
    xm.master_print(f"\n=== model ===\n{pprint.pformat(model)}\n")

    parameters = list(model.parameters())
    xm.master_print(f"per-TPU (sharded) parameter num: {sum(p.numel() for p in parameters)}")

    # build optimizer and scheduler
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = get_warmup_cosine_scheduler(
        optimizer, warmup_iteration=cfg.warmup_steps, max_iteration=len(train_dataset) // batch_size * num_epochs,
    )
    xm.rendezvous("loaded optimizer")
    xm.master_print(f"\n=== optimizer ===\n{pprint.pformat(optimizer)}\n")

    # resume training from previous checkpoint (in FSDP, each rank saves and loads its own checkpoint)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    if cfg.resume_epoch > 0:
        ckpt_path = os.path.join(cfg.ckpt_dir, f"epoch_{cfg.resume_epoch}_rank_{rank}.ckpt")
        load_ckpt(ckpt_path, model, optimizer, lr_scheduler)

    smoothed_loss = SmoothedValue(window_size=5)
    smoothed_time = SmoothedValue(window_size=5)
    xm.rendezvous("training begins")
    xm.master_print("training begins (the first few iterations are very slow due to compilation)")
    # accuracy, _, _ = eval_on_val(val_loader, model, device)
    # xm.master_print(f"accuracy on val: {accuracy:.4f}")
    for epoch in range(cfg.resume_epoch + 1, num_epochs + 1):
        xm.master_print(f"starting epoch {epoch}")
        time_epoch_b = time_step_b = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        for step, (data, target) in enumerate(train_loader):
            # 1. forward pass
            output = model(data)
            loss = loss_fn(output, target)

            # 2. backward pass and gradient clipping (if specified via a positive cfg.clip_grad_norm)
            loss.backward()
            # if not cfg.run_without_fsdp:
            #     # !!! DO NOT reduce (sharded) gradients across XLA devices when using FSDP
            #     # !!! use `model.clip_grad_norm_` to clip based on full (instead of sharded) gradient's norm
            #     if cfg.clip_grad_norm > 0:
            #         model.clip_grad_norm_(cfg.clip_grad_norm)
            # else:
            #     # the baseline setting without FSDP (as a comparison)
            #     xm.reduce_gradients(optimizer)
            #     if cfg.clip_grad_norm > 0:
            #         torch.nn.utils.clip_grad_norm_(parameters, cfg.clip_grad_norm)

            # 3. parameter update
            # optimizer.step()
            xm.optimizer_step(optimizer)
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)  # note: set_to_none saves more memory

            # 4. logging
            t_new = time.time()
            time_step_elapsed, time_step_b = t_new - time_step_b, t_new
            smoothed_time.update(time_step_elapsed, batch_size=1)
            is_first_iter = epoch == cfg.resume_epoch + 1 and step == 0
            if is_first_iter or (step + 1) % cfg.log_step_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                xm.add_step_closure(
                    run_logging, args=(epoch, step, smoothed_loss, smoothed_time, loss, lr, device),
                )

        time_epoch_elapsed = time.time() - time_epoch_b
        xm.master_print(f"epoch {epoch} done ({time_epoch_elapsed:.2f} sec)")

        # save checkpoint
        if epoch % cfg.ckpt_epoch_interval == 0 or epoch == num_epochs:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"epoch_{epoch}_rank_{rank}.ckpt")
            save_ckpt(ckpt_path, model, optimizer, lr_scheduler, master_only=True)
        # evaluate on val
        if epoch % cfg.test_epoch_interval == 0 or epoch == num_epochs:
            accuracy, _, _ = eval_on_val(val_loader, model, device)
            xm.master_print(f"accuracy on val: {accuracy:.4f}")

@torch.no_grad
def eval_on_val(val_loader, model, device):
    model.eval()
    local_correct = torch.zeros(1, dtype=torch.long, device=device)
    local_total = 0
    for data, target in val_loader:
        output = model(data)
        pred = output.argmax(dim=-1)
        local_correct.add_(pred.eq(target.view_as(pred)).sum())
        local_total += target.size(0)
    correct = xm.mesh_reduce("local_correct", local_correct.item(), sum)
    total = xm.mesh_reduce("local_total", local_total, sum)
    accuracy = correct / total
    return accuracy, correct, total


def main(device_id, cfg):
    xm.master_print(f"\n=== cfg ===\n{pprint.pformat(cfg)}\n")
    train(cfg)
    xm.master_print("training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/datasets/imagenet-1k")
    parser.add_argument("--fake_data", action="store_true", dest="fake_data")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_dir", type=str, default="/tmp/vit_fsdp")
    parser.add_argument("--resume_epoch", type=int, default=0)
    parser.add_argument("--ckpt_epoch_interval", type=int, default=50)
    parser.add_argument("--test_epoch_interval", type=int, default=1)
    parser.add_argument("--log_step_interval", type=int, default=40)

    # the default model hyperparameters is a ViT with 10 billion parameters
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--embed_dim", type=int, default=5120)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=32)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--pos_dropout", type=float, default=0.0)
    parser.add_argument("--att_dropout", type=float, default=0.0)
    parser.add_argument("--mlp_dropout", type=float, default=0.0)
    parser.add_argument("--num_classes", type=int, default=1000)

    # these default learning hyperparameters are not necessarily optimal
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--no_grad_ckpt", action="store_false", dest="grad_ckpt")
    parser.add_argument("--no_reshard_after_forward", action="store_false", dest="reshard_after_forward")
    parser.add_argument("--flatten_parameters", action="store_true", dest="flatten_parameters")
    parser.add_argument("--run_without_fsdp", action="store_true", dest="run_without_fsdp")
    parser.add_argument("--shard_on_cpu", action="store_true", dest="shard_on_cpu")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_nested_fsdp", action="store_true", dest="use_nested_fsdp")

    cfg = parser.parse_args()
    # print(cfg)
    # assert False
    xmp.spawn(main, args=(cfg,))


    # python3 -u run_vit_training.py   --data_dir /home/lumine7x/disk3-400/imagenet  --ckpt_dir ${SAVE_DIR}   --image_size 224   --patch_size 16   --embed_dim 256   --mlp_ratio 4.0   --num_heads 8   --num_blocks 12   --batch_size 1024   --num_epochs 300   --lr 1e-3   --weight_decay 0.05   --clip_grad_norm 10   --warmup_steps 5000   --log_step_interval 40   --shard_on_cpu
