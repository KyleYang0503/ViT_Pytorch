# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:09:03 2022

@author: Kyle
"""

import torch.nn as nn
import torch
from functools import partial



# drop_path參考pytorch官方碼
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Patch_embedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_cannels=3, embedding_dim=768, norm_layer=None):
        super(Patch_embedding, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.partition = nn.Conv2d(in_cannels, embedding_dim, patch_size, stride=patch_size)
        self.norm = norm_layer(embedding_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.partition(x)
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)  # 把cannel移到最後 結構 [batch,HxW,C]
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 輸入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_, hidden_=None, output_=None, activation=nn.GELU, drop=0):
        super(MLP, self).__init__()
        output_ = output_ or input_
        hidden_ = hidden_ or input_
        self.fc1 = nn.Linear(input_, hidden_)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_, output_)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Transformer_encoder(nn.Module):
    def __init__(self, dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 activation=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Transformer_encoder, self).__init__()
        self.norm1 = norm_layer(dim)
        self.mult_attention = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.MLP_ = MLP(input_=dim, hidden_=mlp_hidden_dim, activation=activation, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.mult_attention(self.norm1(x)))
        x = x + self.drop_path(self.MLP_(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embedding_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=Patch_embedding, norm_layer=None,
                 act_layer=None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = embedding_dim
        self.embed_dim = embedding_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.Patch_embedding = embed_layer(img_size=img_size, patch_size=patch_size, in_cannels=in_c
                                           , embedding_dim=embedding_dim)
        num_patches = self.Patch_embedding.num_patches
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embedding_dim))

        self.Transformer_encoder_ = nn.Sequential(*[
            Transformer_encoder(dim=embedding_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                attn_drop_ratio=attn_drop_ratio, drop_path_ratio=0,
                                norm_layer=norm_layer, activation=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embedding_dim)

        self.has_logits = False
        self.pre_logits = nn.Identity()

        # 分類部分head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # init

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        self.apply(_init_vit_weights)

    def Linear_Proj(self, x):
        x = self.Patch_embedding(x)
        # [1,196,768] -> [B,H*W,C]
        class_token = self.class_token.expand(x.shape[0], -1, -1)  # expand把class_token 形狀弄得跟batch一樣
        x = torch.cat((class_token, x), dim=1)  # [1,197,768]

        x = self.pos_drop(x + self.pos_embed)
        x = self.Transformer_encoder_(x)
        x = self.norm(x)

        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.Linear_Proj(x)
        x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

_input = torch.ones((1, 3, 224, 224))

t = VisionTransformer()
result = t(_input)
print(result.shape)
