from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import lru_cache
import math

class AdaptiveModalityFusion(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=1, dropout=0.0):
        super(AdaptiveModalityFusion, self).__init__()
        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels, l_in_channels, key_channels, value_channels, num_heads=num_heads)
        self.lang_image_att = SpatialLanguageImageAttention(l_in_channels, v_in_channels, key_channels, value_channels, num_heads=num_heads)
        
        self.vis_project = nn.Sequential(
            nn.Conv1d(dim, dim, 1, 1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.lang_to_vis_proj = nn.Conv1d(value_channels, value_channels, 1, 1)
        self.modality_assessment = nn.Sequential(
            nn.Linear(dim + l_in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(value_channels * 2, value_channels, 1, 1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.spatial_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, l, l_mask):
        B, HW, C = x.shape
        
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)
        
        lang_att = self.image_lang_att(x, l, l_mask)  # (B, H*W, value_channels)
        lang_att = lang_att.permute(0, 2, 1)  # (B, value_channels, H*W)
        
        lang = l.permute(0, 2, 1)  # (B, N_l, l_in_channels)
        vis_att = self.lang_image_att(lang, x, l_mask)  # (B, N_l, value_channels)
        vis_att = vis_att.permute(0, 2, 1)  # (B, value_channels, N_l)
        vis_att = self.lang_to_vis_proj(vis_att)
        vis_att = F.interpolate(vis_att, size=lang_att.size(-1), mode='nearest')
        
        vis_global = torch.mean(x, dim=1)  # (B, dim)
        lang_global = torch.mean(l * l_mask.permute(0, 2, 1), dim=2)  # (B, l_in_channels)

        modality_input = torch.cat([vis_global, lang_global], dim=1)  # (B, dim+l_in_channels)
        modality_weights = self.modality_assessment(modality_input)  # (B, 2)
        
        vis_weight = modality_weights[:, 0].view(B, 1, 1)
        lang_weight = modality_weights[:, 1].view(B, 1, 1)
        
        weighted_lang_att = lang_att * lang_weight
        weighted_vis_att = vis_att * vis_weight
        
        mm = torch.cat([weighted_lang_att, weighted_vis_att], dim=1)
        mm = self.fusion_layer(mm).permute(0, 2, 1)  # (B, H*W, C)

        gate_input = torch.cat([x, mm], dim=-1)
        gate = self.spatial_gate(gate_input)  # (B, H*W, 1)
        output = x + gate * mm

        return output

class SpatialLanguageImageAttention(nn.Module):
    def __init__(self, l_in_channels, v_in_channels, key_channels, value_channels, num_heads=1):
        super(SpatialLanguageImageAttention, self).__init__()
        self.l_in_channels = l_in_channels
        self.v_in_channels = v_in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads

        self.f_query = nn.Sequential(
            nn.Conv1d(l_in_channels, key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(key_channels),
        )
        self.f_key = nn.Sequential(
            nn.Conv1d(v_in_channels, key_channels, kernel_size=1, stride=1),
        )
        self.f_value = nn.Sequential(
            nn.Conv1d(v_in_channels, value_channels, kernel_size=1, stride=1),
        )
        self.W = nn.Sequential(
            nn.Conv1d(value_channels, value_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(value_channels),
        )

    def forward(self, l, x, l_mask):
        # l: (B, N_l, l_in_channels)
        # x: (B, H*W, v_in_channels)
        # l_mask: (B, N_l, 1)

        B, N_l = l.size(0), l.size(1)
        B, HW = x.size(0), x.size(1)
        l = l.permute(0, 2, 1)  # (B, l_in_channels, N_l)
        x = x.permute(0, 2, 1)  # (B, v_in_channels, HW)

        query = self.f_query(l)  # (B, key_channels, N_l)
        query = query.permute(0, 2, 1)  # (B, N_l, key_channels)

        key = self.f_key(x)  # (B, key_channels, HW)
        value = self.f_value(x)  # (B, value_channels, HW)

        query = query.reshape(B, N_l, self.num_heads, self.key_channels // self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, self.num_heads, self.key_channels // self.num_heads, HW)
        value = value.reshape(B, self.num_heads, self.value_channels // self.num_heads, HW)

        l_mask = l_mask.unsqueeze(1)

        sim_map = torch.matmul(query, key)  # (B, num_heads, N_l, HW)
        sim_map = (self.key_channels ** -0.5) * sim_map
        sim_map = sim_map + (1e4 * l_mask - 1e4)

        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, N_l, HW)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, N_l, value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, N_l, self.value_channels)  # (B, N_l, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, N_l)
        out = self.W(out)  # (B, value_channels, N_l)
        out = out.permute(0, 2, 1)  # (B, N_l, value_channels)

        return out

class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, value_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)

        return out

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B Wh*Ww C
        x = self.norm(x)
        return x, Wh, Ww


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])

        self.downsample = downsample(dim=dim) if downsample else None

    def forward(self, x, H, W):
        """x: B, H*W, C"""
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)

        if self.downsample:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W

class MultiModalSwinTransformer(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 out_indices=(0, 1, 2, 3),
                 fusion_drop=0.0,
                 l_in_channels=768):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.out_indices = out_indices
        self.l_in_channels = l_in_channels

        # ---- stem ----
        self.patch_embed = PatchEmbed(patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim,
                                      norm_layer=nn.LayerNorm)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ---- stochastic depth ----
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)

        self.mm_fusions = nn.ModuleList([
            AdaptiveModalityFusion(
                dim=int(embed_dim * 2 ** i),
                v_in_channels=int(embed_dim * 2 ** i),
                l_in_channels=self.l_in_channels,
                key_channels=int(embed_dim * 2 ** i),
                value_channels=int(embed_dim * 2 ** i),
                num_heads=1,
                dropout= fusion_drop
            ) for i in range(self.num_layers)
        ])

        for i in out_indices:
            layer = nn.LayerNorm(int(embed_dim * 2 ** i))
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            ckpt = torch.load(pretrained, map_location='cpu')
            state_dict = ckpt.get('model', ckpt)
            new_dict = {}
            for k, v in state_dict.items():
                k = k[7:] if k.startswith('module.') else k
                new_dict[k] = v
            self.load_state_dict(new_dict, strict=False)
            print(f'Load Swin backbone from {pretrained}')

    def forward(self, x, l, l_mask):
        """
        x:  image  (B, 3, H, W)
        l:  language feature  (B, N_l, l_in_channels)
        l_mask: (B, N_l, 1)
        """
        x, Wh, Ww = self.patch_embed(x)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            x_out, H, W, x, Wh, Ww = self.layers[i](x, Wh, Ww)
            x_out = self.mm_fusions[i](x_out, l, l_mask)
            if i in self.out_indices:
                norm = getattr(self, f'norm{i}')
                x_out = norm(x_out)
                out = x_out.view(-1, H, W, x_out.size(-1)).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)

