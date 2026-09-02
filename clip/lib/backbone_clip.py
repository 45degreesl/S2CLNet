from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict([("-1", nn.AvgPool2d(stride)),
                             ("0",
                              nn.Conv2d(inplanes,
                                        planes * self.expansion,
                                        1,
                                        stride=1,
                                        bias=False)),
                             ("1", nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

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
            nn.Linear(dim + 512, 256),
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

class MMFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Cross-modal Alignment Branch
        self.fusion = AdaptiveModalityFusion(dim,  # both the visual input and for combining, num of channels
                                             dim,  # v_in
                                             512,  # l_in # 768 for bert feature, 512 for clip
                                             dim,  # key
                                             dim,  # value
                                             num_heads=1,
                                             dropout=0.0)


    def forward(self, x, l, l_mask):
        # multi-modal fusion
        x_residual = self.fusion(x, l, l_mask)
        return x_residual, x_residual

class MMModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """
    def __init__(self,
                 layers,
                 output_dim,
                 heads,
                 input_resolution=224,
                 width=64,
                 pretrained=None):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.pretrained = pretrained

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3,
                               width // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2,
                               width // 2,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2,
                               width,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        # mutil-modal fusion
        embed_dim = width
        # self.patch_size = patch_size

        # self.MMFusion_blks = nn.ModuleList()
        # for i_blk in range(len(layers)):
        #     blk = MMFusion(embed_dim)
        #     self.MMFusion_blks.append(blk)
        self.mmfusion1 = MMFusion(256)
        self.mmfusion2 = MMFusion(512)
        self.mmfusion3 = MMFusion(1024)
        self.mmfusion4 = MMFusion(2048)

        self.normlayer1 = nn.LayerNorm(256)
        self.normlayer2 = nn.LayerNorm(512)
        self.normlayer3 = nn.LayerNorm(1024)
        self.normlayer4 = nn.LayerNorm(2048)


    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, l, l_mask):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                             (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        outs = []
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        B, D, H, W = x.shape
        x_re = x.view(B, D, H*W).permute(0, 2, 1).contiguous()
        x_out, _ = self.mmfusion1(x_re, l, l_mask)
        x_out = self.normlayer1(x_out)
        out = x_out.view(-1, H, W, D).permute(0, 3, 1, 2).contiguous()
        outs.append(out)

        x2 = self.layer2(x)
        B, D, H, W = x2.shape
        x_re = x2.view(B, D, H*W).permute(0, 2, 1).contiguous()
        x_out, _ = self.mmfusion2(x_re, l, l_mask)
        x_out = self.normlayer2(x_out)
        out = x_out.view(-1, H, W, D).permute(0, 3, 1, 2).contiguous()
        outs.append(out)

        x3 = self.layer3(x2)
        B, D, H, W = x3.shape
        x_re = x3.view(B, D, H*W).permute(0, 2, 1).contiguous()
        x_out, _ = self.mmfusion3(x_re, l, l_mask)
        x_out = self.normlayer3(x_out)
        out = x_out.view(-1, H, W, D).permute(0, 3, 1, 2).contiguous()
        outs.append(out)

        x4 = self.layer4(x3)
        B, D, H, W = x4.shape
        x_re = x4.view(B, D, H*W).permute(0, 2, 1).contiguous()
        x_out, _ = self.mmfusion4(x_re, l, l_mask)
        x_out = self.normlayer4(x_out)
        out = x_out.view(-1, H, W, D).permute(0, 3, 1, 2).contiguous()
        outs.append(out)

        return outs

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {} #new model

            for k in checkpoint.keys():
                # print(k)
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer')  # it should be nothing is misaligned
