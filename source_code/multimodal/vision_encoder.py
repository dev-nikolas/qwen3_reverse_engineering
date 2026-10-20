# Copyright 2026 Alibaba Group and the Qwen Team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Codificador de Visión (Vision Transformer) y Reductor Espacial (PatchMerger) para Qwen3-VL.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_qwen3_vl import Qwen3VisionConfig

class PatchEmbed(nn.Module):
    """
    Convierte una imagen 2D [batch, channels, H, W] en una secuencia de tokens de parches.
    """
    def __init__(self, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_channels, H, W]
        # proj: [batch, embed_dim, H/P, W/P] -> flat: [batch, embed_dim, num_patches] -> transpose: [batch, num_patches, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchMerger(nn.Module):
    """
    Fusión espacial 2x2 (Spatial Patch Merger).
    Comprime una cuadrícula de 2x2 parches visuales adyacentes en 1 único token latente,
    reduciendo la longitud de secuencia visual en un 75% antes de inyectarla al LLM.
    """
    def __init__(self, dim: int = 1024, output_dim: int = 2048, merge_size: int = 2):
        super().__init__()
        self.merge_size = merge_size
        self.ln_q = nn.LayerNorm(dim * (merge_size ** 2), eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim * (merge_size ** 2), output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x: torch.Tensor, height_patches: int, width_patches: int) -> torch.Tensor:
        # x: [batch, H_p * W_p, dim]
        batch, _, dim = x.shape
        x = x.view(batch, height_patches, width_patches, dim)
        
        # Agrupar bloques de 2x2
        h_m = height_patches // self.merge_size
        w_m = width_patches // self.merge_size
        x = x.view(batch, h_m, self.merge_size, w_m, self.merge_size, dim)
        # Permutar para juntar los canales merge: [batch, h_m, w_m, merge_size, merge_size, dim]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        # Aplanar canales espaciales combinados: [batch, h_m * w_m, dim * 4]
        x = x.view(batch, h_m * w_m, dim * (self.merge_size ** 2))
        
        x = self.ln_q(x)
        out = self.mlp(x) # [batch, compressed_patches, output_dim]
        return out


class VisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, seq_len, dim)
        return self.proj(out)


class VisionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = VisionAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VisionTransformer(nn.Module):
    """
    Torre visual ViT completa para Qwen3-VL con PatchMerger 2x2.
    """
    def __init__(self, config: Qwen3VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim
        )
        self.blocks = nn.ModuleList([
            VisionBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio
            )
            for _ in range(config.depth)
        ])
        self.merger = PatchMerger(
            dim=config.embed_dim,
            output_dim=config.output_dim,
            merge_size=config.spatial_merge_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Tensor [batch, channels, H, W]
        Returns:
            Representaciones visuales comprimidas [batch, compressed_patches, output_dim]
        """
        _, _, h, w = pixel_values.shape
        h_patches = h // self.config.patch_size
        w_patches = w // self.config.patch_size

        x = self.patch_embed(pixel_values) # [batch, h_p * w_p, embed_dim]
        for block in self.blocks:
            x = block(x)

        # Compresión 2x2
        projected = self.merger(x, h_patches, w_patches)
        return projected


__all__ = ["Qwen3VisionTransformer", "PatchMerger", "PatchEmbed"]
