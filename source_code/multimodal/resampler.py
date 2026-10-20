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
Módulo de Remuestreo Perceiver (Perceiver Resampler) para Video y Secuencias Multi-Imagen en Qwen3-VL.

Fundamento:
En la decodificación de video (`<|video_start|>`, `<|video_end|>`), el número de frames
y parches desborda la ventana de atención del LLM.
El Perceiver Resampler utiliza un conjunto fijo de 'num_latents' consultas aprendibles
que realizan atención cruzada (Cross-Attention) sobre la secuencia visual completa,
comprimiendo miles de parches temporales en un número constante y óptimo de tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceiverCrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int = 16, head_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        self.norm_latents = nn.LayerNorm(dim, eps=1e-6)
        self.norm_context = nn.LayerNorm(context_dim, eps=1e-6)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # latents: [batch, num_latents, dim]
        # context: [batch, seq_len, context_dim]
        latents_norm = self.norm_latents(latents)
        context_norm = self.norm_context(context)

        batch, n_latents, _ = latents.shape
        _, seq_len, _ = context.shape

        q = self.to_q(latents_norm).view(batch, n_latents, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context_norm).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context_norm).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(batch, n_latents, self.num_heads * self.head_dim)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """
    Comprime secuencias visuales de longitud arbitraria en un número constante de tokens latentes.
    """
    def __init__(
        self,
        dim: int = 2048,
        context_dim: int = 1024,
        num_latents: int = 64,
        depth: int = 3,
        num_heads: int = 16,
        head_dim: int = 64
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverCrossAttention(dim, context_dim, num_heads=num_heads, head_dim=head_dim),
                nn.Sequential(
                    nn.LayerNorm(dim, eps=1e-6),
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            ])
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, visual_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_sequence: Tensor [batch, variable_tokens, context_dim]
        Returns:
            compressed_tokens: Tensor [batch, num_latents, dim]
        """
        batch = visual_sequence.shape[0]
        # Expandir latentes aprendibles para el batch
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)

        for cross_attn, mlp in self.layers:
            latents = latents + cross_attn(latents, visual_sequence)
            latents = latents + mlp(latents)

        return self.norm_out(latents)


__all__ = ["PerceiverResampler", "PerceiverCrossAttention"]
