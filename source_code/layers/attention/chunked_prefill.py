# Copyright 2025 The Qwen Team and Alibaba Group.
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
Chunked Prefill & Tiled Attention Engine for Ultra-Long Contexts
================================================================
Motor de atención particionada por bloques (Tiled Attention / Online Softmax)
y prefill escalonado en trozos (Chunked Prefill) para Qwen3.

Permite:
1. Online Softmax (FlashAttention-style): Cálculo de atención exacta en memoria O(1)
   evitando instanciar la matriz completa [seq_len, seq_len] en RAM/VRAM.
2. Chunked Prefill: Particiona prompts masivos (hasta 131,072 tokens) en ventanas
   modulares de prefijo, intercalando el llenado del KV-Cache sin saturar la memoria.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, Generator, List, Any


class TiledOnlineSoftmax:
    """
    Simulador en PyTorch puro del algoritmo de Online Softmax utilizado
    en FlashAttention para cómputo de atención sin materializar matrices N x N.
    """
    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    def compute_tiled_attention(
        self,
        q: torch.Tensor,  # [batch, heads, seq_len_q, head_dim]
        k: torch.Tensor,  # [batch, heads, seq_len_k, head_dim]
        v: torch.Tensor,  # [batch, heads, seq_len_k, head_dim]
        scale: Optional[float] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        batch, heads, q_len, dim = q.shape
        _, _, k_len, _ = k.shape
        if scale is None:
            scale = 1.0 / math.sqrt(dim)

        # Inicialización de acumuladores de salida, normalizadores y máximos locales
        out = torch.zeros_like(q)
        m = torch.full((batch, heads, q_len, 1), -float("inf"), device=q.device)
        l = torch.zeros((batch, heads, q_len, 1), device=q.device)

        # Iterar sobre bloques de la dimensión de claves/valores (KV tiles)
        for kv_start in range(0, k_len, self.block_size):
            kv_end = min(kv_start + self.block_size, k_len)
            k_tile = k[:, :, kv_start:kv_end, :]  # [B, H, kv_chunk, D]
            v_tile = v[:, :, kv_start:kv_end, :]

            # Producto escalar del bloque: Q * K_tile^T
            s_tile = torch.matmul(q, k_tile.transpose(-1, -2)) * scale  # [B, H, q_len, kv_chunk]

            # Enmascaramiento causal
            if is_causal:
                q_indices = torch.arange(q_len, device=q.device).unsqueeze(-1)
                kv_indices = torch.arange(kv_start, kv_end, device=q.device).unsqueeze(0)
                causal_mask = q_indices < kv_indices
                s_tile = s_tile.masked_fill(causal_mask, -float("inf"))

            # Paso Online Softmax
            tile_max = torch.max(s_tile, dim=-1, keepdim=True).values
            new_m = torch.maximum(m, tile_max)

            # Factores de corrección
            exp_m_diff = torch.exp(m - new_m)
            p_tile = torch.exp(s_tile - new_m)

            # Acumulador del denominador
            new_l = exp_m_diff * l + torch.sum(p_tile, dim=-1, keepdim=True)

            # Acumulador del numerador: out = exp(m - new_m) * out + P_tile * V_tile
            out = exp_m_diff * out + torch.matmul(p_tile, v_tile)

            # Actualizar m y l
            m = new_m
            l = new_l

        # Normalización final por el divisor l acumulado
        out = out / l.clamp(min=1e-8)
        return out


class ChunkedPrefillEngine:
    """
    Motor de prefill por trozos para secuencias masivas de contexto en Qwen3.
    """
    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size

    def split_prompt_into_chunks(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Divide el prompt en trozos de longitud fija."""
        seq_len = input_ids.shape[1]
        chunks = []
        for i in range(0, seq_len, self.chunk_size):
            chunks.append(input_ids[:, i : i + self.chunk_size])
        return chunks

    def execute_chunked_forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        kv_cache: Any
    ) -> torch.Tensor:
        """
        Ejecuta el forward pass de un prompt masivo alimentando el modelo
        trozo por trozo y llenando el KV-Cache sin picos de VRAM.
        """
        chunks = self.split_prompt_into_chunks(input_ids)
        last_hidden = None

        for chunk in chunks:
            # En cada trozo, el modelo atiende a su contenido y a los tokens previos en kv_cache
            if hasattr(model, "model") and hasattr(model, "lm_head"):
                hidden = model.model(chunk, kv_cache=kv_cache)
                last_hidden = hidden
            else:
                out = model(chunk, kv_cache=kv_cache)
                last_hidden = out

        return last_hidden
