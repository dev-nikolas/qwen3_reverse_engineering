# Copyright 2025 The Qwen team, Alibaba Group and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from .qk_norm import QKNorm
from ..embeddings.rotary import apply_rotary_pos_emb

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

class GroupedQueryAttention(nn.Module):
    """
    Atención por Grupos de Consultas (GQA) con QK-Norm nativo.
    En Qwen3-1.7B: 16 cabezas de Query, 8 cabezas de Key/Value, d_k = 128.
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, self.num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * head_dim, hidden_size, bias=False)

        # Módulo desacoplado de QK-Norm
        self.qk_norm = QKNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Aplicación de QK-Norm
        q, k = self.qk_norm(q, k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Rotación RoPE
        cos_u = cos.unsqueeze(1)
        sin_u = sin.unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos_u, sin_u)

        # Repetición KV para GQA
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # Matmul
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(context), weights
