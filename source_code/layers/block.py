# Copyright 2025 The Qwen team, Alibaba Group and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn
from .attention.gqa import GroupedQueryAttention
from .feed_forward.swiglu import SwiGLU
from .normalization.rmsnorm import RMSNorm

class Qwen3DecoderBlock(nn.Module):
    """
    Bloque decodificador Transformer de Qwen3 de alto calibre.
    Integra normalización previa (Pre-Norm) con RMSNorm, atención GQA con QK-Norm,
    red SwiGLU no lineal y doble conexión residual.
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        intermediate_size: int = 6144,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        # 1. Atención residual
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            layer_idx=layer_idx
        )
        
        # 2. Feed-Forward residual
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = SwiGLU(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Pre-norm Self-Attention + Residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, _ = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + attn_out

        # Pre-norm MLP + Residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + mlp_out

        return hidden_states
