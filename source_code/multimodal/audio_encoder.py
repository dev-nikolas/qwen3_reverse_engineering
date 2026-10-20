# Copyright 2025 The Qwen team, Alibaba Group and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig

class SinusoidsPositionEmbedding(nn.Module):
    """Embedding posicional sinusoidal para el codificador de audio."""
    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale

        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding requiere canales pares")

        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        positional_embedding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        self.register_buffer("positional_embedding", positional_embedding, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen, :]

class Qwen3ASRAudioAttention(nn.Module):
    """Atención Multi-Cabeza para el codificador de audio de Qwen3."""
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(context)

class Qwen3ASRAudioEncoderLayer(nn.Module):
    """Capa Transformer del codificador de audio de Qwen3."""
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3ASRAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm Self Attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # Pre-norm Feed-Forward
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(F.gelu(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states

        return hidden_states

class Qwen3ASRAudioEncoder(nn.Module):
    """
    Codificador de Audio nativo en puro PyTorch para Qwen3-ASR / Qwen3-Audio.
    Procesa espectrogramas de Mel logarítmicos, reduce resolución temporal 8x mediante CNNs,
    aplica 32 capas Transformer y proyecta al espacio del modelo de lenguaje (output_dim = 3584).
    """
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        
        # Positional Embedding
        self.positional_embedding = SinusoidsPositionEmbedding(config.max_source_positions, embed_dim)

        # 3 Capas Convolucionales 2D (stride=2 cada una -> reducción total 2^3 = 8x en tiempo)
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, kernel_size=3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, kernel_size=3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, kernel_size=3, stride=2, padding=1)

        # Cálculo de dimensión de salida de la convolución en frecuencia
        freq_dim = (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        conv_out_dim = config.downsample_hidden_size * freq_dim
        self.conv_out = nn.Linear(conv_out_dim, config.d_model, bias=False)

        # Capas de Transformer Encoder
        self.layers = nn.ModuleList([
            Qwen3ASRAudioEncoderLayer(config) for _ in range(config.encoder_layers)
        ])

        # Proyecciones de salida
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        mel_spectrogram: [batch, num_mel_bins, time_frames]
        """
        # Añadir canal de imagen: [batch, 1, num_mel_bins, time_frames]
        x = mel_spectrogram.unsqueeze(1)
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))

        # [batch, channels, freq, time] -> [batch, time, channels * freq]
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x = self.conv_out(x)

        # Positional embedding
        pos_emb = self.positional_embedding(x.size(1)).unsqueeze(0).to(x.dtype)
        x = x + pos_emb

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Proyección de salida hacia el LLM
        x = self.ln_post(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x
