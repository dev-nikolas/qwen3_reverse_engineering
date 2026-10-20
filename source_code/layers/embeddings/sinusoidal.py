# Copyright 2025 The Qwen team, Alibaba Group and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import torch
import torch.nn as nn

class SinusoidalEmbedding(nn.Module):
    """Embedding posicional sinusoidal utilizado en codificadores acústicos."""
    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale

        if channels % 2 != 0:
            raise ValueError("SinusoidalEmbedding requiere canales pares")

        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        positional_embedding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        self.register_buffer("positional_embedding", positional_embedding, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen, :]
