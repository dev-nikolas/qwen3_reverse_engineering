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
Módulos de Gestión de Caché de Claves y Valores (KV-Cache) para Inferencia de Qwen3.
Incluye:
1. Qwen3KVCache: Caché dinámico estándar para inferencia autoregresiva acelerada.
2. AttentionSinkKVCache: Caché basado en Sumideros de Atención (StreamingLLM), diseñado
   a partir del descubrimiento forense de que 279 de las 448 cabezas de Qwen3 son Attention Sinks
   que requieren retener permanentemente los tokens iniciales para evitar el colapso de perplejidad.
"""

import torch

class Qwen3KVCache:
    """
    Caché de Claves y Valores (KV-Cache) dinámico para Qwen3.
    Evita recalcular las activaciones de atención de tokens pasados
    durante la generación autoregresiva, acelerando la inferencia exponencialmente.
    """
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        self.key_cache: list[torch.Tensor] = [torch.empty(0, device=device, dtype=dtype) for _ in range(num_layers)]
        self.value_cache: list[torch.Tensor] = [torch.empty(0, device=device, dtype=dtype) for _ in range(num_layers)]
        self.seq_len: int = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatena los nuevos pares K, V al historial acumulado en la capa correspondiente.
        """
        if self.key_cache[layer_idx].numel() == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset(self):
        """Reinicia el caché para una nueva conversación."""
        for i in range(self.num_layers):
            self.key_cache[i] = torch.empty(0, device=self.device, dtype=self.dtype)
            self.value_cache[i] = torch.empty(0, device=self.device, dtype=self.dtype)
        self.seq_len = 0


class AttentionSinkKVCache:
    """
    Caché con Sumideros de Atención (StreamingLLM) para inferencia de flujo infinito en memoria constante.
    
    Fundamento Mecanicista:
    El censo forense de 448 cabezas en Qwen3 reveló que el 62.3% (279 cabezas) son Attention Sinks,
    descargando su exceso de probabilidad en los primeros tokens (BOS).
    
    Este caché retiene permanentemente los primeros `num_sink_tokens` (típicamente 4) y mantiene una ventana
    deslizante de los últimos `window_size` tokens recientes, descartando el medio sin causar
    la explosión de perplejidad observada al podar el token inicial.
    """
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_sink_tokens: int = 4,
        window_size: int = 512,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_sink_tokens = num_sink_tokens
        self.window_size = window_size
        self.device = device
        self.dtype = dtype

        self.key_cache: list[torch.Tensor] = [torch.empty(0, device=device, dtype=dtype) for _ in range(num_layers)]
        self.value_cache: list[torch.Tensor] = [torch.empty(0, device=device, dtype=dtype) for _ in range(num_layers)]

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Añade nuevos tokens y aplica la política de desalojo con sumideros de atención:
        Mantiene: [0 : num_sink_tokens] + [-window_size :]
        """
        if self.key_cache[layer_idx].numel() == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        current_len = self.key_cache[layer_idx].shape[2]
        max_allowed = self.num_sink_tokens + self.window_size

        if current_len > max_allowed:
            # Preservar tokens de sumidero iniciales
            sink_k = self.key_cache[layer_idx][:, :, :self.num_sink_tokens, :]
            sink_v = self.value_cache[layer_idx][:, :, :self.num_sink_tokens, :]

            # Preservar ventana rodante reciente
            recent_k = self.key_cache[layer_idx][:, :, -self.window_size:, :]
            recent_v = self.value_cache[layer_idx][:, :, -self.window_size:, :]

            self.key_cache[layer_idx] = torch.cat([sink_k, recent_k], dim=2)
            self.value_cache[layer_idx] = torch.cat([sink_v, recent_v], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset(self):
        """Reinicia el caché completo."""
        for i in range(self.num_layers):
            self.key_cache[i] = torch.empty(0, device=self.device, dtype=self.dtype)
            self.value_cache[i] = torch.empty(0, device=self.device, dtype=self.dtype)


__all__ = ["Qwen3KVCache", "AttentionSinkKVCache"]
