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
Geometric Refusal Direction & Safety Circuit Analyzer
======================================================
Analizador mecanicista de la dirección de rechazo (Refusal Direction) en Qwen3.

Fundamento Mecanicista:
  La decisión de rechazar un prompt (por ejemplo, 'No puedo ayudarte con eso...')
  se encuentra concentrada en un único subespacio unidimensional r_hat dentro
  del residual stream en las capas medias (Capas 12 a 18).

Capacidades:
1. compute_refusal_direction: Extrae el vector unitario r_hat a partir de pares
   contrastivos (prompts dañinos/sensibles vs inocuos).
2. measure_refusal_projection: Mide la alineación escalar (h . r_hat) para predecir
   si el modelo rechazará el prompt antes de emitir tokens.
3. orthogonalize: Neutraliza el rechazo proyectando el flujo residual al subespacio
   ortogonal: h_perp = h - (h . r_hat) * r_hat.
4. amplify_refusal: Refuerza la respuesta defensiva contra técnicas de jailbreak.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class RefusalDirectionAnalyzer:
    """
    Analizador geométrico del circuito de rechazo y alineación de seguridad.
    """
    def __init__(self, hidden_size: int = 2048):
        self.hidden_size = hidden_size
        self.refusal_direction: Optional[torch.Tensor] = None

    def compute_refusal_direction(
        self,
        harmful_activations: torch.Tensor,   # [N, hidden_size]
        harmless_activations: torch.Tensor   # [N, hidden_size]
    ) -> torch.Tensor:
        """
        Calcula el vector de dirección unitario r_hat mediante la media contrastiva.
        """
        mean_harmful = harmful_activations.mean(dim=0).float()
        mean_harmless = harmless_activations.mean(dim=0).float()

        raw_direction = mean_harmful - mean_harmless
        norm = torch.norm(raw_direction).clamp(min=1e-8)
        self.refusal_direction = (raw_direction / norm).to(harmful_activations.device)
        return self.refusal_direction

    def measure_refusal_projection(self, hidden_state: torch.Tensor) -> float:
        """
        Calcula el producto escalar normalizado entre el estado oculto y la dirección de rechazo.
        """
        if self.refusal_direction is None:
            raise RuntimeError("La dirección de rechazo no ha sido calculada.")
        
        r = self.refusal_direction.to(device=hidden_state.device, dtype=torch.float32)
        proj = torch.sum(hidden_state.float() * r, dim=-1)
        if proj.ndim == 0:
            return proj.item()
        return proj

    def orthogonalize(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Elimina la componente de rechazo del vector: h_perp = h - (h . r) * r.
        """
        if self.refusal_direction is None:
            return hidden_state
            
        r = self.refusal_direction.to(device=hidden_state.device, dtype=hidden_state.dtype)
        # Producto punto sobre la última dimensión
        dot = torch.sum(hidden_state * r, dim=-1, keepdim=True)
        return hidden_state - dot * r

    def amplify_refusal(self, hidden_state: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
        """
        Refuerza la señal del vector de rechazo: h_boost = h + factor * r.
        """
        if self.refusal_direction is None:
            return hidden_state
            
        r = self.refusal_direction.to(device=hidden_state.device, dtype=hidden_state.dtype)
        return hidden_state + (factor * r)
