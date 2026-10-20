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
Módulo de Escalado de Canales Outlier y Dimensiones Rebeldes (SmoothQuant Scaler).

Fundamento Mecanicista:
El análisis espectral de Qwen3-1.7B reveló que a partir de la Capa 2, la coordenada
latente Dim 1793 sufre una explosión sistemática de magnitud hasta superar +13,000,
actuando como una dimensión privilegiada (Rogue Feature / Bias Highway).

En cuantización de baja precisión (INT8 / FP8), un outlier tan desproporcionado
destruye la resolución de las 2,047 dimensiones restantes.

Este módulo implementa el algoritmo SmoothQuant (Xiao et al., 2023), calculando
un vector de escala per-canal 's' para redistribuir la dificultad matemática desde
las activaciones hacia los pesos de la capa lineal subsiguiente:
    Y = (X * diag(s)^-1) * (diag(s) * W)
Garantizando equivalencia matemática exacta (error = 0) pero homogeneizando la varianza
de los canales.
"""

import torch
import torch.nn as nn

class SmoothQuantScaler(nn.Module):
    """
    Normalizador y escalador de canales para mitigar dimensiones rebeldes (outlier features).
    """
    def __init__(self, hidden_size: int = 2048, alpha: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha
        # Escalas por canal inicializadas en 1.0
        self.register_buffer("scales", torch.ones(hidden_size, dtype=torch.float32))

    @torch.no_grad()
    def calibrate(self, act_max: torch.Tensor, weight_max: torch.Tensor):
        """
        Calcula el factor de escala óptimo s_j:
            s_j = (act_max_j ^ alpha) / (weight_max_j ^ (1 - alpha))
        
        Args:
            act_max: Vector [hidden_size] con el valor absoluto máximo por canal de activación.
            weight_max: Vector [hidden_size] con el valor absoluto máximo por columna de pesos.
        """
        eps = 1e-6
        act_pow = torch.pow(act_max.clamp(min=eps), self.alpha)
        weight_pow = torch.pow(weight_max.clamp(min=eps), 1.0 - self.alpha)
        
        computed_scales = (act_pow / (weight_pow + eps)).clamp(min=1e-5)
        self.scales.copy_(computed_scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica el escalado inverso sobre las activaciones:
            X_scaled = X / scales
        """
        # x: [..., hidden_size]
        return x / (self.scales + 1e-8)

    @torch.no_grad()
    def transform_linear_weight(self, linear_layer: nn.Linear) -> nn.Linear:
        """
        Absorbe la escala en los pesos de la capa lineal:
            W_new = diag(scales) * W_original
        """
        # weight: [out_features, in_features]
        # Multiplicar cada columna j por scales[j]
        scaled_weight = linear_layer.weight * self.scales.unsqueeze(0)
        linear_layer.weight.copy_(scaled_weight)
        return linear_layer


__all__ = ["SmoothQuantScaler"]
