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
Sondas Lineares (Linear Probes) para Auditar Representaciones Capa por Capa.

Permite evaluar si una propiedad semántica o sintáctica (ej. idioma, sentimiento,
entidad geográfica, modo de razonamiento) está codificada linealmente en el flujo
residual de una capa específica de Qwen3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProbe(nn.Module):
    """
    Sonda lineal clasificadora para una capa de representación latente.
    """
    def __init__(self, hidden_size: int = 2048, num_classes: int = 2):
        super().__init__()
        self.probe = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] o [batch, hidden_size]
        Returns:
            logits: [..., num_classes]
        """
        return self.probe(hidden_states)

    def predict_proba(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.forward(hidden_states)
        return F.softmax(logits, dim=-1)


__all__ = ["LinearProbe"]
