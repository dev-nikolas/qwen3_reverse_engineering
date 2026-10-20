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
Contrastive Activation Addition (CAA) & Representation Engineering
===================================================================
Motor de control conceptual mediante adición de activaciones contrastivas (CAA).

Permite extraer vectores de dirección en el espacio latente de Qwen3 comparando
pares de prompts contrastivos (ej. Verídico vs Engañoso, Formal vs Creativo, etc.):
    v_steer = E[h_pos] - E[h_neg]

Luego, inyecta v_steer durante el forward pass en capas intermedias (ej. Capa 18)
con un multiplicador alpha para controlar el comportamiento del modelo sin fine-tuning.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Callable


class ContrastiveActivationAddition:
    """
    Controlador de conceptos e inyección de activaciones contrastivas.
    """
    def __init__(self, hidden_size: int = 2048):
        self.hidden_size = hidden_size
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.active_vector: Optional[torch.Tensor] = None
        self.active_scale: float = 1.0

    def compute_steering_vector(
        self,
        positive_activations: torch.Tensor,  # [N, hidden_size]
        negative_activations: torch.Tensor   # [N, hidden_size]
    ) -> torch.Tensor:
        """
        Calcula el vector de dirección contrastivo promedio normalizado.
        """
        mean_pos = positive_activations.mean(dim=0).float()
        mean_neg = negative_activations.mean(dim=0).float()
        direction = mean_pos - mean_neg
        norm = torch.norm(direction).clamp(min=1e-8)
        return direction / norm

    def attach_to_module(
        self,
        target_module: nn.Module,
        steering_vector: torch.Tensor,
        scale: float = 1.5
    ):
        """
        Instala un forward hook en el módulo destino (ej. DecoderBlock o MLP)
        que suma el vector de dirección al residual stream.
        """
        self.detach()  # Limpiar hooks previos
        self.active_vector = steering_vector.clone().detach()
        self.active_scale = scale

        def hook_fn(module, input_args, output):
            # Si el output es una tupla (output, kv_cache), modificamos el primer elemento
            if isinstance(output, tuple):
                h = output[0]
                steered_h = h + (self.active_scale * self.active_vector.to(h.device, dtype=h.dtype))
                return (steered_h,) + output[1:]
            else:
                return output + (self.active_scale * self.active_vector.to(output.device, dtype=output.dtype))

        self.hook_handle = target_module.register_forward_hook(hook_fn)

    def detach(self):
        """Remueve el hook activo si existe."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            self.active_vector = None
