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
Autoencoder Disperso Top-K (Top-K Sparse Autoencoder / SAE) para Desentrañar Polisemántica.

Fundamento Mecanicista:
En el Experimento 4 de Qwen3 descubrimos neuronas polisemánticas densas (como la neurona 1950
y la dimensión rebelde 1793). En lugar de que cada neurona represente un único concepto limpio,
las redes profundas utilizan 'superposición' (superposition) para empaquetar más características
que dimensiones disponibles.

Este Sparse Autoencoder expande las 2,048 dimensiones del flujo residual a un espacio
sobredimensionado (ej. 8,192 o 16,384 latentes) forzando que solo 'K' características estén activas,
aislando conceptos monosemánticos puros y auditables.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKSparseAutoencoder(nn.Module):
    """
    Top-K Sparse Autoencoder (SAE) para interpretabilidad del flujo residual.
    """
    def __init__(self, d_model: int = 2048, d_sae: int = 8192, k: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        # Parámetros del encoder
        self.W_enc = nn.Linear(d_model, d_sae, bias=True)
        # Parámetros del decoder
        self.W_dec = nn.Linear(d_sae, d_model, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Inicialización ortogonal o normalizada
        nn.init.kaiming_uniform_(self.W_enc.weight)
        nn.init.kaiming_uniform_(self.W_dec.weight)
        self.normalize_decoder_weights()

    @torch.no_grad()
    def normalize_decoder_weights(self):
        """Fuerza que cada columna de W_dec tenga norma L2 unitaria."""
        norms = torch.norm(self.W_dec.weight, dim=0, keepdim=True) + 1e-8
        self.W_dec.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula las activaciones latentes dispersas Top-K:
            x_cent = x - b_dec
            pre_acts = W_enc(x_cent) + b_enc
            acts = TopK(pre_acts, k)
        """
        x_cent = x - self.b_dec
        pre_acts = self.W_enc(x_cent)
        
        # Aplicar activación Top-K (preserva solo los K valores más altos)
        topk_vals, topk_indices = torch.topk(pre_acts, self.k, dim=-1)
        # Crear tensor disperso
        sparse_acts = torch.zeros_like(pre_acts)
        # Solo mantener valores positivos si se combinan con ReLU
        topk_vals = F.relu(topk_vals)
        sparse_acts.scatter_(-1, topk_indices, topk_vals)
        return sparse_acts

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruye el flujo residual a partir de las características dispersas."""
        return self.W_dec(latents) + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            reconstructed: Tensor reconstruido [batch, seq_len, d_model]
            latents: Tensor latente disperso [batch, seq_len, d_sae]
            l2_loss: Error cuadrático de reconstrucción
        """
        latents = self.encode(x)
        reconstructed = self.decode(latents)
        l2_loss = F.mse_loss(reconstructed, x)
        return reconstructed, latents, l2_loss


__all__ = ["TopKSparseAutoencoder"]
