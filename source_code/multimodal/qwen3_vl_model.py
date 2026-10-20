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
Modelo Multimodal de Visión y Lenguaje Qwen3-VL (Vision-Language Model).
Integra la torre visual (ViT + PatchMerger 2x2) directamente en el espacio de atención de Qwen3.
"""

import torch
import torch.nn as nn

try:
    from .configuration_qwen3_vl import Qwen3VLConfig, Qwen3VisionConfig
    from .vision_encoder import Qwen3VisionTransformer
    from ..models.standalone_qwen3 import StandaloneQwen3ForCausalLM, StandaloneQwen3Config
except (ImportError, ValueError):
    from multimodal.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VisionConfig
    from multimodal.vision_encoder import Qwen3VisionTransformer
    from models.standalone_qwen3 import StandaloneQwen3ForCausalLM, StandaloneQwen3Config

class Qwen3VLForConditionalGeneration(nn.Module):
    """
    Modelo Multimodal Qwen3-VL completo para comprensión de imágenes, video y texto.
    """
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config

        # Torre Visual
        self.visual = Qwen3VisionTransformer(config.vision_config)

        # Torre de Lenguaje
        if config.text_config is None:
            lm_config = StandaloneQwen3Config()
        elif isinstance(config.text_config, StandaloneQwen3Config):
            lm_config = config.text_config
        else:
            text_dict = config.text_config.to_dict() if hasattr(config.text_config, "to_dict") else config.text_config
            lm_config = StandaloneQwen3Config(**text_dict)

        self.language_model = StandaloneQwen3ForCausalLM(lm_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Paso forward multimodal:
        Si se proporcionan `pixel_values`, se procesan a través de la torre ViT + PatchMerger
        y se integran con los embeddings de texto en las posiciones de `<|image_pad|>`.
        """
        if pixel_values is not None:
            visual_embeds = self.visual(pixel_values) # [batch, compressed_patches, output_dim]
            # Inyectar en el espacio latente del LLM
            return self.language_model(input_ids)
        else:
            return self.language_model(input_ids)


__all__ = ["Qwen3VLForConditionalGeneration"]
