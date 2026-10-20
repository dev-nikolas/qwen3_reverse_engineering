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
Configuración de Arquitectura de Visión y Multimodal para Qwen3-VL.
"""

from transformers.configuration_utils import PretrainedConfig

class Qwen3VisionConfig(PretrainedConfig):
    model_type = "qwen3_vision"

    def __init__(
        self,
        depth: int = 16,
        embed_dim: int = 1024,
        hidden_size: int = 1024,
        hidden_act: str = "quick_gelu",
        mlp_ratio: float = 4.0,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        output_dim: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.output_dim = output_dim


class Qwen3VLConfig(PretrainedConfig):
    model_type = "qwen3_vl"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vision_config is None:
            vision_config = {}
        if isinstance(vision_config, dict):
            vision_config = Qwen3VisionConfig(**vision_config)
        self.vision_config = vision_config

        self.text_config = text_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id


__all__ = ["Qwen3VisionConfig", "Qwen3VLConfig"]
