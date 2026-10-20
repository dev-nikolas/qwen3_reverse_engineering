# Copyright 2026 Alibaba Group and the Qwen Team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

# Audio / ASR
from .configuration_qwen3_asr import (
    Qwen3ASRConfig,
    Qwen3ASRThinkerConfig,
    Qwen3ASRTextConfig,
    Qwen3ASRAudioEncoderConfig
)
from .audio_encoder import (
    Qwen3ASRAudioEncoder,
    Qwen3ASRAudioEncoderLayer,
    Qwen3ASRAudioAttention,
    SinusoidsPositionEmbedding
)
from .qwen3_asr_model import Qwen3ASRForConditionalGeneration

# Vision / VL
from .configuration_qwen3_vl import Qwen3VisionConfig, Qwen3VLConfig
from .vision_encoder import Qwen3VisionTransformer, PatchMerger, PatchEmbed
from .qwen3_vl_model import Qwen3VLForConditionalGeneration
from .resampler import PerceiverResampler, PerceiverCrossAttention

__all__ = [
    # Audio
    "Qwen3ASRConfig",
    "Qwen3ASRThinkerConfig",
    "Qwen3ASRTextConfig",
    "Qwen3ASRAudioEncoderConfig",
    "Qwen3ASRAudioEncoder",
    "Qwen3ASRAudioEncoderLayer",
    "Qwen3ASRAudioAttention",
    "SinusoidsPositionEmbedding",
    "Qwen3ASRForConditionalGeneration",
    # Vision & Video
    "Qwen3VisionConfig",
    "Qwen3VLConfig",
    "Qwen3VisionTransformer",
    "PatchMerger",
    "PatchEmbed",
    "Qwen3VLForConditionalGeneration",
    "PerceiverResampler",
    "PerceiverCrossAttention"
]
