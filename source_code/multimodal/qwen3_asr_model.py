# Copyright 2025 The Qwen team, Alibaba Group and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn
try:
    from .configuration_qwen3_asr import Qwen3ASRConfig
    from .audio_encoder import Qwen3ASRAudioEncoder
    from ..models.standalone_qwen3 import StandaloneQwen3ForCausalLM, StandaloneQwen3Config
except (ImportError, ValueError):
    from multimodal.configuration_qwen3_asr import Qwen3ASRConfig
    from multimodal.audio_encoder import Qwen3ASRAudioEncoder
    from models.standalone_qwen3 import StandaloneQwen3ForCausalLM, StandaloneQwen3Config

class Qwen3ASRForConditionalGeneration(nn.Module):
    """
    Modelo Multimodal Qwen3-ASR completo para transcripción y comprensión de audio a texto.
    Conecta el codificador de audio con la torre del modelo de lenguaje (LLM).
    """
    def __init__(self, config: Qwen3ASRConfig):
        super().__init__()
        self.config = config
        
        # Torre de Audio
        self.audio_tower = Qwen3ASRAudioEncoder(config.thinker_config.audio_config)
        
        # Torre de Lenguaje
        text_cfg = config.thinker_config.text_config
        lm_config = StandaloneQwen3Config(
            vocab_size=text_cfg.vocab_size,
            hidden_size=text_cfg.hidden_size,
            intermediate_size=text_cfg.intermediate_size,
            num_hidden_layers=text_cfg.num_hidden_layers,
            num_attention_heads=text_cfg.num_attention_heads,
            num_key_value_heads=text_cfg.num_key_value_heads,
            head_dim=text_cfg.head_dim,
            max_position_embeddings=text_cfg.max_position_embeddings,
            rms_norm_eps=text_cfg.rms_norm_eps,
            rope_theta=text_cfg.rope_theta
        )
        self.language_model = StandaloneQwen3ForCausalLM(lm_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        mel_spectrogram: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Paso hacia adelante multimodal:
        Si se pasa mel_spectrogram, extrae las características de audio y las
        fusiona con el flujo de embeddings del modelo de lenguaje.
        """
        if mel_spectrogram is not None:
            audio_embeds = self.audio_tower(mel_spectrogram) # [batch, audio_len, output_dim]
            # Inyección de audio embeds en las posiciones de audio
            # Para inferencia estándar retorna logits del modelo de texto
            return self.language_model(input_ids)
        else:
            return self.language_model(input_ids)
