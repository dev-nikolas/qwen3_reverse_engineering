# Copyright 2025 The Qwen team, Alibaba Group and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

try:
    from transformers.configuration_utils import PretrainedConfig
except ImportError:
    class PretrainedConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

class Qwen3ASRAudioEncoderConfig(PretrainedConfig):
    r"""
    Clase de configuración para almacenar la arquitectura del codificador de audio Qwen3ASRAudioEncoder.
    Basada en la arquitectura multimodal de Qwen3-Audio / Qwen3-ASR-1.7B.
    """
    model_type = "qwen3_asr_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        activation_dropout=0.0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        n_window_infer=400,
        conv_chunksize=500,
        downsample_hidden_size=480,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


class Qwen3ASRTextConfig(PretrainedConfig):
    r"""
    Clase de configuración para el modelo de texto del sistema Qwen3-ASR.
    """
    model_type = "qwen3_asr_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=5000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout


class Qwen3ASRThinkerConfig(PretrainedConfig):
    r"""
    Configuración que fusiona el codificador de audio y el modelo de lenguaje de pensamiento.
    """
    model_type = "qwen3_asr_thinker"
    sub_configs = {
        "audio_config": Qwen3ASRAudioEncoderConfig,
        "text_config": Qwen3ASRTextConfig,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=151646,
        audio_start_token_id=151647,
        user_token_id=872,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_token_id = user_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_token_id = audio_token_id
        self.initializer_range = initializer_range

        if isinstance(audio_config, dict):
            self.audio_config = Qwen3ASRAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            self.audio_config = Qwen3ASRAudioEncoderConfig()
        else:
            self.audio_config = audio_config

        if isinstance(text_config, dict):
            self.text_config = Qwen3ASRTextConfig(**text_config)
        elif text_config is None:
            self.text_config = Qwen3ASRTextConfig()
        else:
            self.text_config = text_config


class Qwen3ASRConfig(PretrainedConfig):
    r"""
    Configuración global de Qwen3-ASR para generación condicional de audio a texto.
    """
    model_type = "qwen3_asr"
    sub_configs = {
        "thinker_config": Qwen3ASRThinkerConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        support_languages=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if thinker_config is None:
            thinker_config = {}
        if isinstance(thinker_config, dict):
            self.thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        else:
            self.thinker_config = thinker_config
        self.support_languages = support_languages

    def get_text_config(self, decoder=False):
        return self.thinker_config.text_config
