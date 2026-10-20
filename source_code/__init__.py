# Copyright 2026 Alibaba Group and the Qwen Team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

# Modelos
from .models.standalone_qwen3 import (
    StandaloneQwen3Config,
    StandaloneRMSNorm,
    StandaloneRotaryEmbedding,
    StandaloneQwen3Attention,
    StandaloneQwen3MLP,
    StandaloneQwen3DecoderLayer,
    StandaloneQwen3Model,
    StandaloneQwen3ForCausalLM
)
from .models.pruned_qwen3 import PrunedQwen3ForCausalLM
from .models.lora import LoRALinear, apply_lora_to_model
from .models.medusa_heads import MedusaResidualBlock, MedusaHead, Qwen3MedusaModel

# Capas de alto calibre
from .layers.block import Qwen3DecoderBlock
from .layers.attention import (
    GroupedQueryAttention,
    QKNorm,
    TiledOnlineSoftmax,
    ChunkedPrefillEngine,
    CrossLayerAttention,
    MultiHeadLatentAttention
)
from .layers.feed_forward import (
    SwiGLU,
    GeGLU,
    Qwen3SparseMoeBlock,
    Qwen3MoeRouter,
    ExpertSaliencyAnalyzer
)
from .layers.normalization import RMSNorm, LayerNorm, SmoothQuantScaler
from .layers.embeddings import (
    RotaryEmbedding,
    SinusoidalEmbedding,
    Qwen3YaRNEmbedding,
    apply_rotary_pos_emb,
    rotate_half
)

# Inferencia, empaquetado y optimización
from .inference.kv_cache import Qwen3KVCache, AttentionSinkKVCache
from .inference.copy_suppression import CopySuppressionController
from .inference.sampling import sample_next_token, apply_repetition_penalty
from .inference.quantization import Int8Linear, quantize_model_to_int8
from .inference.generation import Qwen3Generator
from .inference.steered_generation import SteeredGenerator
from .inference.logit_lens import LogitLensAnalyzer
from .inference.fim_engine import FIMEngine
from .inference.speculative_decoding import SpeculativeDecoder
from .inference.reasoning_engine import Qwen3ReasoningEngine
from .inference.tool_calling import Qwen3ToolCallingEngine
from .inference.paged_attention import PagedKVCacheManager, PhysicalBlock
from .inference.awq_quantizer import AWQChannelProtector, AWQLinear
from .inference.grammar_sampler import TokenHealingProcessor, JSONGrammarMasker
from .inference.quantized_kv_cache import QuantizedKVCache
from .inference.continuous_batching import (
    ContinuousBatchingScheduler,
    SequenceRequest,
    RequestState
)
from .inference.jit_optimizer import JITOptimizer
from .inference.h2o_cache import H2OKVCache

# Loaders y Exporters
from .loaders.weight_loader import SafetensorsLoader
from .loaders.tokenization_qwen3 import Qwen2Tokenizer, Qwen3TokenizerNative
from .loaders.gguf_exporter import Qwen3GGUFExporter, map_qwen3_tensor_name, quantize_to_q8_0

# Entrenamiento y Alineación
from .training.reasoning_loss import GRPOLoss, DPOLoss, ReasoningFormatReward

# Multimodal (Audio, Visión y Video)
from .multimodal import (
    Qwen3ASRConfig,
    Qwen3ASRAudioEncoderConfig,
    Qwen3ASRAudioEncoder,
    Qwen3ASRForConditionalGeneration,
    Qwen3VisionConfig,
    Qwen3VLConfig,
    Qwen3VisionTransformer,
    PatchMerger,
    PatchEmbed,
    Qwen3VLForConditionalGeneration,
    PerceiverResampler
)

# Interpretabilidad Mecanicista y Cirugía de Modelos
from .interpretability import (
    TopKSparseAutoencoder,
    LinearProbe,
    DirectLogitAttributionAnalyzer,
    ROMEKnowledgeEditor,
    HallucinationDetector,
    ContrastiveActivationAddition,
    RefusalDirectionAnalyzer
)

__all__ = [
    # Models
    "StandaloneQwen3Config",
    "StandaloneRMSNorm",
    "StandaloneRotaryEmbedding",
    "StandaloneQwen3Attention",
    "StandaloneQwen3MLP",
    "StandaloneQwen3DecoderLayer",
    "StandaloneQwen3Model",
    "StandaloneQwen3ForCausalLM",
    "PrunedQwen3ForCausalLM",
    "LoRALinear",
    "apply_lora_to_model",
    "MedusaResidualBlock",
    "MedusaHead",
    "Qwen3MedusaModel",
    # Layers
    "Qwen3DecoderBlock",
    "GroupedQueryAttention",
    "QKNorm",
    "TiledOnlineSoftmax",
    "ChunkedPrefillEngine",
    "CrossLayerAttention",
    "MultiHeadLatentAttention",
    "SwiGLU",
    "GeGLU",
    "Qwen3SparseMoeBlock",
    "Qwen3MoeRouter",
    "ExpertSaliencyAnalyzer",
    "RMSNorm",
    "LayerNorm",
    "SmoothQuantScaler",
    "RotaryEmbedding",
    "SinusoidalEmbedding",
    "Qwen3YaRNEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
    # Inference
    "Qwen3KVCache",
    "AttentionSinkKVCache",
    "CopySuppressionController",
    "sample_next_token",
    "apply_repetition_penalty",
    "Int8Linear",
    "quantize_model_to_int8",
    "SafetensorsLoader",
    "Qwen3Generator",
    "SteeredGenerator",
    "LogitLensAnalyzer",
    "FIMEngine",
    "SpeculativeDecoder",
    "Qwen3ReasoningEngine",
    "Qwen3ToolCallingEngine",
    "PagedKVCacheManager",
    "PhysicalBlock",
    "AWQChannelProtector",
    "AWQLinear",
    "TokenHealingProcessor",
    "JSONGrammarMasker",
    "QuantizedKVCache",
    "ContinuousBatchingScheduler",
    "SequenceRequest",
    "RequestState",
    "JITOptimizer",
    "H2OKVCache",
    # Loaders & Exporters
    "Qwen2Tokenizer",
    "Qwen3TokenizerNative",
    "Qwen3GGUFExporter",
    "map_qwen3_tensor_name",
    "quantize_to_q8_0",
    # Training
    "GRPOLoss",
    "DPOLoss",
    "ReasoningFormatReward",
    # Multimodal
    "Qwen3ASRConfig",
    "Qwen3ASRAudioEncoderConfig",
    "Qwen3ASRAudioEncoder",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3VisionConfig",
    "Qwen3VLConfig",
    "Qwen3VisionTransformer",
    "PatchMerger",
    "PatchEmbed",
    "Qwen3VLForConditionalGeneration",
    "PerceiverResampler",
    # Interpretability
    "TopKSparseAutoencoder",
    "LinearProbe",
    "DirectLogitAttributionAnalyzer",
    "ROMEKnowledgeEditor",
    "HallucinationDetector",
    "ContrastiveActivationAddition",
    "RefusalDirectionAnalyzer"
]
