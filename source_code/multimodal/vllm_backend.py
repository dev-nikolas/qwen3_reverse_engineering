# Copyright 2025 The Qwen team, Alibaba Group and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

"""
Implementación de soporte para motores de alto rendimiento vLLM.
Carga dinámica para compatibilidad con entornos que no tienen vLLM instalado.
"""

import importlib.util

VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

def get_vllm_classes():
    """Retorna las clases de vLLM de forma dinámica si están instaladas."""
    if not VLLM_AVAILABLE:
        return None
    try:
        mod = __import__("vllm.config", fromlist=["MultiModalConfig", "ModelConfig", "SpeechToTextConfig", "VllmConfig"])
        return {
            "MultiModalConfig": getattr(mod, "MultiModalConfig", None),
            "ModelConfig": getattr(mod, "ModelConfig", None),
            "SpeechToTextConfig": getattr(mod, "SpeechToTextConfig", None),
            "VllmConfig": getattr(mod, "VllmConfig", None)
        }
    except Exception:
        return None
