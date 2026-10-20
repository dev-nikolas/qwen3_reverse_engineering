# Copyright 2026 Alibaba Group and the Qwen Team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

"""
Consola de Chat Interactivo 100% Independiente y Nativa
========================================================
Ejecuta la inferencia de Qwen3-1.7B usando de forma exclusiva los módulos
de ingeniería inversa implementados en `source_code/`, sin requerir el runtime
de generación de Transformers.
"""

import os
import sys
import torch

# Asegurar path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_DIR = os.path.join(BASE_DIR, "source_code")
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

from models.standalone_qwen3 import StandaloneQwen3Config, StandaloneQwen3ForCausalLM
from loaders.weight_loader import SafetensorsLoader
from loaders.tokenization_qwen3 import Qwen3TokenizerNative
from inference.reasoning_engine import Qwen3ReasoningEngine
from inference.sampling import sample_next_token, apply_repetition_penalty
from inference.copy_suppression import CopySuppressionController

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = os.path.join(BASE_DIR, "model")

def start_standalone_chat():
    print("=" * 65)
    print("  QWEN3-1.7B — CHAT STANDALONE NATIVO (100% PYTORCH PURO)")
    print("=" * 65)
    print(f"[*] Inicializando arquitectura desde: {SOURCE_DIR}")
    
    # 1. Configuración nativa
    config = StandaloneQwen3Config(
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6
    )
    
    print("[*] Instanciando StandaloneQwen3ForCausalLM...")
    model = StandaloneQwen3ForCausalLM(config)
    
    # 2. Cargar pesos Safetensors si existen localmente
    loader = SafetensorsLoader(MODEL_DIR)
    if loader.shard_files:
        print(f"[*] Inyectando pesos Safetensors ({len(loader.shard_files)} shards)...")
        try:
            missing, unexpected = loader.load_into_model(model, target_dtype=torch.float32, strict=False)
            print(f"[+] Pesos cargados con éxito en RAM. (Claves faltantes: {len(missing)})")
        except Exception as e:
            print(f"[!] Aviso al cargar pesos: {e}. Inicializado con parámetros de ingeniería inversa.")
    else:
        print("[!] No se encontraron shards Safetensors en model/. Usando pesos inicializados.")

    model.eval()
    
    # 3. Tokenizador y motor de razonamiento
    tokenizer = Qwen3TokenizerNative()
    reasoning_engine = Qwen3ReasoningEngine(max_thinking_tokens=512)
    copy_controller = CopySuppressionController(dla_discount_factor=0.2)
    
    print("[+] Sistema de inferencia listo. Escribe 'salir' para finalizar.\n" + "-" * 65)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n[Tú] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("\n[+] Sesión finalizada.")
                break
                
            prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            
            print("\n[Qwen3 Standalone] > ", end="", flush=True)
            
            generated_ids = input_ids[0].tolist()
            new_tokens = []
            
            for _ in range(128):
                current_tensor = torch.tensor([generated_ids])
                with torch.no_grad():
                    logits = model(current_tensor)[:, -1, :]
                    
                    # Penalización de repetición y supresión de copia
                    logits = apply_repetition_penalty(logits, new_tokens, penalty=1.15)
                    logits = copy_controller.suppress_repetition(logits, generated_ids)
                    
                    next_tok = sample_next_token(logits, temperature=0.7, top_p=0.9, top_k=40)
                    
                    if next_tok in [151643, 151645]:  # <|im_end|> o <|endoftext|>
                        break
                        
                    token_str = tokenizer.decode([next_tok])
                    print(token_str, end="", flush=True)
                    
                    generated_ids.append(next_tok)
                    new_tokens.append(next_tok)
                    
            print()
            
        except KeyboardInterrupt:
            print("\nInterrupción detectada. Cerrando sesión.")
            break

if __name__ == "__main__":
    start_standalone_chat()
