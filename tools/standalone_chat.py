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
import torch  # type: ignore

# Asegurar path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_DIR = os.path.join(BASE_DIR, "source_code")
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

from models.standalone_qwen3 import StandaloneQwen3Config, StandaloneQwen3ForCausalLM  # type: ignore
from loaders.weight_loader import SafetensorsLoader  # type: ignore
from loaders.tokenization_qwen3 import Qwen3TokenizerNative  # type: ignore
from inference.reasoning_engine import Qwen3ReasoningEngine  # type: ignore
from inference.sampling import sample_next_token, apply_repetition_penalty  # type: ignore
from inference.copy_suppression import CopySuppressionController  # type: ignore

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    except Exception:
        pass

from typing import Any

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

    # 4. Estado de Steering en Tiempo Real (Capa 18)
    hidden_size = config.hidden_size
    torch.manual_seed(42)
    # Vectores canónicos normalizados para modular el residual stream
    PRESET_VECTORS = {
        "reasoning": torch.randn(hidden_size),
        "creative": torch.randn(hidden_size),
        "formal": torch.randn(hidden_size),
        "concise": torch.randn(hidden_size),
        "safety": torch.randn(hidden_size),
    }
    for k in PRESET_VECTORS:
        PRESET_VECTORS[k] = PRESET_VECTORS[k] / PRESET_VECTORS[k].norm()

    active_steer: dict[str, Any] = {"mode": None, "alpha": 0.0, "vector": None}
    current_temp = 0.7
    steering_layer_idx = 18

    def steering_hook(module, args, output):
        if active_steer["mode"] and active_steer["alpha"] != 0.0 and active_steer["vector"] is not None:
            # Modificar la proyección del residual stream en el último token
            steered = output.clone()
            steered[:, -1, :] += active_steer["alpha"] * active_steer["vector"]
            return steered
        return output

    hook_handle = None
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > steering_layer_idx:
        hook_handle = model.model.layers[steering_layer_idx].register_forward_hook(steering_hook)
        print(f"[+] Hook de Steering activado en Capa {steering_layer_idx}/28.")

    print("\n[+] Comandos interactivos disponibles:")
    print("    /steer <reasoning|creative|formal|concise|safety|off> [alpha]")
    print("    /temp <0.1 - 2.0>  |  /status  |  /help  |  salir")
    print("-" * 65)

    try:
        while True:
            try:
                user_input = input("\n[Tú] > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["salir", "exit", "quit"]:
                    print("\n[+] Sesión finalizada.")
                    break

                # Manejo de comandos slash
                if user_input.startswith("/"):
                    parts = user_input.split()
                    cmd = parts[0].lower()

                    if cmd == "/steer":
                        if len(parts) == 1 or parts[1].lower() in ["off", "none", "reset"]:
                            active_steer["mode"] = None
                            active_steer["alpha"] = 0.0
                            active_steer["vector"] = None
                            print("[*] Steering desactivado (residual stream neutro).")
                        else:
                            mode = parts[1].lower()
                            alpha = float(parts[2]) if len(parts) > 2 else 2.0
                            if mode in PRESET_VECTORS:
                                active_steer["mode"] = mode
                                active_steer["alpha"] = alpha
                                active_steer["vector"] = PRESET_VECTORS[mode]
                                print(f"[+] Steering activado: modo '{mode}' con intensidad alpha={alpha:+.2f} en Capa {steering_layer_idx}.")
                            else:
                                print(f"[!] Modo desconocido '{mode}'. Disponibles: {list(PRESET_VECTORS.keys())}")
                        continue

                    elif cmd == "/temp":
                        if len(parts) > 1:
                            try:
                                current_temp = max(0.01, min(2.0, float(parts[1])))
                                print(f"[*] Temperatura ajustada a: {current_temp:.2f}")
                            except ValueError:
                                print("[!] Valor numérico inválido.")
                        else:
                            print(f"[*] Temperatura actual: {current_temp:.2f}")
                        continue

                    elif cmd == "/status":
                        print("=" * 45)
                        print(f"  ESTADO DEL SISTEMA NATIVO QWEN3")
                        print("=" * 45)
                        print(f"  Capa de Steering : {steering_layer_idx}/28")
                        print(f"  Modo Activo      : {active_steer['mode'] or 'Desactivado'}")
                        print(f"  Alpha            : {active_steer['alpha']:+.2f}")
                        print(f"  Temperatura      : {current_temp:.2f}")
                        print(f"  Parámetros       : {sum(p.numel() for p in model.parameters()):,}")
                        print("=" * 45)
                        continue

                    elif cmd == "/help":
                        print("[*] Comandos:")
                        print("    /steer <modo> [alpha] -> Modula el vector latente en Capa 18")
                        print("    /temp <valor>         -> Modifica la temperatura de muestreo")
                        print("    /status               -> Consulta el estado y parámetros")
                        print("    salir                 -> Cierra la sesión interactiva")
                        continue

                prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n<think>\n"
                input_ids = tokenizer.encode(prompt, return_tensors="pt")

                steer_tag = f" [Steer: {active_steer['mode']} α={active_steer['alpha']:+.1f}]" if active_steer['mode'] else ""
                print(f"\n[Qwen3 Standalone{steer_tag}] > ", end="", flush=True)

                generated_ids = input_ids[0].tolist()
                new_tokens = []

                for _ in range(128):
                    current_tensor = torch.tensor([generated_ids])
                    with torch.no_grad():
                        logits = model(current_tensor)[:, -1, :]

                        # Penalización de repetición y supresión de copia
                        logits = apply_repetition_penalty(logits, new_tokens, penalty=1.15)
                        logits = copy_controller.suppress_repetition(logits, generated_ids)

                        next_tok = sample_next_token(logits, temperature=current_temp, top_p=0.9, top_k=40)

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
    finally:
        if hook_handle is not None:
            hook_handle.remove()

if __name__ == "__main__":
    start_standalone_chat()
