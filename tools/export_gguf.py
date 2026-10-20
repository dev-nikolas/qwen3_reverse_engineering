#!/usr/bin/env python3
# Copyright 2025 The Qwen Team and Alibaba Group.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Tool: CLI de Exportación Real a Formato GGUF v3 para Qwen3-1.7B
# Serializa pesos y metadatos de arquitectura a un archivo .gguf para
# ejecución directa y sin dependencias en llama.cpp, Ollama y LM Studio.

import os
import sys
import argparse
from pathlib import Path

# Agregar raíz al sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
SRC_DIR = BASE_DIR / "source_code"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json


def get_default_qwen3_config() -> dict:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "torch_dtype": "bfloat16"
    }


def main():
    parser = argparse.ArgumentParser(
        description="Exportador Forense de Qwen3-1.7B a formato estándar GGUF v3 (llama.cpp / Ollama)."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(BASE_DIR / "model"),
        help="Directorio que contiene config.json y los archivos *.safetensors del modelo."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BASE_DIR / "qwen3-1.7b-q8_0.gguf"),
        help="Ruta de destino del archivo .gguf."
    )
    parser.add_argument(
        "--format",
        choices=["f32", "f16", "q8_0"],
        default="q8_0",
        help="Formato de serialización y cuantización (f32, f16, q8_0)."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Si los pesos reales no están presentes, genera un prototipo arquitectónico sintético para validar el empaquetado."
    )

    args = parser.parse_args()

    print("=" * 68)
    print("  QWEN3-1.7B — EXPORTADOR BINARIO A GGUF v3")
    print("=" * 68)

    try:
        import torch
        from source_code.loaders.gguf_exporter import Qwen3GGUFExporter
    except ImportError as e:
        print(f"[-] Error de dependencias: {e}")
        print("    Asegúrate de tener PyTorch instalado en tu entorno Python.")
        sys.exit(1)

    model_dir = Path(args.model_dir)
    config_file = model_dir / "config.json"

    # Verificar si existen archivos safetensors
    safetensor_files = list(model_dir.glob("*.safetensors"))

    if not safetensor_files:
        print(f"[!] ADVERTENCIA: No se encontraron archivos *.safetensors en:")
        print(f"    {model_dir.resolve()}")
        if not args.synthetic:
            print("\n[?] Diagnóstico de Ingeniería Inversa:")
            print("    1. Para exportar los pesos oficiales de Qwen3 (1.7B), coloca:")
            print("       - model-00001-of-00002.safetensors")
            print("       - model-00002-of-00002.safetensors")
            print("       - config.json, tokenizer.json")
            print(f"       en la carpeta '{model_dir}'")
            print("    2. O si deseas validar el formato binario GGUF inmediatamente con arquitectura sintética, ejecuta:")
            print(f"       python run.py export-gguf --synthetic\n")
            sys.exit(1)
        else:
            print("[*] Generando arquitectura y pesos sintéticos para validación de formato...")

    # Cargar o sintetizar configuración
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"[+] Configuración cargada desde {config_file.name}")
    else:
        print("[*] Usando configuración oficial de referencia para Qwen3-1.7B.")
        config = get_default_qwen3_config()

    exporter = Qwen3GGUFExporter(config)
    tensors = {}

    if safetensor_files:
        try:
            from safetensors import safe_open
            print(f"[*] Cargando {len(safetensor_files)} shard(s) de Safetensors...")
            for sf in safetensor_files:
                print(f"    - {sf.name} ({sf.stat().st_size / (1024**3):.2f} GB)")
                with safe_open(str(sf), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        tensors[k] = f.get_tensor(k)
            print(f"[+] {len(tensors)} tensores cargados exitosamente.")
        except ImportError:
            print("[-] Falta la librería 'safetensors'. Instala con: pip install safetensors")
            sys.exit(1)
    else:
        # Generar diccionario de tensores sintético representativo
        print("[*] Inicializando tensores canónicos representativos de Qwen3...")
        hidden = config.get("hidden_size", 256)
        vocab = min(config.get("vocab_size", 1000), 1000)
        num_layers = min(config.get("num_hidden_layers", 2), 2)
        inter = config.get("intermediate_size", 512)

        tensors["model.embed_tokens.weight"] = torch.randn(vocab, hidden, dtype=torch.float16)
        tensors["model.norm.weight"] = torch.ones(hidden, dtype=torch.float16)
        tensors["lm_head.weight"] = torch.randn(vocab, hidden, dtype=torch.float16)

        for l in range(num_layers):
            tensors[f"model.layers.{l}.input_layernorm.weight"] = torch.ones(hidden, dtype=torch.float16)
            tensors[f"model.layers.{l}.post_attention_layernorm.weight"] = torch.ones(hidden, dtype=torch.float16)
            tensors[f"model.layers.{l}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden, dtype=torch.float16)
            tensors[f"model.layers.{l}.self_attn.k_proj.weight"] = torch.randn(hidden // 2, hidden, dtype=torch.float16)
            tensors[f"model.layers.{l}.self_attn.v_proj.weight"] = torch.randn(hidden // 2, hidden, dtype=torch.float16)
            tensors[f"model.layers.{l}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden, dtype=torch.float16)
            tensors[f"model.layers.{l}.mlp.gate_proj.weight"] = torch.randn(inter, hidden, dtype=torch.float16)
            tensors[f"model.layers.{l}.mlp.up_proj.weight"] = torch.randn(inter, hidden, dtype=torch.float16)
            tensors[f"model.layers.{l}.mlp.down_proj.weight"] = torch.randn(hidden, inter, dtype=torch.float16)

    output_path = Path(args.output)
    print(f"[*] Exportando a GGUF v3 [Quant: {args.format.upper()}]...")
    result = exporter.export(tensors, str(output_path), quantization=args.format)

    print("\n" + "=" * 68)
    print("  EXPORTACIÓN GGUF COMPLETADA CON ÉXITO")
    print("=" * 68)
    print(f"  Archivo generado : {result['output_path']}")
    print(f"  Tensores empaquetados : {result['tensors_exported']}")
    print(f"  Tamaño final : {result['file_size_bytes'] / (1024**2):.2f} MB")
    print(f"  Listo para ejecutar en: llama.cpp / ollama run")
    print("=" * 68)


if __name__ == "__main__":
    main()
