# Copyright 2025 The Qwen team, Alibaba Group and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

import os
import sys
import json
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 70)
print("  INGENIERÍA INVERSA: FORENSE DEL TOKENIZADOR Y TOKENS ESPECIALES")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# 1. Análisis de Tamaño y Rangos del Vocabulario
vocab = tokenizer.get_vocab()
total_tokens = len(vocab)
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

print(f"[*] Tamaño total del vocabulario BPE: {total_tokens:,} tokens")
print(f"[*] Rango de IDs: {sorted_vocab[0][1]} hasta {sorted_vocab[-1][1]}")

# 2. Extracción de Tokens Especiales y de Control
special_tokens_map = tokenizer.special_tokens_map
all_special_ids = set(tokenizer.all_special_ids)

# Buscar tokens de control clave por prefijo
control_tokens = {}
reasoning_tokens = {}
fim_tokens = {} # Fill-in-the-middle
multimodal_tokens = {}

for token, tid in sorted_vocab:
    if tid >= 151643 or token.startswith("<|") or token.startswith("<"):
        if "think" in token:
            reasoning_tokens[token] = tid
        elif "fim" in token:
            fim_tokens[token] = tid
        elif any(k in token for k in ["audio", "image", "video", "quad", "box"]):
            multimodal_tokens[token] = tid
        elif token.startswith("<|"):
            control_tokens[token] = tid

print(f"\n[+] Tokens de Razonamiento (<think>): {reasoning_tokens}")
print(f"[+] Tokens de Edición de Código (FIM): {fim_tokens}")
print(f"[+] Muestra de Tokens de Control: {dict(list(control_tokens.items())[:8])}")
print(f"[+] Muestra de Tokens Multimodales: {dict(list(multimodal_tokens.items())[:8])}")

# 3. Verificación de Templates de Conversación ChatML
messages = [
    {"role": "system", "content": "Eres un asistente de IA experto en ingeniería inversa."},
    {"role": "user", "content": "¿Que es un transformer?"},
    {"role": "assistant", "content": "<think>\nDefinir auto-atencion y residual stream.\n</think>\nUn transformer es..."}
]

try:
    formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False)
except Exception:
    formatted_chat = "Chat template no nativo en tokenizer_config.json"

print("\n--- FORMATO DE PLANTILLA CHATML FORENSE ---")
print(formatted_chat)

report = {
    "total_vocab_size": total_tokens,
    "special_tokens_count": len(all_special_ids),
    "reasoning_tokens": reasoning_tokens,
    "fim_code_tokens": fim_tokens,
    "multimodal_tokens": multimodal_tokens,
    "control_tokens_sample": control_tokens
}

out_path = os.path.join(REPORTS_DIR, "qwen3_tokenizer_forensics.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 70)
print(f"[+] Informe guardado en: reports/qwen3_tokenizer_forensics.json")
print("=" * 70)
