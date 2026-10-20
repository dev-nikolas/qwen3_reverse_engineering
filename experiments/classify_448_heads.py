# Copyright 2025 The Qwen team, Alibaba Group and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 70)
print("  INGENIERÍA INVERSA: CLASIFICACIÓN ARQUETÍPICA DE LAS 448 CABEZAS")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# Importante: attn_implementation='eager' para extraer matrices de atención
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.float32,
    attn_implementation="eager"
)
model.eval()

# Secuencia sintética de prueba: tokens repetidos para medir inducción y token anterior
# Ejemplo: [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]
tokens_pattern = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
seq = tokens_pattern + tokens_pattern
input_ids = torch.tensor([seq])
seq_len = len(seq)

with torch.no_grad():
    outputs = model(input_ids, output_attentions=True)
    # outputs.attentions: tupla de 28 tensores, cada uno [batch=1, heads=16, seq_len=16, seq_len=16]
    attentions = outputs.attentions

print(f"[*] Total capas analizadas: {len(attentions)}")
print(f"[*] Total cabezas analizadas: {len(attentions) * 16} (448 cabezas)")

catalog = {
    "induction_heads": [],       # Arquetipo 1: Copiadoras (miran a token pos+1 de la primera repetición)
    "previous_token_heads": [],  # Arquetipo 2: Token inmediatamente anterior (diagonal -1)
    "attention_sink_heads": [],  # Arquetipo 3: Fijan atención en token 0 (BOS/Sink)
    "diffuse_heads": []          # Arquetipo 4: Alta entropía / atención de largo alcance
}

for layer_idx, layer_attn in enumerate(attentions):
    # layer_attn: [1, 16, seq_len, seq_len]
    attn_matrix = layer_attn[0] # [16, seq_len, seq_len]
    
    for head_idx in range(16):
        mat = attn_matrix[head_idx] # [seq_len, seq_len]
        
        # 1. Métrica de Attention Sink (atención al token 0 promedio sobre tokens posteriores)
        sink_score = mat[2:, 0].mean().item()
        
        # 2. Métrica de Previous-Token (atención a i-1 en diagonal inferior)
        prev_scores = [mat[i, i-1].item() for i in range(1, seq_len)]
        prev_score = sum(prev_scores) / len(prev_scores)
        
        # 3. Métrica de Induction (cuando estamos en la segunda mitad seq[8..], cuánto atiende a seq[1..7])
        # En posición 8 (segundo 1024), debería atender a la posición 1 (2048) si predice el siguiente
        induction_scores = []
        for i in range(len(tokens_pattern), seq_len - 1):
            target_pos = i - len(tokens_pattern) + 1
            induction_scores.append(mat[i, target_pos].item())
        induction_score = sum(induction_scores) / len(induction_scores)
        
        # 4. Entropía de atención (promedio)
        # Entropía de Shannon = -sum(p * log(p))
        entropy = -(mat * torch.log(mat + 1e-9)).sum(dim=-1).mean().item()
        
        head_id = f"L{layer_idx:02d}H{head_idx:02d}"
        metrics = {
            "head": head_id,
            "layer": layer_idx,
            "head_idx": head_idx,
            "induction_score": round(induction_score, 4),
            "previous_token_score": round(prev_score, 4),
            "sink_score": round(sink_score, 4),
            "entropy": round(entropy, 4)
        }
        
        # Clasificación por umbral dominante
        if induction_score >= 0.25:
            catalog["induction_heads"].append(metrics)
        elif prev_score >= 0.40:
            catalog["previous_token_heads"].append(metrics)
        elif sink_score >= 0.50:
            catalog["attention_sink_heads"].append(metrics)
        elif entropy >= 2.0:
            catalog["diffuse_heads"].append(metrics)

# Ordenar por fuerza de la métrica
catalog["induction_heads"].sort(key=lambda x: x["induction_score"], reverse=True)
catalog["previous_token_heads"].sort(key=lambda x: x["previous_token_score"], reverse=True)
catalog["attention_sink_heads"].sort(key=lambda x: x["sink_score"], reverse=True)
catalog["diffuse_heads"].sort(key=lambda x: x["entropy"], reverse=True)

print("\n--- RESUMEN ARQUETÍPICO DE LAS 448 CABEZAS ---")
print(f"1. Cabezas Copiadoras (Induction Heads)     : {len(catalog['induction_heads'])} cabezas")
top_ind = [f"{h['head']} ({h['induction_score']})" for h in catalog['induction_heads'][:3]]
print(f"   Top 3: {top_ind}")

print(f"2. Cabezas de Token Previo (Previous-Token) : {len(catalog['previous_token_heads'])} cabezas")
top_prev = [f"{h['head']} ({h['previous_token_score']})" for h in catalog['previous_token_heads'][:3]]
print(f"   Top 3: {top_prev}")

print(f"3. Sumideros de Atención (Attention Sinks)  : {len(catalog['attention_sink_heads'])} cabezas")
top_sink = [f"{h['head']} ({h['sink_score']})" for h in catalog['attention_sink_heads'][:3]]
print(f"   Top 3: {top_sink}")

print(f"4. Cabezas Difusas de Contexto Global       : {len(catalog['diffuse_heads'])} cabezas")

out_file = os.path.join(REPORTS_DIR, "qwen3_448_heads_functional_catalog.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(catalog, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 70)
print(f"[+] Catálogo completo guardado en: reports/qwen3_448_heads_functional_catalog.json")
print("=" * 70)
