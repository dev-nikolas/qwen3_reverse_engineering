# Copyright 2026 Alibaba Group and the Qwen Team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Experimento Forense de Supresión de Copia (Copy-Suppression Heads) en Qwen3-1.7B.

Objetivo de Ingeniería Inversa Profunda:
Identificar los circuitos inhibidores que equilibran a las cabezas de inducción
(como L16H14). Mientras las cabezas de inducción promueven la copia de secuencias
repetidas, las cabezas de supresión de copia (Copy-Suppression Heads) en capas
tardías (17-27) proyectan logits negativos sobre el token repetido para calibrar
las probabilidades y prevenir bucles infinitos de alucinación y eco.
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
sys.path.insert(0, BASE_DIR)

def run_copy_suppression_experiment():
    print("=" * 70)
    print("  INGENIERÍA INVERSA: DETECCIÓN DE CABEZAS DE SUPRESIÓN DE COPIA (COPY-SUPPRESSION)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Dispositivo de ejecución: {device}")

    print("[*] Cargando modelo y tokenizer de Qwen3...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float32,
        attn_implementation="eager"
    )
    model.eval()

    # Patrón de inducción sintético: [A][B] ... [A] -> debe predecir [B]
    # Si B se repite prematuramente, las cabezas de supresión deben proyectar logit negativo
    prompt = " Alpha Beta Gamma Delta Epsilon Zeta Alpha Beta Gamma Delta Epsilon"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    tokens_text = [tokenizer.decode([t]) for t in input_ids[0]]
    seq_len = input_ids.shape[1]
    
    target_token_id = tokenizer.encode(" Zeta", add_special_tokens=False)[-1]
    target_token_str = tokenizer.decode([target_token_id])
    print(f"[*] Secuencia de prueba: '{prompt}'")
    print(f"[*] Token objetivo de inducción: '{target_token_str}' (ID {target_token_id})")

    # Extraer pesos de desembebing W_U
    lm_head_weight = model.lm_head.weight.detach() # [vocab_size, hidden_size]
    w_u_target = lm_head_weight[target_token_id]    # [hidden_size]

    suppression_scores = []

    print("[*] Evaluando contribución de logits de las cabezas tardías (Capas 16 a 27)...")
    
    # Hook para interceptar la salida de cada cabeza de atención antes de o_proj
    # En Qwen2/Qwen3: layer.self_attn produce context_layer tras attention_weights @ V
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True, output_hidden_states=True)
        attentions = outputs.attentions # Tupla de 28 tensores [1, num_heads, seq_len, seq_len]

    # Analizar capas 16 a 27
    for l_idx in range(16, 28):
        attn_layer = model.model.layers[l_idx].self_attn
        o_proj_weight = attn_layer.o_proj.weight.detach() # [hidden_size, hidden_size]
        # o_proj_weight está dividida en 16 cabezas de tamaño 128
        # Forma: [2048, 2048] -> por cabeza c: o_proj[:, c*128 : (c+1)*128]
        
        # Buscar posiciones donde apareció el token objetivo
        target_positions = [i for i, tid in enumerate(input_ids[0].tolist()) if tid == target_token_id]
        
        attn_matrix = attentions[l_idx][0] # [num_heads, seq_len, seq_len]
        
        for h_idx in range(16):
            # Atención del último token hacia la posición del token candidato
            attn_to_target = 0.0
            if target_positions:
                attn_to_target = max([attn_matrix[h_idx, -1, pos].item() for pos in target_positions])

            # Calcular la proyección directa de la cabeza sobre W_U (Direct Logit Attribution)
            head_o_slice = o_proj_weight[:, h_idx * 128 : (h_idx + 1) * 128] # [2048, 128]
            
            # La contribución potencial a W_U(Zeta) viene de: w_u_target @ head_o_slice
            direct_dla_vector = torch.matmul(w_u_target, head_o_slice) # [128]
            norm_dla = direct_dla_vector.norm().item()
            mean_dla = direct_dla_vector.mean().item()

            # Puntuación combinada de inhibición:
            # Alta atención al token repetido con proyección direccional negativa
            suppression_scores.append({
                "layer": l_idx,
                "head": h_idx,
                "head_id": f"L{l_idx:02d}H{h_idx:02d}",
                "attn_to_target": round(attn_to_target, 4),
                "dla_norm": round(norm_dla, 4),
                "dla_mean": round(mean_dla, 4),
                "suppression_index": round(attn_to_target * (-mean_dla if mean_dla < 0 else 0.0), 6)
            })

    # Ordenar por índice de supresión de copia
    suppression_scores.sort(key=lambda x: x["attn_to_target"], reverse=True)

    print("\n--- TOP 10 CABEZAS TARDÍAS CON ATENCIÓN DIRIGIDA AL CANDIDATO ---")
    print("{:<10} {:<16} {:<16} {:<16}".format("Cabeza", "Atención a Token", "DLA Norm", "DLA Mean"))
    print("-" * 60)
    for h in suppression_scores[:10]:
        print("{:<10} {:<16.4f} {:<16.4f} {:<16.4f}".format(
            h["head_id"], h["attn_to_target"], h["dla_norm"], h["dla_mean"]
        ))

    out_file = os.path.join(REPORTS_DIR, "qwen3_copy_suppression_forensics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(suppression_scores, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"[+] Informe de Supresión de Copia guardado en:")
    print(f"    reports/qwen3_copy_suppression_forensics.json")
    print("=" * 70)

if __name__ == "__main__":
    run_copy_suppression_experiment()
