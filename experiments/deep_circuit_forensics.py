"""
Deep Forensic Circuit Extraction for Qwen3-1.7B
1. Induction Heads Detector: Rastreo de los 448 cabezales de atención para hallar los circuitos de aprendizaje en contexto.
2. Divergencia Residual y Capa de Inyección de Conocimiento (Contrastive Activation Probing).
3. Prueba del Modo de Razonamiento Profundo (<think> ... </think>).
"""

from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def detect_induction_heads(model, tokenizer) -> List[Dict[str, Any]]:
    """
    Detecta 'Induction Heads' entre los 448 cabezales de atención (28 capas x 16 cabezales).
    Un cabezal de inducción busca patrones [A][B]...[A] -> predice [B] atendiendo al token posterior a la primera ocurrencia de A.
    """
    print("\n--- [1] Rastreando Cabezales de Inducción (448 cabezales) ---")
    # Generar secuencia de prueba repetida con tokens aleatorios pero estructurados
    sequence_a = [" alfa", " delta", " sigma", " omega", " beta", " gamma", " zeta", " kappa"]
    repeated_seq = " ".join(sequence_a) + " | " + " ".join(sequence_a)
    inputs = tokenizer(repeated_seq, return_tensors="pt")
    input_ids = inputs.input_ids[0].tolist()
    seq_len = len(input_ids)

    # Identificar índices de repetición
    first_half_tokens = input_ids[:seq_len // 2]
    second_half_tokens = input_ids[seq_len // 2:]

    # Capturar matrices de atención de todas las 28 capas
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # outputs.attentions es una tupla de 28 tensores con shape [batch=1, num_heads=16, seq_len, seq_len]
    induction_scores = []
    
    # Un cabezal de inducción en la posición i (segunda ocurrencia de un token) atiende a j+1 (donde j era la primera ocurrencia)
    for layer_idx, layer_attn in enumerate(outputs.attentions):
        attn_matrix = layer_attn[0]  # [16, seq_len, seq_len]
        num_heads = attn_matrix.shape[0]

        for head_idx in range(num_heads):
            head_attn = attn_matrix[head_idx]  # [seq_len, seq_len]
            # Medir la masa de atención dirigida a la posición subsecuente de la coincidencia previa
            # Tomamos el promedio de la diagonal desplazada de inducción
            score = 0.0
            count = 0
            for i in range(seq_len // 2 + 1, seq_len):
                tok = input_ids[i]
                # Buscar si tok apareció en la primera mitad
                if tok in first_half_tokens:
                    first_pos = first_half_tokens.index(tok)
                    target_pos = first_pos + 1
                    if target_pos < seq_len:
                        score += head_attn[i, target_pos].item()
                        count += 1
            avg_induction_score = (score / count) if count > 0 else 0.0
            induction_scores.append({
                "layer": layer_idx,
                "head": head_idx,
                "induction_score": round(avg_induction_score, 4),
            })

    # Ordenar por puntaje de inducción descendente
    induction_scores.sort(key=lambda x: x["induction_score"], reverse=True)
    top_induction_heads = induction_scores[:10]
    
    for h in top_induction_heads:
        print(f"  -> Capa {h['layer']:2d}, Cabezal {h['head']:2d}: Score de Inducción = {h['induction_score']:.4f}")

    return top_induction_heads


def test_contrastive_knowledge_injection(model, tokenizer) -> List[Dict[str, Any]]:
    """
    Rastrea la 'Capa de Inyección de Conocimiento' comparando la divergencia entre dos afirmaciones:
    A: 'La capital de Francia es París' (Verdadera)
    B: 'La capital de Francia es Madrid' (Falsa / Contrafactual)
    Mide la distancia angular (coseno) y la distancia euclídea entre los residual streams de ambas.
    """
    print("\n--- [2] Análisis Contrastivo de Inyección de Hechos Fácticos ---")
    prompt_a = "La capital de Francia es París"
    prompt_b = "La capital de Francia es Madrid"

    inputs_a = tokenizer(prompt_a, return_tensors="pt")
    inputs_b = tokenizer(prompt_b, return_tensors="pt")

    states_a = []
    states_b = []

    def hook_a(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        states_a.append(h.detach().clone().float())

    def hook_b(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        states_b.append(h.detach().clone().float())

    # Registrar hooks en cada bloque
    handles_a = [l.register_forward_hook(hook_a) for l in model.model.layers]
    with torch.no_grad():
        model(**inputs_a)
    for h in handles_a:
        h.remove()

    handles_b = [l.register_forward_hook(hook_b) for l in model.model.layers]
    with torch.no_grad():
        model(**inputs_b)
    for h in handles_b:
        h.remove()

    # Comparar la representación del token de sujeto 'Francia' (índice 3)
    target_token_idx = 3
    divergence_per_layer = []

    for l_idx in range(len(states_a)):
        v_a = states_a[l_idx][0, target_token_idx]
        v_b = states_b[l_idx][0, target_token_idx]

        cos_sim = F.cosine_similarity(v_a.unsqueeze(0), v_b.unsqueeze(0)).item()
        l2_dist = torch.norm(v_a - v_b).item()

        divergence_per_layer.append({
            "layer": l_idx,
            "cosine_similarity": round(cos_sim, 4),
            "angular_distance_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, cos_sim)))), 2),
            "l2_distance": round(l2_dist, 4),
        })

    # Mostrar puntos clave
    for d in divergence_per_layer[::4]:
        print(f"  Capa {d['layer']:2d}: Similitud Coseno = {d['cosine_similarity']:.4f} (Distancia Angular = {d['angular_distance_deg']}°)")
    last_d = divergence_per_layer[-1]
    print(f"  Capa {last_d['layer']:2d}: Similitud Coseno = {last_d['cosine_similarity']:.4f} (Distancia Angular = {last_d['angular_distance_deg']}°)")

    return divergence_per_layer


def test_thinking_mode_generation(model, tokenizer) -> Dict[str, Any]:
    """Prueba la activación y emisión de bloques de pensamiento profundo (<think>...</think>)."""
    print("\n--- [3] Prueba del Modo de Razonamiento (<think> / Chain-of-Thought) ---")
    reasoning_prompt = (
        "<|im_start|>user\n"
        "Tengo 3 cajas rojas. Dentro de cada caja roja hay 2 cajas verdes. "
        "Dentro de cada caja verde hay 4 monedas. ¿Cuántas monedas y cajas hay en total? "
        "Piensa detenidamente antes de responder.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )

    inputs = tokenizer(reasoning_prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = round(time.time() - t0, 2)
    full_output = tokenizer.decode(out[0], skip_special_tokens=False)
    
    # Extraer la porción generada después del prompt
    prompt_len_chars = len(reasoning_prompt)
    generated_suffix = full_output[prompt_len_chars:]

    print(f"Generación completada en {gen_time} s.")
    print("Salida generada:\n" + generated_suffix[:300] + ("..." if len(generated_suffix) > 300 else ""))

    return {
        "prompt": reasoning_prompt,
        "generation_time_s": gen_time,
        "generated_text": generated_suffix,
        "contains_think_close": "</think>" in generated_suffix,
    }


def main():
    print("=== Cargando modelo Qwen3-1.7B en memoria ===")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Modelo cargado en {time.time() - t0:.2f} s")

    # 1. Cabezales de Inducción
    induction_data = detect_induction_heads(model, tokenizer)

    # 2. Divergencia Residual Fáctica
    contrastive_data = test_contrastive_knowledge_injection(model, tokenizer)

    # 3. Modo de Razonamiento
    thinking_data = test_thinking_mode_generation(model, tokenizer)

    results = {
        "top_induction_heads": induction_data,
        "contrastive_knowledge_divergence": contrastive_data,
        "thinking_mode_experiment": thinking_data,
    }

    out_file = REPORT_DIR / "qwen3_circuits_and_reasoning_forensics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nReporte de circuitos guardado en: {out_file}")


if __name__ == "__main__":
    main()
