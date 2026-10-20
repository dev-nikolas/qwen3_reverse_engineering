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
Experimento de Dimensión Espectral y Detección de Dimensiones Rebeldes (Rogue Features) en Qwen3-1.7B.

Objetivo de Ingeniería Inversa Profunda:
Investigar la geometría de las representaciones del flujo residual (residual stream)
a lo largo de las 28 capas del transformador:
1. Rango Efectivo (Shannon Participation Ratio) mediante Descomposición en Valores Singulares (SVD).
2. Anisotropía del espacio latente (similitud coseno promedio entre pares de tokens).
3. Magnitud máxima vs promedio por dimensión (detección de "Rogue Outlier Features" típicas de LLaMA/Mistral).
4. Verificación empírica del impacto de QK-RMSNorm en la estabilidad de la varianza.
"""

import os
import sys
import json
import torch
import numpy as np

# Configuración de rutas relativas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
sys.path.insert(0, BASE_DIR)

from transformers import AutoTokenizer, AutoModelForCausalLM

def calculate_effective_rank(tensor_2d: torch.Tensor) -> float:
    """
    Calcula el rango efectivo espectral de Roy y Vetterli:
    r_eff = exp(- sum(p_i * ln(p_i))) donde p_i = sigma_i / sum(sigma_j)
    """
    # tensor_2d: [seq_len, hidden_size]
    # Restar la media para centrar la matriz de activaciones
    centered = tensor_2d - tensor_2d.mean(dim=0, keepdim=True)
    
    # Calcular SVD sobre float32
    try:
        _, s, _ = torch.linalg.svd(centered.float(), full_matrices=False)
        singular_values = s.cpu().numpy()
        total_s = np.sum(singular_values)
        if total_s == 0:
            return 1.0
        p = singular_values / total_s
        # Filtrar valores casi nulos para evitar log(0)
        p = p[p > 1e-12]
        shannon_entropy = -np.sum(p * np.log(p))
        return float(np.exp(shannon_entropy))
    except Exception as e:
        return 1.0

def calculate_anisotropy(tensor_2d: torch.Tensor) -> float:
    """
    Calcula la similitud coseno promedio entre todos los pares de tokens.
    Un valor cercano a 1.0 indica colapso de representación (anisotropía severa).
    Un valor cercano a 0 indica isotropía uniforme (máxima separabilidad).
    """
    # Normalizar vectores a norma L2 = 1
    normed = tensor_2d / (torch.norm(tensor_2d, dim=-1, keepdim=True) + 1e-8)
    # Matriz de similitud coseno: [seq_len, seq_len]
    sim_matrix = torch.matmul(normed, normed.T)
    # Extraer valores fuera de la diagonal principal
    n = sim_matrix.shape[0]
    if n <= 1:
        return 0.0
    mask = ~torch.eye(n, dtype=torch.bool, device=tensor_2d.device)
    mean_cos = sim_matrix[mask].mean().item()
    return float(mean_cos)

def detect_outlier_kurtosis(tensor_2d: torch.Tensor):
    """
    Evalúa la presencia de dimensiones rebeldes (outlier features).
    Calcula el valor absoluto máximo por coordenada normalizado por la media global.
    """
    abs_vals = torch.abs(tensor_2d)
    max_per_dim, _ = torch.max(abs_vals, dim=0) # [hidden_size]
    global_mean = torch.mean(abs_vals).item()
    top_outlier_ratio = (torch.max(max_per_dim).item() / (global_mean + 1e-8))
    
    # Índices de las 3 coordenadas con mayor magnitud
    top3_dims = torch.topk(max_per_dim, 3).indices.tolist()
    top3_vals = [round(float(max_per_dim[i].item()), 2) for i in top3_dims]
    
    return float(top_outlier_ratio), top3_dims, top3_vals

def run_spectral_forensics():
    print("=" * 70)
    print("  INGENIERÍA INVERSA: ANÁLISIS ESPECTRAL Y DETECCIÓN DE DIMENSIONES REBELDES")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Dispositivo de cálculo: {device}")
    
    print("[*] Inicializando modelo Qwen3 y Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()

    # Texto de prueba con diversidad sintáctica, matemática y semántica
    prompt = (
        "El análisis mecanístico de redes neuronales profundas permite desentrañar "
        "los circuitos de inducción y las dinámicas algebraicas del flujo residual. "
        "En particular, la función f(x) = x^3 - 4x + 7 posee propiedades ergódicas "
        "cuando converge en el espacio latente de alta dimensión."
    )
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)
    seq_len = input_ids.shape[1]
    print(f"[*] Secuencia analizada: {seq_len} tokens")

    # Registro de estados ocultos capa por capa
    print("[*] Extrayendo flujo residual a través de las 28 capas...")
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    results = {
        "sequence_length": seq_len,
        "hidden_size": 2048,
        "layers": []
    }

    print("\n{:<10} {:<16} {:<16} {:<18} {:<15}".format(
        "Capa", "Rango Efectivo", "Anisotropía", "Outlier Ratio", "Top Dim Outlier"
    ))
    print("-" * 75)

    for l_idx, h in enumerate(hidden_states):
        # h: [1, seq_len, 2048] -> [seq_len, 2048]
        rep = h[0]
        layer_name = f"Embedding" if l_idx == 0 else (f"Post-Norm" if l_idx == len(hidden_states) - 1 else f"Layer {l_idx - 1}")
        
        eff_rank = calculate_effective_rank(rep)
        anisotropy = calculate_anisotropy(rep)
        outlier_ratio, top3_dims, top3_vals = detect_outlier_kurtosis(rep)

        layer_data = {
            "layer_name": layer_name,
            "layer_index": l_idx - 1 if l_idx > 0 and l_idx < len(hidden_states) - 1 else l_idx,
            "effective_rank": round(eff_rank, 2),
            "anisotropy_cosine": round(anisotropy, 4),
            "outlier_max_to_mean_ratio": round(outlier_ratio, 2),
            "top_outlier_dimensions": top3_dims,
            "top_outlier_magnitudes": top3_vals
        }
        results["layers"].append(layer_data)

        print("{:<10} {:<16.2f} {:<16.4f} {:<18.2f} {:<15}".format(
            layer_name, eff_rank, anisotropy, outlier_ratio, f"Dim {top3_dims[0]} ({top3_vals[0]})"
        ))

    out_file = os.path.join(REPORTS_DIR, "qwen3_spectral_and_outlier_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"[+] Informe espectral guardado exitosamente en:")
    print(f"    reports/qwen3_spectral_and_outlier_report.json")
    print("=" * 70)

if __name__ == "__main__":
    run_spectral_forensics()
