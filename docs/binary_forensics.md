# Análisis Forense Binario y Tokenizador — Qwen3-1.7B

Este documento describe la estructura binaria de almacenamiento de los tensores de peso de **Qwen3-1.7B**, el mapeo en memoria y el análisis forense de su tokenizador de 151,936 tokens.

---

## 📦 1. Análisis Forense Binario de los Pesos (Safetensors)

El modelo oficial de Qwen3-1.7B se almacena bajo el formato **Safetensors**, el cual ofrece deserialización segura (*zero-copy memory mapping*) y previene ataques de inyección de código binario presentes en formatos Pickle estándar.

### Distribución de Shards y Memoria
* **Total de Parámetros:** Exactamente **1,720,574,976** parámetros de coma flotante (`bfloat16`).
* **Total de Tensores:** 311 matrices y vectores individuales.
* **Índice Sharded:** `model.safetensors.index.json` que mapea cada tensor a su respectivo fragmento binario:
  * `model-00001-of-00002.safetensors` (~3.44 GB): Contiene la matriz de embeddings (`model.embed_tokens.weight`) y las capas Transformer de la **00 a la 23**.
  * `model-00002-of-00002.safetensors` (~622 MB): Contiene las capas Transformer de la **24 a la 27**, la normalización final (`model.norm.weight`) y la proyección del cabezal de lenguaje (`lm_head.weight`).

### Estructura Tensor a Tensor por Bloque Transformer ($l \in [0, 27]$):
Cada uno de los 28 bloques decodificadores contiene exactamente 11 tensores:
1. `model.layers.{l}.input_layernorm.weight`: Normalización RMS pre-atención ($[2048]$).
2. `model.layers.{l}.self_attn.q_proj.weight`: Proyección Query ($[2048, 2048]$).
3. `model.layers.{l}.self_attn.k_proj.weight`: Proyección Key ($[1024, 2048]$ - GQA 8 cabezas).
4. `model.layers.{l}.self_attn.v_proj.weight`: Proyección Value ($[1024, 2048]$ - GQA 8 cabezas).
5. `model.layers.{l}.self_attn.o_proj.weight`: Proyección Output de Atención ($[2048, 2048]$).
6. `model.layers.{l}.self_attn.q_norm.weight`: Normalización RMS estabilizadora por cabeza de Query ($[128]$).
7. `model.layers.{l}.self_attn.k_norm.weight`: Normalización RMS estabilizadora por cabeza de Key ($[128]$).
8. `model.layers.{l}.post_attention_layernorm.weight`: Normalización RMS pre-MLP ($[2048]$).
9. `model.layers.{l}.mlp.gate_proj.weight`: Compuerta Swish de SwiGLU ($[6144, 2048]$).
10. `model.layers.{l}.mlp.up_proj.weight`: Proyección Lineal de SwiGLU ($[6144, 2048]$).
11. `model.layers.{l}.mlp.down_proj.weight`: Proyección Inversa de SwiGLU ($[2048, 6144]$).

---

## 🔡 2. Forense del Tokenizador y Razonamiento Multimodal

Al inspeccionar los 151,936 identificadores en `model/tokenizer.json` mediante `experiments/tokenizer_forensics.py`, se identifican familias específicas de tokens especializados:

### A. Tokens Nativos de Razonamiento (Chain-of-Thought)
* `<think>` (ID `151667`)
* `</think>` (ID `151668`)
* *Propósito:* El modelo fue entrenado para emitir razonamientos preliminares paso a paso dentro de estos delimitadores antes de producir la respuesta visible al usuario.

### B. Tokens de Relleno en el Medio (Fill-In-The-Middle / FIM)
* `<|fim_prefix|>` (ID `151659`)
* `<|fim_middle|>` (ID `151660`)
* `<|fim_suffix|>` (ID `151661`)
* `<|repo_name|>` (ID `151664`)
* `<|file_sep|>` (ID `151665`)
* *Propósito:* Capacidades de completado de código dentro de repositorios completos y reescritura puntual en IDEs.

### C. Tokens de Agente y Llamada a Herramientas (Tool Calling)
* `<tool_call>`
* `</tool_call>`
* `<tool_response>`
* `</tool_response>`
* *Propósito:* Invocación estructurada de funciones JSON Schema y parseo de retornos de APIs externas.

### D. Tokens Multimodales y Coordenadas Geométricas
* **Visión:** `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`.
* **Audio:** `<|audio_start|>`, `<|audio_end|>`.
* **Detección Espacial de Objetos:**
  * `<|box_start|>`, `<|box_end|>` (delimitadores de Bounding Boxes $[y_{min}, x_{min}, y_{max}, x_{max}]$).
  * `<|quad_start|>`, `<|quad_end|>` (delimitadores de cuadriláteros orientados para OCR y segmentación gráfica).

---
> **Comandos útiles:**  
> Inspeccionar cabeceras binarias: `python run.py inspect`  
> Analizar tokens del vocabulario: `python run.py tokenizer`
