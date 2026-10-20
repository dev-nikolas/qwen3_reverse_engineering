# Comparativa Arquitectónica: Qwen3 vs Llama 3 vs Gemma 2 vs DeepSeek

Este documento presenta una comparativa rigurosa y formal entre la arquitectura interna de **Qwen3-1.7B** y los modelos abiertos contemporáneos más representativos del estado del arte.

---

## 📊 Matriz Comparativa Exhaustiva

| Parámetro / Característica | **Qwen3-1.7B** | **Llama 3.2-1B** | **Gemma 2-2B** | **DeepSeek-V3 / R1 (MoE)** |
| :--- | :--- | :--- | :--- | :--- |
| **Creador** | Alibaba Cloud | Meta AI | Google DeepMind | DeepSeek AI |
| **Licencia** | Apache 2.0 | Llama 3.2 Community | Gemma Terms of Use | MIT |
| **Parámetros Totales** | **1,720,574,976** | 1,235,814,400 | 2,614,341,888 | 671B (37B activos / token) |
| **Capas Transformer ($L$)** | **28** | 16 | 26 | 61 |
| **Dimensión Oculta ($d_{model}$)** | **2,048** | 2,048 | 2,304 | 2,048 (compresión latente) |
| **Atención** | **GQA (16 Q / 8 KV)** | GQA (32 Q / 8 KV) | GQA Local/Global alternada | **MLA (Multi-Head Latent)** |
| **Ratio de Compresión KV** | **2 : 1** | 4 : 1 | 2 : 1 (local 4096 / global) | **Latente $d_c=512$** |
| **Dimensión por Cabeza ($d_k$)** | **128** | 64 | 256 | 128 |
| **QK-RMSNorm** | **SÍ (por cabeza)** | NO | NO | **SÍ** |
| **Activación FFN / MLP** | **SwiGLU** | SwiGLU | GeGLU | SwiGLU MoE (256 expertos) |
| **Dimensión Intermedia FFN** | **6,144** | 8,192 | 9,216 | 2,048 por experto |
| **Posicional (RoPE)** | **$\theta = 1,000,000$** | $\theta = 500,000$ | $\theta = 10,000$ | $\theta = 10,000$ (con YaRN) |
| **Contexto Nativo** | **32,768 (131k YaRN)**| 131,072 | 8,192 | 128,000 |
| **Normalización** | **RMSNorm ($\epsilon=10^{-6}$)**| RMSNorm ($\epsilon=10^{-5}$)| RMSNorm ($\epsilon=10^{-6}$)| RMSNorm ($\epsilon=10^{-6}$) |
| **Pre / Post Normalización** | Pre-Norm Dual | Pre-Norm | Pre y Post-Norm por bloque| Pre-Norm Dual |
| **Logit Soft-Capping** | NO | NO | **SÍ (cap a 30.0 / 50.0)** | NO |
| **Tokens Razonamiento CoT** | **SÍ (`<think>`, `</think>`)**| NO | NO | **SÍ (`<think>`, `</think>`)** |
| **Tokens FIM (Fill-In-The-Middle)** | **SÍ (Nativo)** | SÍ (Especiales) | NO | SÍ |
| **Tamaño del Vocabulario** | **151,936** | 128,256 | 256,000 | 129,280 |

---

## 🔬 Diferenciadores Clave de Qwen3

1. **Estabilidad Numérica con QK-RMSNorm:**  
   A diferencia de Llama 3 o Gemma 2, Qwen3 implementa normalización RMS sobre las matrices proyectadas $Q$ y $K$ de cada cabeza antes de calcular el producto punto escalado. Esto previene la dispersión del espectro atencional en contextos largos y elimina el crecimiento explosivo de logits atencionales.

2. **Equilibrio GQA 16:8:**  
   Con un ratio de compresión KV de 2:1 y cabezas de 128 dimensiones, ofrece el punto óptimo entre fidelidad de recuperación atencional y consumo eficiente de memoria VRAM para inferencia local.

3. **Arquitectura con Pensamiento Nativo:**  
   El soporte nativo a nivel de tokenizador y alineación para los delimitadores `<think>` y `</think>` permite un flujo cognitivo estructurado de Chain-of-Thought sin requerir envoltorios o prompting externo.

---
> **Comando de ejecución:**  
> `python run.py compare`
