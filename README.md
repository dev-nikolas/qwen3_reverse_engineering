# Qwen3-1.7B — Reingeniería y Desmitificación de un LLM

> **Proyecto Personal con Fin Educativo:** Un estudio práctico de ingeniería inversa para aprender cómo se compone y opera un Large Language Model (LLM) moderno desde adentro hacia afuera, desmitificando el misticismo en torno a la inteligencia artificial.

Implementado en **PyTorch puro**, este repositorio reconstruye fielmente la arquitectura de `Qwen/Qwen3-1.7B` (Alibaba Cloud / Licencia Apache 2.0), demostrando que un LLM no es una caja negra impenetrable, sino una composición rigurosa y elegante de álgebra lineal, normalizaciones RMS, atención agrupada (GQA) y cálculo tensorial.

---

## 📐 Especificaciones de la Arquitectura

| Parámetro | Valor Oficial Qwen3-1.7B | Función Matemática |
| :--- | :--- | :--- |
| **Parámetros Totales** | **1,720,574,976** | Pesos entrenables en `bfloat16` (4.06 GB sharded) |
| **Capas Transformer** | **28 bloques decodificadores** | Conexiones residuales con pre-normalización dual |
| **Dimensión Oculta ($d_{model}$)** | **2,048** | Dimensión del flujo residual latente |
| **Mecanismo de Atención** | **GQA (16 Q / 8 KV)** | 16 cabezas de consulta y 8 de clave/valor (ratio 2:1) |
| **Dimensión por Cabeza ($d_k$)** | **128** | Normalizadas individualmente con **QK-RMSNorm** |
| **MLP / Feed-Forward** | **SwiGLU ($d_{ff} = 6,144$)** | Compuertas no lineales `gate_proj`, `up_proj`, `down_proj` |
| **Embeddings Posicionales** | **RoPE ($\theta = 1,000,000$)** | Contexto extendible de 32k a 131k tokens con YaRN |
| **Normalización** | **RMSNorm ($\epsilon = 10^{-6}$)** | Pre-atención y pre-MLP sin sesgo residual |
| **Vocabulario BPE** | **151,936 tokens** | Soporta razonamiento nativo (`<think>`) y FIM |

---

## ⚡ Inicio Rápido (Getting Started)

### 1. Requisitos e Instalación
Recomendado Python 3.10+ con PyTorch:
```bash
pip install torch safetensors
```

### 2. Descargar los Pesos Oficiales (4.06 GB)
El repositorio incluye un script con descarga reanudable (HTTP Range) para obtener los pesos de Hugging Face sin esfuerzo:
```bash
# En Windows:
download_weights.bat

# O directamente en Python:
python tools/download_weights.py
```

### 3. Chat Interactivo en Tiempo Real
Prueba la inferencia nativa en PyTorch puro (cero dependencias externas):
```bash
python run.py standalone-chat
```
*Soporta delimitadores `<think>` en vivo, comandos dinámicos como `/temp 0.7` y activación guiada por capas con `/steer reasoning 2.0`.*

### 4. Validar Pruebas Unitarias
Ejecuta la suite completa de 58 pruebas unitarias matemáticas automatizadas:
```bash
python run.py test
```

---

## 📚 Documentación Técnica Detallada

Toda la investigación profunda, resultados de experimentos y análisis matemáticos se encuentran organizados en documentos modulares:

* 🔬 **[Experimentos Forenses y Circuitos Mecanicistas](docs/experiments_and_circuits.md):**
  - *Exp 1:* Mapa de criticidad por ablación de capas (descubrimiento de la Capa 26 redundante).
  - *Exp 2:* Trasplante causal de activaciones en la Capa 18 (81.9% de certeza).
  - *Exp 3:* Censo arquetípico de las 448 cabezas de atención (62.3% sumideros de atención).
  - *Exp 4:* Disección espectral de la cabeza copiadora L16H14.
  - *Exp 5:* Circuito inhibidor anti-repetición (Copy-Suppression Heads en L19/L20).
  - *Exp 6:* Dimensión rebelde 1793 ($+13,029$ DC offset) y rango efectivo SVD.
  - *Exp 7:* Radiografía de neuronas de memoria conceptual en el MLP (ej. Neurona 1950).

* 📦 **[Análisis Forense Binario y Tokenizador](docs/binary_forensics.md):**
  - Mapeo en memoria de los 311 tensores Safetensors y shards binarios.
  - Forense de tokens de razonamiento (`<think>`), FIM de código y visión multimodal.

* 📈 **[Logit Lens y Emergencia de Conceptos](docs/logit_lens.md):**
  - Desproyección del flujo residual capa por capa y maduración de predicciones.

* 🏛️ **[Comparativa Arquitectónica](docs/architectural_comparison.md):**
  - Matriz comparativa: Qwen3 vs Llama 3 vs Gemma 2 vs DeepSeek-V3/R1.

* 💻 **[Código Fuente de Alto Calibre](source_code/README.md):**
  - Arquitectura modular limpia: atención (`gqa`, `mla`, `swa`), capas (`swiglu`, `moe`, `rmsnorm`, `yarn`), motores de inferencia (`paged_attention`, `awq_quantizer`, `grammar_sampler`), interpretabilidad (`rome`, `dla`, `refusal_direction`) y exportador GGUF v3.

---

## 🛠️ Consola Maestra (`run.py`)

Todas las funciones y herramientas del repositorio se ejecutan a través del CLI maestro:

```bash
# Inferencia y Chat
python run.py standalone-chat      # Chat nativo 100% PyTorch
python run.py chat                 # Chat interactivo estándar
python run.py benchmark            # Profiling de tokens/seg y memoria VRAM

# Pruebas e Inspección
python run.py test                 # 58 pruebas unitarias (58/58 OK)
python run.py inspect              # Rayos X del Residual Stream en vivo
python run.py logit-lens           # Desproyección semántica capa por capa

# Experimentos Mecanicistas
python run.py causal               # Ablación de capas y trasplante causal
python run.py heads                # Censo de las 448 cabezas de atención
python run.py dissect              # Disección SVD de la cabeza de inducción L16H14
python run.py suppression          # Circuito inhibidor de copia
python run.py spectral             # Análisis de rango efectivo y dimensión 1793
python run.py surgery              # Cirugía de memorias factuales ROME en MLP
python run.py refusal              # Extracción y ablación del vector de rechazo
python run.py compare              # Comparativa con Llama 3 y Gemma 2
```

---

## 📄 Licencia

Este proyecto educativo está basado en la arquitectura de **Qwen3-1.7B** creada por The Qwen Team (Alibaba Cloud) y se distribuye bajo la licencia **Apache 2.0**.
