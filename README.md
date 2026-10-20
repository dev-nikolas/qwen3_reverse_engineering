# Qwen3-1.7B — Reingenieria y Desmitificacion de un LLM

> **Estudio Forense y Mecanicista:** Reconstruccion integral en PyTorch puro de un Large Language Model moderno desde primeros principios matematicos, eliminando la opacidad en torno a la inteligencia artificial.

---

## Motivacion y Proposito

La mayoria de desarrolladores interactuan con los LLMs unicamente como consumidores de APIs o mediante librerias monoliticas que tratan la red como una caja negra. Este proyecto se desarrollo para:

1. **Desmitificar la Arquitectura:** Demostrar que un LLM de frontera es una composicion determinista y comprensible de algebra lineal, normalizaciones RMSNorm, atencion agrupada (GQA), compuertas SwiGLU y rotaciones RoPE.
2. **Interpretabilidad Mecanicista Profunda:** Localizar donde reside el conocimiento en la red mediante trasplante causal de activaciones, censo de cabezas de induccion y diseccion espectral de tensores de pesos.
3. **Control Causal en Tiempo Real:** Demostrar que es posible intervenir quirurgicamente el flujo residual latente para alterar el comportamiento cognitivo del modelo sin reentrenarlo.

---

## Funcionalidad Destacable: Chat Nativo con Modulacion Guiada (/steer)

El repositorio incluye un motor de inferencia interactivo en PyTorch puro (con dependencias unicamente en `torch` y `safetensors`) que ofrece:

* **Soporte Nativo de Razonamiento:** Renderizado en streaming en tiempo real de los bloques deliberativos internos (`<think>...</think>`).
* **Activacion Guiada por Capas (`/steer`):** Intervencion causal en caliente para modular la intensidad del vector latente de razonamiento en la Capa 18 (ej. `/steer reasoning 2.0`), forzando deducciones mas analiticas o directas sobre la marcha.
* **Control Termodinamico y Monitoreo:** Ajuste dinamico de temperatura (`/temp <f32>`) e inspeccion instantanea de normas residuales.

---

## Especificaciones de la Arquitectura

| Parametro | Valor Oficial Qwen3-1.7B | Funcion Matematica y Diseno |
| :--- | :--- | :--- |
| **Parametros Totales** | **1,720,574,976** | Pesos entrenables en `bfloat16` (4.06 GB sharded) |
| **Capas Transformer** | **28 bloques decodificadores** | Pre-normalizacion dual con conexiones residuales |
| **Dimension Oculta ($d_{model}$)** | **2,048** | Dimension del flujo residual latente |
| **Mecanismo de Atencion** | **GQA (16 Q / 8 KV)** | 16 cabezas de consulta y 8 de clave/valor (ratio 2:1) |
| **Dimension por Cabeza ($d_k$)** | **128** | Normalizadas individualmente con **QK-RMSNorm** |
| **MLP / Feed-Forward** | **SwiGLU ($d_{ff} = 6,144$)** | Compuertas no lineales `gate_proj`, `up_proj`, `down_proj` |
| **Embeddings Posicionales** | **RoPE ($\theta = 1,000,000$)** | Contexto extendible de 32k a 131k tokens con YaRN |
| **Normalizacion** | **RMSNorm ($\epsilon = 10^{-6}$)** | Pre-atencion y pre-MLP sin sesgo residual |
| **Vocabulario BPE** | **151,936 tokens** | Soporta razonamiento nativo (`<think>`) y relleno FIM |

---

## Inicio Rapido (Getting Started)

### 1. Requisitos e Instalacion
Recomendado Python 3.10+ con PyTorch:
```bash
pip install torch safetensors
```

### 2. Descargar los Pesos Oficiales (4.06 GB)
El repositorio incluye un script con descarga reanudable (HTTP Range) directo desde Hugging Face:
```bash
# En Windows:
download_weights.bat

# O directamente en Python:
python tools/download_weights.py
```

### 3. Chat Interactivo en Tiempo Real
Prueba la inferencia nativa en PyTorch puro con comandos de guiado:
```bash
python run.py standalone-chat
```

### 4. Validar Suite de Pruebas Unitarias
Ejecuta la verificacion automatizada de paridad matematica:
```bash
python run.py test
```

---

## Documentacion Tecnica Detallada

La investigacion forense completa se encuentra estructurada en documentos tematicos:

* **[Experimentos Forenses y Circuitos Mecanicistas](docs/experiments_and_circuits.md):**
  - Ablacion de capas (descubrimiento de la Capa 26 redundante).
  - Trasplante causal en Capa 18 (81.9% de certeza restaurada).
  - Censo de las 448 cabezas de atencion (62.3% sumideros de atencion).
  - Diseccion espectral de la cabeza copiadora L16H14.
  - Circuito inhibidor anti-repeticion (Copy-Suppression en L19/L20).
  - Dimension rebelde 1793 (+13,029 offset) y analisis SVD.
  - Neuronas de memoria factual en MLP (ej. Neurona 1950).

* **[Analisis Forense Binario y Tokenizador](docs/binary_forensics.md):**
  - Mapeo de memoria de 311 tensores Safetensors.
  - Tokens especiales de razonamiento (`<think>`) y codigo FIM.

* **[Logit Lens y Emergencia de Conceptos](docs/logit_lens.md):**
  - Desproyeccion del flujo residual capa por capa.

* **[Comparativa Arquitectonica](docs/architectural_comparison.md):**
  - Matriz analitica: Qwen3 vs Llama 3 vs Gemma 2 vs DeepSeek-V3/R1.

---

## Consola Maestra (`run.py`)

Todas las herramientas de analisis e inferencia se ejecutan centralizadamente:

```bash
# Inferencia y Benchmark
python run.py standalone-chat      # Chat interactivo con /steer y <think>
python run.py chat                 # Chat interactivo estandar
python run.py benchmark            # Medicion de tokens/seg y memoria VRAM

# Pruebas e Inspeccion
python run.py test                 # 58 pruebas unitarias (58/58 OK)
python run.py inspect              # Rayos X del Residual Stream en vivo
python run.py logit-lens           # Desproyeccion semantica capa a capa

# Investigacion Mecanicista
python run.py causal               # Ablacion y trasplante causal
python run.py heads                # Censo y clasificacion de cabezas
python run.py dissect              # Diseccion SVD de cabezas de induccion
python run.py suppression          # Evaluacion de supresion de copia
python run.py spectral             # Rango efectivo y dimensiones outlier
python run.py surgery              # Cirugia de memorias ROME en MLP
```

---

## Licencia

Este proyecto educativo esta basado en la arquitectura de **Qwen3-1.7B** creada por The Qwen Team (Alibaba Cloud) y se distribuye bajo la licencia **Apache 2.0**.
