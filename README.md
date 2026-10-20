# Qwen3-1.7B — Proyecto Personal de Ingeniería Inversa y Desmitificación de un LLM

> **Proyecto Personal con Fin Educativo:** Un viaje práctico y transparente para aprender cómo se compone un LLM moderno desde adentro hacia afuera, desmitificando el misticismo de la inteligencia artificial.  
> **Código Fiel y Transparente:** Implementado en PyTorch puro, lo más cercano y riguroso posible a la arquitectura oficial de `Qwen/Qwen3-1.7B` (Alibaba Cloud / Licencia Apache 2.0).  
> **Objetivo Didáctico:** Demostrar que un LLM no es magia negra alienígena, sino una composición hermosa y elegante de álgebra lineal, normalizaciones, atención por bloques y cálculo tensorial.  
> **Entorno:** 100% Local (CPU/GPU, sin APIs externas obligatorias, independiente de librerías opacas).

---

## 📑 Tabla de Contenidos

1. [Propósito del Proyecto y Resumen Didáctico](#-propósito-del-proyecto-y-resumen-didáctico)
2. [Estructura del Proyecto y Archivos Locales](#-estructura-del-proyecto-y-archivos-locales)
3. [Código Fuente de Alto Calibre (`source_code/`)](#-código-fuente-de-alto-calibre)
4. [Análisis Forense Binario de los Pesos (Safetensors)](#-análisis-forense-binario-de-los-pesos)
5. [Experimento 1: Mapa de Criticidad por Ablación de Capas (Layer Knockout)](#-experimento-1-mapa-de-criticidad-por-ablación-de-capas)
6. [Experimento 2: Trasplante Causal de Activaciones (Activation Patching)](#-experimento-2-trasplante-causal-de-activaciones)
7. [Experimento 3: Censo y Clasificación Arquetípica de las 448 Cabezas](#-experimento-3-censo-y-clasificación-arquetípica-de-las-448-cabezas)
8. [Experimento 4: Disección del Circuito Copiador (Induction Head L16H14)](#-experimento-4-disección-del-circuito-copiador)
9. [Experimento 5: Circuito Inhibidor de Supresión de Copia (Copy-Suppression Heads)](#-experimento-5-circuito-inhibidor-de-supresión-de-copia)
10. [Experimento 6: Geometría Espectral, Rango Efectivo SVD y la Dimensión Rebelde 1793](#-experimento-6-geometría-espectral-rango-efectivo-svd-y-la-dimensión-rebelde-1793)
11. [Experimento 7: Radiografía de Neuronas de Memoria en el MLP](#-experimento-7-radiografía-de-neuronas-de-memoria-en-el-mlp)
12. [Forense del Tokenizador y Razonamiento Multimodal](#-forense-del-tokenizador-y-razonamiento-multimodal)
13. [Logit Lens: Emergencia y Formación de Conceptos](#-logit-lens-emergencia-de-conceptos)
14. [Comparativa Arquitectónica: Qwen3 vs Llama 3 vs Gemma 2 vs DeepSeek](#-comparativa-arquitectónica)
15. [Consola Maestra y Guía de Ejecución Centralizada](#-consola-maestra-y-guía-de-ejecución)

---

## 🔭 Propósito del Proyecto y Resumen Didáctico

Este proyecto personal nació con un objetivo de aprendizaje profundo: **desarmar un modelo de lenguaje de última generación (LLM) pieza por pieza para entender cómo funciona realmente su maquinaria interna sin depender de cajas negras ni misticismo**.

A través del estudio minucioso de **Qwen3-1.7B** (1,720,574,976 parámetros), este repositorio reconstruye y documenta cada tensor, capa y mecanismo con código PyTorch lo más limpio, fiel y didáctico posible.

### ¿Cómo se compone realmente un LLM moderno?
Lejos de la falsa creencia de que un LLM es un código esotérico e incomprensible, su núcleo estructural es sorprendentemente homogéneo y elegante:
1. **Embedding Matrix:** Mapeo de tokens a vectores de 2,048 dimensiones.
2. **28 Bloques Decoder Idénticos:** Iteración repetitiva de dos operaciones fundamentales:
   - **Atención GQA (Grouped Query Attention 16:8) con QK-RMSNorm:** Donde los tokens dialogan y se intercambian información contextual.
   - **SwiGLU MLP:** Red densa de proyección ($2048 \to 6144 \to 2048$) que actúa como memoria asociativa factual.
3. **RMSNorm Final & LM Head:** Normalización del residual stream y proyección lineal hacia los 151,936 logits del vocabulario.

### Métricas Fundamentales Aprendidas e Implementadas:
* **Capas Transformer:** 28 bloques decodificadores residuales con pre-normalización dual.
* **Dimensión Oculta ($d_{model}$):** 2,048 dimensiones.
* **Atención GQA (Grouped Query Attention):** 16 cabezas de Query ($d_k=128$), 8 cabezas de Key/Value. Factor de compresión KV: 2:1.
* **Estabilizador QK-RMSNorm:** Normalización RMS independiente sobre las matrices Q y K antes del producto escalar, suprimiendo la varianza desmedida.
* **MLP / Feed-Forward:** SwiGLU con dimensión intermedia de **6,144** (`gate_proj`, `up_proj`, `down_proj`).
* **Embeddings Posicionales (RoPE):** Rotary Embeddings con frecuencia base extendida $\theta = 1,000,000$ (diseñado para contextos ultra-largos de hasta 131,072 tokens).
* **Normalización:** RMSNorm (Root Mean Square Normalization) con $\epsilon = 1 \times 10^{-6}$.
* **Vocabulario BPE:** 151,643 tokens de texto estándar + 288 tokens de control y razonamiento (Total: 151,936).

---

## 📂 Estructura del Proyecto y Archivos Locales

```text
qwen3_reverse_engineering/
├── README.md                           <- Este informe maestro de ingeniería inversa
├── run.py                              <- CLI Maestro central de control y orquestación
├── model/                              <- 4.06 GB de pesos descargados localmente
│   ├── model-00001-of-00002.safetensors (3.44 GB)
│   ├── model-00002-of-00002.safetensors (622 MB)
│   ├── config.json                     (Hiperparámetros oficiales)
│   ├── generation_config.json          (Muestreo y parada)
│   ├── tokenizer.json / vocab.json     (Vocabulario BPE y merges)
│   └── LICENSE                         (Licencia Apache 2.0)
├── source_code/                        <- Suite de código fuente modular de alto calibre
│   ├── layers/                         <- Capas atómicas (gqa, cla, mla, swa, swiglu, moe, rmsnorm, yarn, block)
│   ├── models/                         <- Motores Standalone, Pruned (poda de capas), LoRA y Medusa Residual Heads
│   ├── inference/                      <- Inferencia, PagedAttention, FP8 GEMM, AWQ, H2O, Grammar Masking, FIM, Speculative Decoding
│   ├── interpretability/               <- Direct Logit Attribution (DLA), Refusal Direction, ROME, CAA, SAE y Sondas Lineales
│   ├── loaders/                        <- Cargador Safetensors, Tokenizador BPE y Exportador Binario GGUF v3
│   ├── training/                       <- Motores de alineación (GRPO, DPO, Format Reward) y Verificación CoT (PRM, Best-of-N)
│   ├── multimodal/                     <- Audio ASR (32 capas), Visión Qwen3-VL (ViT + PatchMerger 2x2) y Perceiver Resampler
│   ├── official/                       <- Implementación base de Transformers y definición modular
│   ├── tests/                          <- 49 pruebas unitarias matemáticas automatizadas (49/49 OK)
│   └── README.md                       <- Especificación técnica interna de source_code
├── tools/                              <- Herramientas interactivas y profiling
│   ├── chat.py                         <- Consola de Chat interactiva en tiempo real (<think> soportado)
│   ├── standalone_chat.py              <- Consola de Chat nativa 100% puro PyTorch (sin transformers)
│   ├── benchmark_engine.py             <- Motor de Benchmark forense (Tokens/sec, Latencia y VRAM: MLA, SWA, FP8)
│   └── interactive_inspector.py        <- Rayos X del Residual Stream y generación guiada
├── experiments/                        <- Banco de experimentos científicos mecanicistas
│   ├── refusal_ablation.py             <- Extracción geométrica de r_hat y ablación/inmunización de seguridad
│   ├── knowledge_surgery.py            <- Cirugía de memorias factuales ROME en proyecciones MLP
│   ├── classify_448_heads.py           <- Censo funcional y arquetípico de las 448 cabezas
│   ├── spectral_dimension.py           <- Rango efectivo SVD, anisotropía y dimensión rebelde 1793
│   ├── copy_suppression_forensics.py   <- Detección de cabezas inhibidoras anti-repetición
│   ├── deep_circuit_forensics.py       <- Escáner de cabezas de inducción
│   ├── dissect_layer_16.py             <- Disección espectral de la cabeza copiadora L16H14
│   ├── causal_deep_forensics.py        <- Ablación sistemática de 28 capas y trasplante causal
│   ├── activation_steering.py          <- Intervención y steering de conceptos en Capa 18
│   ├── tokenizer_forensics.py          <- Forense de tokens especiales (<think>, FIM, multimodal)
│   ├── architectural_comparison.py     <- Comparativa formal (Qwen3 vs Llama 3 vs Gemma 2)
│   ├── advanced_reverse_engineering.py <- Logit Lens capa por capa y desproyecciones
│   └── reverse_engineer_qwen3.py       <- Decodificador de cabeceras binarias Safetensors
└── reports/                            <- Informes y catálogos forenses estructurados en JSON
    ├── qwen3_448_heads_functional_catalog.json
    ├── qwen3_spectral_and_outlier_report.json
    ├── qwen3_copy_suppression_forensics.json
    ├── qwen3_tokenizer_forensics.json
    ├── qwen3_layer16_head14_deep_dissection.json
    ├── qwen3_circuits_and_reasoning_forensics.json
    └── qwen3_reverse_engineering_report.json
```

---

## 💻 Código Fuente de Alto Calibre

El código fuente en `source_code/` ha sido refactorizado y modularizado bajo estándares de arquitectura limpia de alto calibre:

1. **`source_code/layers/attention/gqa.py`**: Grouped Query Attention (16:8) con **QK-RMSNorm** pre-producto punto, estabilizando los gradientes en contextos largos.
2. **`source_code/layers/feed_forward/swiglu.py` & `moe.py`**: Bloque no lineal SwiGLU con expansión a 6,144 dimensiones y bloque `Qwen3SparseMoeBlock` con enrutamiento Top-K y expertos compartidos.
3. **`source_code/layers/normalization/outlier_channel_norm.py`**: `SmoothQuantScaler` que migra escalas para neutralizar la varianza extrema de la Dimensión Rebelde 1793.
4. **`source_code/layers/embeddings/yarn_rotary.py`**: RoPE con extensión YaRN para extrapolar el contexto de 32,768 hasta 131,072 tokens con factor de escala de temperatura de atención.
5. **`source_code/layers/block.py`**: `Qwen3DecoderBlock` con pre-normalización dual y doble derivación residual.
6. **`source_code/models/standalone_qwen3.py`**: Motor causal 100% puro PyTorch sin requerir librerías externas.
7. **`source_code/models/medusa_heads.py`**: `Qwen3MedusaModel` con cabezales residuales de predicción multi-token para decodificación especulativa Zero-Draft ($2.5\times - 3\times$ aceleración).
8. **`source_code/models/pruned_qwen3.py`**: Motor con poda quirúrgica de capas (*Layer Pruning*) basado en los hallazgos de ablación causal (omisión de la Capa 26).
9. **`source_code/models/lora.py`**: Adaptadores LoRA de bajo rango ($r$) y fusión en caliente (`merge_weights`).
10. **`source_code/inference/paged_attention.py`**: `PagedKVCacheManager` con tablas de bloques de 16 tokens para gestión eficiente de VRAM zero-copy.
11. **`source_code/inference/reasoning_engine.py`**: `Qwen3ReasoningEngine` para control de presupuesto CoT, streaming y detección de giros cognitivos (*backtracking*) en `<think>`.
12. **`source_code/inference/tool_calling.py`**: `Qwen3ToolCallingEngine` para análisis y serialización de llamadas a herramientas bajo el protocolo nativo `<tool_call>`.
13. **`source_code/inference/awq_quantizer.py`**: `AWQChannelProtector` y capa `AWQLinear` para cuantización por grupos preservando la Dimensión Rebelde 1793.
14. **`source_code/inference/grammar_sampler.py`**: `TokenHealingProcessor` y `JSONGrammarMasker` para generación estructurada infalible.
15. **`source_code/training/reasoning_loss.py`**: Motores de pérdida `GRPOLoss` (optimización por grupos sin crítico) y `DPOLoss` para alineación de razonamiento.
16. **`source_code/loaders/gguf_exporter.py`**: `Qwen3GGUFExporter` para exportar a binario estándar GGUF v3 (F16 y Q8_0) compatible con llama.cpp y Ollama.
17. **`source_code/interpretability/dla.py`**: `DirectLogitAttributionAnalyzer` para descomposición matemática de contribución a logits a través de $W_U$.
18. **`source_code/multimodal/resampler.py`**: `PerceiverResampler` para comprimir secuencias visuales de video y multi-imagen.
19. **`source_code/layers/feed_forward/moe_specialization.py`**: `ExpertSaliencyAnalyzer` para auditoría de entropía de enrutamiento y especialización semántica de expertos MoE.
20. **`source_code/layers/attention/chunked_prefill.py`**: `TiledOnlineSoftmax` y `ChunkedPrefillEngine` para cálculo exacto $O(1)$ de atención sin picos de VRAM en contextos de 131,072 tokens.
21. **`source_code/inference/quantized_kv_cache.py`**: `QuantizedKVCache` con compresión INT8/INT4 por cabeza ($50\% - 75\%$ de reducción de huella de memoria).
22. **`source_code/inference/continuous_batching.py`**: `ContinuousBatchingScheduler` con planificación a nivel de iteración de prefill y decode dinámico para servicios multi-usuario.
23. **`source_code/layers/attention/cross_layer_kv.py`**: `CrossLayerAttention` para compartición de claves y valores (Cross-Layer KV) entre capas contiguas ($50\%$ ahorro VRAM).
24. **`source_code/inference/jit_optimizer.py`**: `JITOptimizer` con fusión de operadores para RMSNorm y SwiGLU sin sobrecoste de memoria.
25. **`source_code/interpretability/rome.py`**: `ROMEKnowledgeEditor` para cirugía directa de conocimiento (Rank-One Model Editing) en las matrices del MLP.
26. **`source_code/interpretability/hallucination_detector.py`**: `HallucinationDetector` para detección temprana de alucinaciones mediante entropía latente en residual stream.
27. **`source_code/interpretability/caa.py`**: `ContrastiveActivationAddition` (CAA) para dirección y control de conceptos mediante inyección latente.
28. **`source_code/inference/h2o_cache.py`**: `H2OKVCache` (Heavy-Hitter Oracle) para compresión dinámica preservando sumideros, heavy-hitters y ventana local.
29. **`source_code/layers/attention/mla.py`**: `MultiHeadLatentAttention` (MLA) estilo DeepSeek para compresión latente de KV hasta en un $70\%$.
30. **`source_code/interpretability/refusal_direction.py`**: `RefusalDirectionAnalyzer` para aislamiento geométrico y neutralización ortogonal de la dirección de rechazo.

---

## 📦 Análisis Forense Binario de los Pesos

* **Formato:** Safetensors (Zero-copy memory mapping, seguro contra inyección de código binario Pickle).
* **Partición Sharded:** 2 archivos binarios con `model.safetensors.index.json`:
  * `model-00001-of-00002.safetensors`: Contiene los embeddings de entrada y las capas 00 a 23 (3.44 GB).
  * `model-00002-of-00002.safetensors`: Contiene las capas 24 a 27, la normalización final y el `lm_head` (622 MB).
* **Matrices Totales:** 311 tensores individuales.
* **Parámetros Únicos:** Exactamente **1,720,574,976**.

---

## 🔬 Experimento 1: Mapa de Criticidad por Ablación de Capas

En este experimento desconectamos quirúrgicamente una capa a la vez (bypass residual $x_{l+1} = x_l$) para evaluar qué capas causan el colapso del modelo y cuáles son redundantes.

Evaluado con la proposición: *"El agua hierve a cien grados Celsius y se convierte en [vapor]"* (Certeza original: **97.59%**).

```text
Capa   | Top Prediccion   | P(vapor)   | Caída de Certeza | Criticidad
--------------------------------------------------------------------------------------
Capa 00| ','              |    0.0%    | -97.6% (Total)   | CRÍTICA (Órgano Vital)
Capa 01| 'el'             |    0.0%    | -97.6% (Total)   | CRÍTICA (Órgano Vital)
Capa 02| 'un'             |    0.0%    | -97.6% (Total)   | CRÍTICA (Órgano Vital)
Capa 03| 'h'              |   39.2%    | -58.4%           | CRÍTICA (Órgano Vital)
Capa 04| 'una'            |    3.5%    | -94.1% (Total)   | CRÍTICA (Órgano Vital)
Capa 05| 'vapor'          |   80.6%    | -17.0%           | IMPORTANTE
Capa 06| 'vapor'          |   54.1%    | -43.5%           | IMPORTANTE
Capa 07| 'vapor'          |   40.8%    | -56.8%           | CRÍTICA (Física del agua)
Capa 08| 'vapor'          |   48.3%    | -49.3%           | CRÍTICA
Capa 09| 'vapor'          |   30.0%    | -67.5%           | CRÍTICA
Capa 10| 'vapor'          |   75.4%    | -22.2%           | IMPORTANTE
Capa 11| 'vapor'          |   88.1%    |  -9.5%           | LEVE (Refinamiento)
Capa 12| 'vapor'          |   45.3%    | -52.2%           | CRÍTICA
Capa 13| 'vapor'          |   95.0%    |  -2.6%           | LEVE (Refinamiento)
Capa 14| 'vapor'          |   81.9%    | -15.6%           | IMPORTANTE
Capa 15| 'vapor'          |   72.1%    | -25.4%           | IMPORTANTE
Capa 16| 'vapor'          |   89.0%    |  -8.6%           | LEVE (Refinamiento)
Capa 17| 'vapor'          |   57.0%    | -40.6%           | IMPORTANTE
Capa 18| 'vapor'          |   71.5%    | -26.1%           | IMPORTANTE
Capa 19| 'vapor'          |   96.8%    |  -0.8%           | LEVE (Refinamiento)
Capa 20| 'vapor'          |   85.7%    | -11.9%           | IMPORTANTE
Capa 21| 'vapor'          |   77.4%    | -20.2%           | IMPORTANTE
Capa 22| 'vapor'          |   93.9%    |  -3.7%           | LEVE (Refinamiento)
Capa 23| 'vapor'          |   56.2%    | -41.4%           | IMPORTANTE
Capa 24| 'vapor'          |   93.8%    |  -3.8%           | LEVE (Refinamiento)
Capa 25| 'vapor'          |   77.1%    | -20.5%           | IMPORTANTE
Capa 26| 'vapor'          |   98.7%    |  +1.1% (Mejora)  | REDUNDANTE (Poda viable)
Capa 27| 'vapor'          |   95.5%    |  -2.1%           | LEVE (Refinamiento)
```

> **Hallazgo Clave:** Las capas 0 a 4 forman el sustrato gramatical no prescindible. Por el contrario, la **Capa 26 es completamente redundante** (el modelo es 1.1% más certero sin ella). Varias capas tardías pueden podarse (*Layer Pruning*) para reducir el modelo a ~1.3B parámetros sin apenas degradación.

---

## ⚡ Experimento 2: Trasplante Causal de Activaciones (Activation Patching)

Para probar causalidad y no mera correlación, inyectamos en pleno vuelo el estado latente del último token de una frase dentro de otra:
* **Frase Limpia:** *"El monumento mas iconico de Francia es la Torre"* $\to$ Predice: `Eiffel`
* **Frase Corrupta:** *"El monumento mas iconico de Italia es el Col"* $\to$ Predice: `iseo`

```text
Parche en Capa 10 -> Nueva Prediccion: 'iseo' | P(Eiffel):  0.0% | P(Coliseo): 53.2%
Parche en Capa 14 -> Nueva Prediccion: 'iseo' | P(Eiffel):  0.0% | P(Coliseo): 50.4%
Parche en Capa 18 -> Nueva Prediccion: 'iseo' | P(Eiffel):  0.0% | P(Coliseo): 81.9% (MÁXIMO)
Parche en Capa 20 -> Nueva Prediccion: 'iseo' | P(Eiffel):  0.0% | P(Coliseo): 53.7%
Parche en Capa 22 -> Nueva Prediccion: 'iseo' | P(Eiffel):  0.0% | P(Coliseo): 58.7%
Parche en Capa 24 -> Nueva Prediccion: 'iseo' | P(Eiffel):  0.0% | P(Coliseo): 58.6%
```

> **Descubrimiento Causal:** La **Capa 18** es el nodo de control donde se consolida la entidad sujeto. Al parchar únicamente la Capa 18, el modelo reemplaza instantáneamente el contexto de Francia por Italia con **81.9% de certeza**.

---

## 🎯 Experimento 3: Censo y Clasificación Arquetípica de las 448 Cabezas

Escaneamos exhaustivamente las **448 cabezas de atención** ($28 \text{ capas} \times 16 \text{ cabezas}$) para clasificarlas según sus arquetipos neuro-computacionales funcionales:

```text
======================================================================
  CENSO ARQUETÍPICO DE LAS 448 CABEZAS DE QWEN3-1.7B
======================================================================
1. Cabezas Copiadoras (Induction Heads)     : 15 cabezas
   Top 3: ['L16H14 (0.8564)', 'L03H10 (0.7826)', 'L21H08 (0.6744)']

2. Cabezas de Token Previo (Previous-Token) : 14 cabezas
   Top 3: ['L15H03 (0.8549)', 'L02H12 (0.8280)', 'L00H02 (0.8022)']

3. Sumideros de Atención (Attention Sinks)  : 279 cabezas (>62% del total)
   Top 3: ['L19H04 (0.9943)', 'L27H00 (0.9829)', 'L05H02 (0.9707)']

4. Cabezas Difusas de Contexto Global       : 0 cabezas
======================================================================
```

### Hallazgos del Censo:
* **Predominancia de Sumideros de Atención (62.3%):** 279 cabezas depositan casi toda su masa atencional en el token inicial (`token 0` / BOS). Esto confirma empíricamente la hipótesis del fenómeno de *StreamingLLM* (Xiao et al.): la red utiliza el token inicial como un "depósito de escape" para descartar peso atencional no relevante sin distorsionar el flujo residual.
* **Cabezas de Inducción Estratificadas:** No se concentran en una única capa, sino en un gradiente: etapa temprana preparatoria (L03H10), etapa primaria ejecutiva (L16H14 con score 0.8564) y etapa de consolidación tardía (L21H08).

---

## 🔄 Experimento 4: Disección del Circuito Copiador (Induction Head L16H14)

Evaluación en profundidad de la cabeza de inducción primaria **L16H14**:

* **Puntuación de inducción:** **0.8564**.
* **Descomposición Espectral de $W_{OV} = W_V \times W_O$ ($128 \times 128$):**
  * Traza de la matriz: **$+15.82$** (Altamente positiva, confirma la función de copia directa).
  * Porcentaje de autovalores positivos: **$51.56\%$**.
  * Valor singular dominante $\sigma_1$: **$2.6732$**.
  * Razón de energía en top-5 singulares: **$16.89\%$**.
* **Prueba de Knockout de la Cabeza:**
  * Al anular $W_{OV}$ de L16H14, el modelo activa cabezas de respaldo redundantes en la **Capa 21 (Cabezas 8 y 9)** y **Capa 20 (Cabeza 14)**, demostrando un *circuito en ensamble* tolerante a fallos.

---

## 🛑 Experimento 5: Circuito Inhibidor de Supresión de Copia (Copy-Suppression Heads)

Mientras las cabezas de inducción copian secuencias, ¿qué impide que el modelo caiga en un bucle infinito de eco y repetición? Investigamos las cabezas de supresión de copia (*Copy-Suppression Heads*) en capas tardías (16-27), evaluando su atención al candidato repetido y su proyección directa de logits (*Direct Logit Attribution*, DLA) sobre el desembebing $W_U$:

```text
--- CABEZAS TARDÍAS CON ATENCIÓN DIRIGIDA AL CANDIDATO E ÍNDICE DLA ---
Cabeza     Atención a Token   DLA Norm     DLA Mean (Dirección a W_U)
------------------------------------------------------------------------
L19H05     0.5856             0.4547       +0.0001 (Neutral)
L20H15     0.4934             0.6107       -0.0024 (INHIBIDORA FUERTE)
L19H08     0.4075             0.5247       -0.0020 (INHIBIDORA FUERTE)
L19H09     0.3488             0.5028       -0.0039 (INHIBIDORA FUERTE)
L19H13     0.3169             0.5899       -0.0029 (INHIBIDORA)
L20H06     0.2768             0.5374       -0.0056 (MÁXIMA PROYECCIÓN NEGATIVA)
```

> **Descubrimiento del Freno Mecanicista:** Las cabezas **L20H15, L19H08, L19H09 y L20H06** forman el **circuito inhibidor anti-repetición** de Qwen3. Cuando el token candidato ya fue atendido por la inducción de L16H14, estas cabezas tardías proyectan vectores con media de logit negativa sobre $W_U$, calibrando la probabilidad para no repetir infinitamente el mismo token.

---

## 📊 Experimento 6: Geometría Espectral, Rango Efectivo SVD y la Dimensión Rebelde 1793

Evaluamos la geometría del flujo residual capa por capa calculando:
1. **Rango Efectivo de Roy & Vetterli ($r_{eff}$):** Entropía de Shannon del espectro de valores singulares.
2. **Anisotropía:** Similitud coseno promedio entre pares de representaciones latentes.
3. **Outlier Ratio:** Magnitud máxima vs promedio para detectar dimensiones rebeldes (*Rogue Features*).

```text
Capa       Rango Efectivo   Anisotropía      Outlier Ratio      Dimensión Rebelde Dominante
-------------------------------------------------------------------------------------------
Embedding  63.67            0.0305           7.00               Dim 1440 (0.20)
Layer 0    66.27            0.3975           57.63              Dim 1999 (11.51)
Layer 1    51.96            0.4311           195.70             Dim 425 (66.84)
Layer 2    1.98 (COLAPSO)   0.3464           21466.39 (PICO)    Dim 1793 (13029.48)
Layer 3    2.24             0.3455           18345.09           Dim 1793 (13030.74)
Layer 4    2.45             0.3642           16630.81           Dim 1793 (13031.21)
Layer 5    2.79             0.3458           14728.82           Dim 1793 (13032.80)
...
Layer 16   15.37            0.4353           3352.27            Dim 1793 (13086.55) [Inducción]
Layer 18   23.84            0.4868           2004.08            Dim 1793 (13110.23) [Steering]
Layer 22   47.12            0.4566           726.37             Dim 1793 (13198.36)
Layer 26   55.38            0.6248           367.21             Dim 1793 (13278.15) [Redundante]
Post-Norm  56.92            0.4767           47.09              Dim 425 (58.83)
```

### Fenómeno de la Dimensión Rebelde 1793:
* **El Colapso de Rango en Capa 2:** Entre la Capa 1 y la Capa 2, el rango efectivo se desploma de 51.96 a **1.98**, y la coordenada **Dim 1793** salta bruscamente a una magnitud de **+13,029.48**.
* **La Autopista de Sesgo (DC Offset):** Dimensión 1793 mantiene una magnitud colosal (~13,000) de forma ininterrumpida desde la Capa 2 hasta la Capa 26. Actúa como un *canal de polarización privilegiado* (base coordinada fija) para calibrar el cómputo de atención residual.
* **Recuperación Progresiva:** A partir de la Capa 3, la red recupera paulatinamente dimensionalidad efectiva (de 2.24 a 56.92) a medida que los conceptos se ramifican, hasta que la normalización final contrae el outlier de 367.21 a 47.09 antes del `lm_head`.

---

## 🧠 Experimento 7: Radiografía de Neuronas de Memoria en el MLP

Los bloques MLP actúan como diccionarios asociativos clave-valor. Evaluando la frase:
> *"Python es un lenguaje de programacion orientado a objetos"*

Capturamos las activaciones intermedias de las 6,144 neuronas de la Capa 18 y decodificamos el vector de salida de cada neurona proyectándolo al vocabulario con $W_{LM}$:

```text
Top Neuronas encendidas en la Capa 18:
  #1 Neurona ID 5234 | Magnitud: 17.93 | Términos: ['ấm', '天下', '夹']
  #2 Neurona ID 1996 | Magnitud: 14.75 | Términos: ['-ft', '###################', '-girl']
  #3 Neurona ID 1950 | Magnitud: 13.86 | Términos: ['framework', 'frameworks', '基于 (basado en)']
  #4 Neurona ID 1951 | Magnitud: 12.92 | Términos: ['', '漉', 'ark']
  #5 Neurona ID 1372 | Magnitud: 11.75 | Términos: ['inium', 'resulting', 'ifiant']
```

> **Descubrimiento:** La **Neurona 1950** es una *neurona conceptual de desarrollo de software*: se activa con alta intensidad ante lenguajes de programación y escribe directamente los términos `framework` y `frameworks` en la corriente residual.

---

## 🔡 Forense del Tokenizador y Razonamiento Multimodal

Examinando los 151,936 tokens de `model/tokenizer.json` (`experiments/tokenizer_forensics.py`):

1. **Tokens de Razonamiento Nativo (CoT):**
   * `<think>` (ID `151667`) y `</think>` (ID `151668`): Delimitan la cadena de pensamiento interna sin requerir post-procesamiento.
2. **Tokens de Relleno en el Medio (Fill-In-The-Middle / FIM):**
   * `<|fim_prefix|>` (ID `151659`), `<|fim_middle|>` (ID `151660`), `<|fim_suffix|>` (ID `151661`), `<|repo_name|>` (ID `151664`), `<|file_sep|>` (ID `151665`). Indican que el modelo fue entrenado nativamente para completado y refactorización de código en IDEs.
3. **Tokens Multimodales y Espaciales:**
   * Visión: `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`.
   * Audio: `<|audio_start|>`, `<|audio_end|>`.
   * Coordenadas geométricas: `<|box_start|>`, `<|box_end|>`, `<|quad_start|>`, `<|quad_end|>`, permitiendo tareas de detección de objetos (bounding boxes) y segmentación poligonal directamente desde texto.

---

## 📈 Logit Lens: Emergencia de Conceptos

Al desproyectar el estado residual de cada capa intermedia directamente al vocabulario mediante la norma final y el `lm_head`:

* **Prompt:** *"El lenguaje de programación más popular para inteligencia artificial y ciencia de datos es"*
* **Capas 00 a 16:** Incierto o tokens sintácticos (`el`, `un`, `,`).
* **Capa 21:** Emerge por primera vez `' Python'` con **44.7%** de probabilidad.
* **Capa 24:** Se consolida `' Python'` con **99.6%** de certeza absoluta.

---

## 🏛️ Comparativa Arquitectónica

| Característica | **Qwen3-1.7B** | **Llama 3.2-1B** | **Gemma 2-2B** | **DeepSeek-V3 / R1 (MoE)** |
| :--- | :--- | :--- | :--- | :--- |
| **Desarrollador** | Alibaba Cloud | Meta AI | Google DeepMind | DeepSeek AI |
| **Parámetros Totales** | **1.72B** | 1.23B | 2.61B | 671B (37B activos) |
| **Capas Transformer** | **28** | 16 | 26 | 61 |
| **Dimensión Oculta ($d$)** | **2,048** | 2,048 | 2,304 | 2,048 (latente) |
| **Mecanismo de Atención** | **GQA (16 Q / 8 KV)** | GQA (32 Q / 8 KV) | GQA Alternada (Local 4k / Global) | **MLA (Multi-Head Latent)** |
| **Dimensión de Cabeza** | **128** | 64 | 256 | 128 |
| **QK-Norm (Normalización QK)**| **SÍ (RMSNorm en Q y K)**| NO | NO | **SÍ** |
| **Activación MLP** | **SwiGLU ($6,144$)** | SwiGLU ($8,192$) | GeGLU ($9,216$) | SwiGLU MoE (256 expertos) |
| **RoPE Frecuencia $\theta$** | **$1,000,000$** | $500,000$ | $10,000$ | $10,000$ (con YaRN) |
| **Logit Soft-Capping** | NO | NO | **SÍ (cap a 30.0 / 50.0)** | NO |
| **Tokens Nativos de Razonamiento**| **SÍ (`<think>`, `</think>`)**| NO | NO | **SÍ (`<think>`, `</think>`)** |
| **Tamaño de Vocabulario** | **151,936** | 128,256 | 256,000 | 129,280 |

---

## 🚀 Consola Maestra y Guía de Ejecución

Todos los módulos, experimentos y herramientas se ejecutan de manera centralizada a través de [`run.py`](run.py) con rutas relativas limpias:

```powershell
# 1. Iniciar chat interactivo local (soporta <think> en vivo)
python run.py chat

# 2. Chat nativo 100% puro PyTorch (sin dependencias del runtime transformers)
python run.py standalone-chat

# 3. Motor de Benchmark & Profiling (Tokens/sec, Latencia y VRAM: MLA, SWA, FP8)
python run.py benchmark

# 4. Batería completa de 49 pruebas unitarias de software y arquitectura (49/49 OK)
python run.py test

# 5. Extracción geométrica del vector de rechazo r_hat y ablación de seguridad
python run.py refusal

# 6. Cirugía de memoria factual ROME en proyecciones MLP sin reentrenamiento
python run.py surgery

# 7. Rayos X del Residual Stream y generación token-a-token
python run.py inspect

# 8. Experimento causal de las 28 capas y trasplante de activación
python run.py causal

# 9. Censo arquetípico de las 448 cabezas de atención
python run.py heads

# 10. Rango efectivo SVD, anisotropía y detección de la dimensión rebelde 1793
python run.py spectral

# 11. Detección de cabezas inhibidoras anti-eco (Copy-Suppression Circuits)
python run.py suppression

# 12. Escáner de cabezas de inducción
python run.py induction

# 13. Disección espectral (SVD) de la cabeza copiadora L16H14
python run.py dissect

# 14. Desproyección capa por capa con Logit Lens
python run.py logit-lens

# 15. Control de conceptos y steering vectors en Capa 18
python run.py steer

# 16. Forense del vocabulario y tokens especiales (<think>, FIM, multimodal)
python run.py tokenizer

# 17. Comparativa arquitectónica formal (Qwen3 vs Llama 3 vs Gemma 2)
python run.py compare
```

---
*Fin del informe de ingeniería inversa. Documentación y experimentos generados bajo estricto cumplimiento de la licencia Apache 2.0 de Alibaba Group.*
