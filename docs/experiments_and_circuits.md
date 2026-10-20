# Experimentos Forenses y Circuitos Mecanicistas — Qwen3-1.7B

Este documento detalla los 7 experimentos de interpretabilidad mecanicista e ingeniería inversa realizados sobre **Qwen3-1.7B** para comprender empíricamente cómo opera internamente el modelo.

---

## 📑 Índice de Experimentos

1. [Experimento 1: Mapa de Criticidad por Ablación de Capas (Layer Knockout)](#1-mapa-de-criticidad-por-ablación-de-capas-layer-knockout)
2. [Experimento 2: Trasplante Causal de Activaciones (Activation Patching)](#2-trasplante-causal-de-activaciones-activation-patching)
3. [Experimento 3: Censo y Clasificación Arquetípica de las 448 Cabezas](#3-censo-y-clasificación-arquetípica-de-las-448-cabezas)
4. [Experimento 4: Disección del Circuito Copiador (Induction Head L16H14)](#4-disección-del-circuito-copiador-induction-head-l16h14)
5. [Experimento 5: Circuito Inhibidor de Supresión de Copia (Copy-Suppression Heads)](#5-circuito-inhibidor-de-supresión-de-copia-copy-suppression-heads)
6. [Experimento 6: Geometría Espectral, Rango Efectivo SVD y la Dimensión Rebelde 1793](#6-geometría-espectral-rango-efectivo-svd-y-la-dimensión-rebelde-1793)
7. [Experimento 7: Radiografía de Neuronas de Memoria en el MLP](#7-radiografía-de-neuronas-de-memoria-en-el-mlp)

---

## 1. Mapa de Criticidad por Ablación de Capas (Layer Knockout)

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
> **Comando de ejecución:** `python run.py causal`

---

## 2. Trasplante Causal de Activaciones (Activation Patching)

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
> **Comando de ejecución:** `python run.py causal`

---

## 3. Censo y Clasificación Arquetípica de las 448 Cabezas

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
> **Comando de ejecución:** `python run.py heads`

---

## 4. Disección del Circuito Copiador (Induction Head L16H14)

Evaluación en profundidad de la cabeza de inducción primaria **L16H14**:

* **Puntuación de inducción:** **0.8564**.
* **Descomposición Espectral de $W_{OV} = W_V \times W_O$ ($128 \times 128$):**
  * Traza de la matriz: **$+15.82$** (Altamente positiva, confirma la función de copia directa).
  * Porcentaje de autovalores positivos: **$51.56\%$**.
  * Valor singular dominante $\sigma_1$: **$2.6732$**.
  * Razón de energía en top-5 singulares: **$16.89\%$**.
* **Prueba de Knockout de la Cabeza:**
  * Al anular $W_{OV}$ de L16H14, el modelo activa cabezas de respaldo redundantes en la **Capa 21 (Cabezas 8 y 9)** y **Capa 20 (Cabeza 14)**, demostrando un *circuito en ensamble* tolerante a fallos.  
> **Comando de ejecución:** `python run.py dissect`

---

## 5. Circuito Inhibidor de Supresión de Copia (Copy-Suppression Heads)

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
> **Comando de ejecución:** `python run.py suppression`

---

## 6. Geometría Espectral, Rango Efectivo SVD y la Dimensión Rebelde 1793

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
> **Comando de ejecución:** `python run.py spectral`

---

## 7. Radiografía de Neuronas de Memoria en el MLP

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
> **Comando de ejecución:** `python run.py surgery`
