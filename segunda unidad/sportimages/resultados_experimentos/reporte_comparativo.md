# Comparativa Sistemática de los 5 Experimentos de CNN
## Proyecto: Clasificador de Deportes (*Sportimages*)

Se ejecutaron 5 variantes de la red neuronal convolucional variando de manera controlada:
1. **La red de aprendizaje**: optimizadores (SGD vs Adam) y tasas de aprendizaje.
2. **El número de épocas**: 10, 12 y 15 épocas.
3. **El número de imágenes por lote (`batch_size`)**: 64 y 32.
4. **El número de capas de convolución y arquitectura**: 1 capa (pág. 79), 2 capas y 3 capas con BatchNormalization.

---
### Tabla Comparativa Resumen

| Exp | Nombre / Movimiento | Capas Conv | Optimizador / LR | Batch | Épocas | Train Acc | Val Acc | Test Acc | Test Loss | Tiempo |
|---|---|---|---|---|---|---|---|---|---|---|
| **EXP1** | Experimento 1 (Baseline Libro Pág. 79) | 1 Conv (32) | SGD (0.001) | 64 | 10 | **57.93%** | **75.25%** | **75.24%** | 0.9587 | 117.5s |
| **EXP2** | Experimento 2 (Aumento Capas Convolución) | 2 Conv (32, 64) | SGD (0.005) | 64 | 10 | **86.18%** | **90.29%** | **90.63%** | 0.3075 | 289.5s |
| **EXP3** | Experimento 3 (Variación Batch Size y Épocas) | 2 Conv (32, 64) | SGD (0.005) | 32 | 15 | **94.23%** | **97.24%** | **97.28%** | 0.0917 | 560.7s |
| **EXP4** | Experimento 4 (Optimizador Adam / Tasa Adaptativa) | 2 Conv (32, 64) | Adam (0.001) | 64 | 10 | **99.09%** | **99.45%** | **99.62%** | 0.0134 | 369.5s |
| **EXP5** | Experimento 5 (Arquitectura Profunda + BatchNorm) | 3 Conv+BN (32, 64, 128) | Adam (0.0008) | 64 | 12 | **99.42%** | **99.72%** | **99.77%** | 0.0076 | 860.5s |

---
### Análisis Detallado de Cada Movimiento (Prueba y Error)

#### Experimento 1 (Baseline Libro Pág. 79)
- **Precisión en Test**: 75.24% (11607 aciertos vs 3819 errores de 15426 imágenes).
- **Pérdida en Test**: 0.9587
**Línea Base (Pág. 79 del libro)**:
- *Configuración*: 1 capa convolucional con 32 filtros, Dense(32), Dropout al 50% y SGD lr=0.001.
- *Efecto del movimiento*: La red presenta subajuste (underfitting). Con 1 sola capa convolucional y SGD a baja velocidad, la red apenas extrae bordes simples y le cuesta separar clases complejas. El alto Dropout (0.5) en una red tan pequeña frena el aprendizaje.
- **Archivos generados**:
  - Curvas de entrenamiento: `resultados_experimentos/exp1_curvas.png`
  - Análisis visual de aciertos/errores: `resultados_experimentos/exp1_analisis_imagenes.png`
  - Matriz de confusión: `resultados_experimentos/exp1_matriz_confusion.png`

#### Experimento 2 (Aumento Capas Convolución)
- **Precisión en Test**: 90.63% (13981 aciertos vs 1445 errores de 15426 imágenes).
- **Pérdida en Test**: 0.3075
**Movimiento 1 (Añadir 2ª Capa Convolucional)**:
- *Configuración*: Conv2D(32) -> Conv2D(64), reducción de Dropout a 0.25 y aumento moderado de LR (0.005).
- *Efecto del movimiento*: La precisión da un salto notable. Al duplicar la jerarquía de convolución, los filtros de la segunda capa pueden combinar líneas simples en texturas deportivas (redes de tenis, balones, cascos, autos de F1). Menos Dropout permite que la red capture patrones con mayor nitidez.
- **Archivos generados**:
  - Curvas de entrenamiento: `resultados_experimentos/exp2_curvas.png`
  - Análisis visual de aciertos/errores: `resultados_experimentos/exp2_analisis_imagenes.png`
  - Matriz de confusión: `resultados_experimentos/exp2_matriz_confusion.png`

#### Experimento 3 (Variación Batch Size y Épocas)
- **Precisión en Test**: 97.28% (15007 aciertos vs 419 errores de 15426 imágenes).
- **Pérdida en Test**: 0.0917
**Movimiento 2 (Reducción de Batch Size a 32 y Aumento a 15 Épocas)**:
- *Configuración*: Batch size reducido a la mitad (32) y 5 épocas adicionales de entrenamiento.
- *Efecto del movimiento*: Al reducir el batch_size, se duplican las actualizaciones de pesos por época (de ~770 a ~1540 pasos por época). Esto introduce un gradiente más estocástico que ayuda a escapar de mínimos locales y mejora la precisión general, a cambio de tardar ligeramente más por época.
- **Archivos generados**:
  - Curvas de entrenamiento: `resultados_experimentos/exp3_curvas.png`
  - Análisis visual de aciertos/errores: `resultados_experimentos/exp3_analisis_imagenes.png`
  - Matriz de confusión: `resultados_experimentos/exp3_matriz_confusion.png`

#### Experimento 4 (Optimizador Adam / Tasa Adaptativa)
- **Precisión en Test**: 99.62% (15368 aciertos vs 58 errores de 15426 imágenes).
- **Pérdida en Test**: 0.0134
**Movimiento 3 (Cambio a Optimizador Adam con Tasas Adaptativas)**:
- *Configuración*: Mismo esqueleto de 2 capas pero sustituyendo SGD por Adam (lr=0.001).
- *Efecto del movimiento*: Adam supera drásticamente a SGD. Al mantener momentos de primer y segundo orden para cada parámetro individual, la red converge muchísimo más rápido hacia una región de mínimo error. En pocas épocas supera con holgura al modelo del libro.
- **Archivos generados**:
  - Curvas de entrenamiento: `resultados_experimentos/exp4_curvas.png`
  - Análisis visual de aciertos/errores: `resultados_experimentos/exp4_analisis_imagenes.png`
  - Matriz de confusión: `resultados_experimentos/exp4_matriz_confusion.png`

#### Experimento 5 (Arquitectura Profunda + BatchNorm)
- **Precisión en Test**: 99.77% (15390 aciertos vs 36 errores de 15426 imágenes).
- **Pérdida en Test**: 0.0076
**Movimiento 4 (Arquitectura Profunda de 3 Bloques + BatchNormalization)**:
- *Configuración*: 3 bloques convolucionales (32, 64 y 128 filtros) con BatchNormalization en cada etapa y Adam (lr=0.0008).
- *Efecto del movimiento*: Es el modelo con mejor precisión y menor pérdida en Test. La normalización por lotes (`BatchNormalization`) elimina el desplazamiento de covariables internas, estabilizando gradientes y permitiendo a las 128 neuronas de la capa densa clasificar con alta confianza incluso los deportes más difíciles.
- **Archivos generados**:
  - Curvas de entrenamiento: `resultados_experimentos/exp5_curvas.png`
  - Análisis visual de aciertos/errores: `resultados_experimentos/exp5_analisis_imagenes.png`
  - Matriz de confusión: `resultados_experimentos/exp5_matriz_confusion.png`
