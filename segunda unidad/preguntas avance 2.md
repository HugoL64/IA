Pregunta 3 — ¿Qué cambia si meto más capas?

**1\. Tamaño tras el primer MaxPool(2x2):**  
Entrada de 21×28. Tras un MaxPooling de 2x2, las dimensiones se reducen a la mitad (aplicando división entera hacia abajo si es impar):  
Alto: ⌊21 / 2⌋ \= 10\. Ancho: ⌊28 / 2⌋ \= 14\.  
El tamaño aproximado será **10×14**.

**2\. ¿Qué detectan los 64 filtros de la segunda Conv2D?**  
Detectan **patrones más compuestos o abstractos** (como texturas complejas, esquinas, o partes de objetos como "un ojo" o "una rueda"). Como una lupa que se aleja, la primera capa detectó líneas o bordes simples (píxeles crudos); la segunda capa combina esos bordes simples para formar figuras más estructuradas.

**3\. Riesgo y medida de mitigación:**  
**Falla:** Verías un claro **sobreajuste (overfitting)**: la curva de *loss* de entrenamiento bajaría casi a 0, pero la curva de *loss* de validación se estancaría o empezaría a subir.  
**Medida:** Aplicar **Data Augmentation** (o Dropout).  
*Justificación:* Si hay pocas imágenes, rotarlas, hacerles zoom o voltearlas horizontalmente (Data Augmentation) obliga a la red a no memorizar características irrelevantes y aprender representaciones más robustas, simulando tener un dataset más grande.

Pregunta 4 — Formas: de volumen a vector

**1\. ¿Cuántos números salen del Flatten?**  
Multiplicamos las dimensiones del volumen: 7 × 7 × 64 \= **3,136 números**.

**2\. Parámetros de la Dense(32) siguiente:**  
Entradas (3,136) × Neuronas (32) \+ Biases (32) \= 100,352 \+ 32 \= **100,384 parámetros**.

**3\. Global Average Pooling en lugar de Flatten:**  
La forma de salida sería un vector *(64,)* conteniendo exactamente **64 valores** (un promedio único por cada mapa de características).  
*Ventaja:* Destruye la dependencia de la resolución espacial (puedes meter imágenes de tamaños variables) y previene masivamente el sobreajuste al reducir los millones de parámetros que se generarían al conectar un mapa grande a una capa Densa.

Pregunta 5 — Softmax y el error “casi seguro”

**1\. ¿Qué clase predice el modelo?**  
El modelo predice la clase con el índice de mayor probabilidad (*argmax*), que corresponde a \[0.85\]. Es decir, predice la tercera clase: **tornado**.

**2\. ¿Está “seguro” o “confundido”?**  
Está **muy seguro, pero terriblemente equivocado**. Al asignar un 85% (0.85) a "tornado", la red denota un alto nivel de confianza estadística en una respuesta que es incorrecta, dejándole apenas un 5% (0.05) a la etiqueta verdadera ("incendio").

**3\. Categorical crossentropy (alta o baja):**  
La pérdida será **muy alta**. La función *categorical crossentropy* evalúa el logaritmo negativo de la probabilidad predicha *solo* para la clase correcta. Como la red asignó una probabilidad minúscula (0.05) a la clase correcta, el castigo (-log(0.05)) disparará el valor de la pérdida.

\---  
Pregunta 6 — Padding, stride y “se me achicó la imagen”

**1\. Tabla:**

\| padding \| stride \| salida (alto×ancho) \|  
\| \--- \| \--- \| \--- \|  
\| valid \| 1 \| \*\*28×28\*\* (Fórmula: 32-5/1 \+ 1\) \|  
\| valid \| 2 \| \*\*14×14\*\* (Fórmula: ⌊(32-5)/2⌋ \+ 1 \= ⌊13.5⌋ \+ 1 \= 14\) \|  
\| same \| 1 \| \*\*32×32\*\* (Same pad con stride 1 mantiene el tamaño) \|

**2\. Problema con los 3 MaxPool(2x2):**  
El tamaño espacial se reduciría drásticamente: 32 → 16 → 8 → 4\. El mapa de características de 4x4 es tan pequeño que **la información espacial de los detalles finos se destruye**. Los objetos pequeños terminarían colapsados en el resumen de un solo píxel, haciendo imposible distinguirlos.

Pregunta 7 — Dropout: ¿dónde y para qué?

**1\. Entrenamiento:**  
En promedio, el 50% de las neuronas se apagan por cada paso (*batch*). De las 32 neuronas, **16 sobrevivirán** y 16 se establecerán en 0\.

**2\. Inferencia (predict):**  
**No, no se apagan neuronas al azar.** Durante inferencia necesitamos la capacidad total (determinista) de la red (el "ensamble" completo). En su lugar, los pesos de todas las neuronas activas se escalan automáticamente por la probabilidad de supervivencia (o se aplica *inverted dropout* en entrenamiento) para mantener la misma magnitud de salida esperada.

**3\. Crítica al Dropout solo en la última época:**  
No tiene sentido matemático ni práctico. El Dropout funciona forzando a la red a no depender de co-adaptaciones específicas de neuronas *durante todo el proceso de aprendizaje*. Ponerlo solo al final no le dará tiempo a la red para reajustar los pesos de forma robusta; de hecho, solo romperá de golpe las representaciones aprendidas y arruinará el desempeño final.

Pregunta 8 — Diseño: 5 clases de animales

**1\. Neuronas y activación final:**  
Debe tener **5 neuronas** (una por cada clase) y usar la función de activación **Softmax**, ya que queremos que las salidas sean una distribución de probabilidad mutuamente excluyente que sume 1\.

**2\. Forma antes del 'fit':**  
Si el modelo espera (21, 28, 3), se **deben redimensionar (resize)** todas las imágenes de 224x224 a 21x28 píxeles antes de dárselas a la red.

**3\. Mejora de arquitectura:**  
**Mejora:** Agregar **Batch Normalization (BatchNorm)** después de las capas Conv2D.  
**Síntoma:** El síntoma que indicaría su necesidad es que la curva de *loss* de entrenamiento sea muy errática (con muchos picos y valles a lo largo de las épocas) o que el modelo sufra de "desvanecimiento del gradiente", tardando una infinidad de épocas en converger.

Pregunta 9 — CNN vs MLP vs RNN

**1\. Predecir alumno aprueba (promedio, faltas, estudio):**  
**MLP**, porque los datos son estructurados y tabulares, sin ninguna relación espacial bidimensional o secuencia temporal entre sus columnas.  
**2\. Fotos de semáforo:**  
**CNN**, porque la información es espacial (una imagen 2D) y necesitamos invariancia traslacional para encontrar la bombilla roja sin importar si la cámara vibró o movió el encuadre.  
**3\. Autocompletar código:**  
**RNN/Transformer**, porque el problema requiere manejar información puramente secuencial donde el contexto previo (orden de las palabras anteriores) define la siguiente salida.

**Sobre el semáforo con MLP:**  
Sí, puede *aprender* (incluso memorizar el set de entrenamiento si es lo bastante grande), pero **sacrificas la invariancia a la traslación y la eficiencia**. Si el semáforo aparece un píxel a la derecha de donde lo vio entrenando, el vector aplastado será completamente distinto y el modelo fallará, además de requerir una cantidad brutalmente ineficiente de parámetros.

Pregunta 10 — Experimento mental: mover un gato

**1\. ¿Por qué la CNN acertaría al correrlo?**  
Gracias al concepto de **pesos compartidos**. La CNN desliza el mismo kernel por toda la imagen; por lo tanto, el filtro diseñado para encontrar la oreja del gato la detectará sin importar si está 10 píxeles a la derecha o a la izquierda. Un MLP, al depender de píxeles absolutos anclados, vería vectores inconexos.

**2\. ¿Giro de 90° (rotación)?**  
**La CNN fallaría.** Las CNN clásicas tienen invariancia a la traslación, pero *no* invariancia innata a la rotación. Para solucionarlo, haría falta aplicar **Data Augmentation** rotando aleatoriamente los gatos durante la etapa de entrenamiento para que los kernels aprendan a detectar bordes en esos ángulos.

**3\. Pregunta para un compañero (Kernel weights):**  
*"Si tienes un kernel de 3x3 lleno enteramente de 1s y lo pasas sobre una imagen de ruido, ¿qué efecto visual y matemático tendrá sobre el mapa resultante de salida?"* (Respuesta esperada: Es un filtro de emborronamiento/desenfoque (blur/box blur) y matemáticamente está calculando una sumatoria general sin detectar bordes).  
