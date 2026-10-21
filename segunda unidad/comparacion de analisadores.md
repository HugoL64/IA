## **Eigenfaces (Análisis de Componentes Principales \- PCA)**

Transforma las imágenes en un conjunto de características principales (eigenfaces) que representan la mayor varianza en los datos de entrenamiento, agrupando la información sin importar a qué persona pertenece cada rasgo.

* **Ventajas:**  
  * Matemáticamente simple y fácil de implementar.  
  * Alta velocidad de reconocimiento y extracción una vez que el modelo está entrenado.  
  * Reduce drásticamente la dimensionalidad y el peso en memoria de las imágenes originales.  
* **Desventajas:**  
  * Falla significativamente ante cambios de iluminación, sombras o variaciones en la expresión facial del sujeto.  
  * No optimiza la separación entre diferentes personas; solo captura la variación visual general de todas las caras en el conjunto de datos.

## **Fisherfaces (Análisis Discriminante Lineal \- LDA)**

Una evolución diseñada específicamente para la clasificación. En lugar de buscar la varianza total de las imágenes, busca los componentes que **separan mejor a un individuo de otro**, maximizando la distancia entre distintas personas y minimizando las diferencias en fotos de la misma persona.

* **Ventajas:**  
  * Maneja mucho mejor los cambios de iluminación y de expresión facial que Eigenfaces.  
  * Al estar enfocado en discriminar clases, es considerablemente más preciso para tareas strictly de identificación.  
* **Desventajas:**  
  * Requiere un mayor tiempo de cálculo matricial en la fase de entrenamiento.  
  * Al seguir siendo un método holístico (global), los cambios severos de pose (como girar el rostro de perfil) destruyen su precisión.

## **LBPH (Local Binary Patterns Histograms)**

A diferencia de los modelos anteriores, LBPH no mira la imagen como un todo. Divide el rostro en una cuadrícula de pequeños bloques, compara la intensidad de cada píxel central con sus vecinos para crear valores binarios, y extrae un histograma de características locales.

* **Ventajas:**  
  * Al comparar diferencias relativas entre píxeles contiguos en lugar de medir la luz absoluta, es prácticamente inmune a los cambios de iluminación exterior.  
  * Al analizar áreas locales, tolera mucho mejor las oclusiones parciales (uso de gafas, bufandas) y pequeños cambios de pose.  
  * Es capaz de aprender y reconocer rostros con alta precisión utilizando una sola fotografía de entrenamiento por usuario.  
* **Desventajas:**  
  * El tamaño de los histogramas concatenados puede ser muy grande, exigiendo mayor almacenamiento.  
  * La fase de predicción pura puede ser ligeramente más lenta en bases de datos masivas comparada con el cálculo vectorial simple de PCA.

