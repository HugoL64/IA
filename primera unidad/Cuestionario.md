# **Cuestionario de Inteligencia Artificial**

## Árboles de Decisión y Redes Neuronales Multicapa

**Estudiante:** Hugo Lemus

**Fecha:** 8 de octubre de 2026

# **Parte I. Conceptos y definiciones**

### **Pregunta 1**

**¿Qué es un árbol de decisión y cuál es su objetivo principal dentro de un problema de clasificación?**

Un árbol de decisión es un modelo que sirve para tomar decisiones a partir de diferentes condiciones. Va haciendo preguntas sobre los datos y, dependiendo de las respuestas, sigue diferentes caminos hasta llegar a un resultado.

Su principal objetivo en la clasificación es determinar a qué grupo o categoría pertenece un dato. Por ejemplo, podría utilizarse para saber si un estudiante está en riesgo de reprobar tomando en cuenta cosas como sus calificaciones, asistencia y tareas entregadas.

Una ventaja es que es fácil de entender porque podemos seguir el camino que tomó el árbol y ver por qué llegó a determinada decisión.

### **Pregunta 2**

**Explique con sus propias palabras los siguientes elementos de un árbol de decisión:**

* **Nodo raíz:** Es el primer punto del árbol y donde comienza la toma de decisiones. Aquí se encuentra la primera condición que se va a evaluar.  
* **Nodo interno:** Es un punto donde se hace otra pregunta o se aplica otra condición para seguir dividiendo los datos.  
* **Rama:** Es el camino que se sigue dependiendo del resultado de una condición. Por ejemplo, una rama puede representar "sí" y otra "no".  
* **Hoja:** Es el punto final del árbol. Aquí se encuentra el resultado o clasificación final.

En pocas palabras, el árbol comienza en el nodo raíz, después va pasando por diferentes condiciones y ramas hasta llegar a una hoja, donde se obtiene la respuesta.

### **Pregunta 3**

**¿Qué es una red neuronal multicapa y qué función cumplen sus capas?**

Una red neuronal multicapa es un modelo de inteligencia artificial que está formado por varias capas de neuronas conectadas entre sí. Estas conexiones permiten que la red pueda aprender patrones a partir de los datos.

Las principales capas son:

* **Capa de entrada:** Es la que recibe los datos que queremos analizar. Por ejemplo, podría recibir la edad, calificaciones y asistencia de un estudiante.  
* **Capas ocultas:** Son las que procesan la información y ayudan a encontrar patrones más complicados dentro de los datos. Una red puede tener una o varias capas ocultas.  
* **Capa de salida:** Es donde se obtiene el resultado final. Por ejemplo, podría indicar si un estudiante tiene riesgo de reprobar o no.

### **Pregunta 4**

**¿Qué representan los pesos y los sesgos dentro de una red neuronal? Explique también por qué sus valores cambian durante el entrenamiento.**

Los **pesos** indican qué tanta importancia tiene cada dato de entrada para una neurona. Dependiendo del valor del peso, una entrada puede tener más o menos influencia en el resultado.

El **sesgo** es otro valor que ayuda a ajustar el resultado de la neurona. Sirve para que la red pueda adaptarse mejor a los datos que está analizando.

Estos valores cambian durante el entrenamiento porque al principio la red no sabe cómo relacionar correctamente los datos con las respuestas. La red hace una predicción, compara el resultado con la respuesta correcta y calcula el error. Después modifica los pesos y los sesgos para intentar cometer menos errores en las siguientes predicciones.

### **Pregunta 5**

**¿Cuál es la principal diferencia entre la forma en que aprende un árbol de decisión y la forma en que aprende una red neuronal multicapa? Explique qué elementos aprende cada modelo.**

La principal diferencia es que un árbol de decisión aprende **reglas y condiciones**, mientras que una red neuronal aprende principalmente **pesos y sesgos**.

El árbol busca las mejores condiciones para dividir los datos. Por ejemplo, puede aprender una regla como: "si la calificación es menor a 70, entonces el estudiante tiene mayor riesgo de reprobar".

En cambio, la red neuronal va ajustando sus pesos y sesgos durante el entrenamiento. De esta manera aprende a reconocer patrones entre los diferentes datos de entrada.

Por eso, un árbol de decisión normalmente es más fácil de entender, mientras que una red neuronal puede encontrar relaciones más complicadas, aunque sea más difícil saber exactamente cómo llegó a su resultado.

# **Parte II. Análisis y aplicación**

### **Pregunta 6**

**Una institución bancaria desea desarrollar un sistema que detecte posibles compras fraudulentas. Analice las ventajas y desventajas de utilizar un árbol de decisión y una red neuronal multicapa. ¿Cuál utilizaría y por qué?**

Un **árbol de decisión** tiene como ventaja que es fácil de entender. Por ejemplo, podría mostrar que una compra es sospechosa porque el monto es muy alto, se realizó desde otra ciudad y además hubo muchas compras en poco tiempo.

También es relativamente sencillo revisar sus decisiones. Sin embargo, si el fraude depende de muchos factores diferentes, el árbol podría tener problemas para encontrar todos esos patrones y también puede caer en sobreajuste si se hace demasiado grande.

Una **red neuronal multicapa** puede aprender relaciones más complicadas entre los datos. Esto puede ser útil para detectar fraudes, ya que muchas veces no existe una sola característica que indique que una compra es sospechosa.

La desventaja es que puede ser más difícil entender exactamente por qué la red tomó una decisión.

En este caso, yo utilizaría una **red neuronal multicapa**, principalmente si se tiene una gran cantidad de datos de compras anteriores. Creo que podría detectar patrones más complicados. Sin embargo, también buscaría alguna forma de revisar y explicar sus resultados.

### **Pregunta 7**

**Una escuela quiere detectar estudiantes que presentan riesgo de reprobar una materia. Si ambos modelos obtienen prácticamente la misma precisión, ¿qué otros factores tomaría en cuenta?**

Si los dos modelos tienen prácticamente la misma precisión, tomaría en cuenta otros aspectos como qué tan fácil es entenderlos, cuánto tardan en entrenarse y qué tan sencillo sería mantenerlos.

En este caso probablemente elegiría el **árbol de decisión**, porque los profesores podrían entender fácilmente por qué el sistema considera que un estudiante está en riesgo.

Por ejemplo, podrían ver que el resultado se debe principalmente a que el estudiante tiene muchas faltas y no ha entregado varias tareas. Esto también permitiría que el profesor tome acciones para ayudar al estudiante.

Por eso, si los dos modelos tienen resultados similares, considero que sería mejor utilizar el que sea más fácil de interpretar.

### **Pregunta 8**

**Un hospital desarrolla un sistema para determinar qué pacientes necesitan atención prioritaria. Una red neuronal obtiene mejores resultados que un árbol de decisión, pero resulta más difícil explicar cómo obtuvo su respuesta. ¿Considera que la mayor precisión es suficiente para elegir la red neuronal?**

No, considero que la precisión por sí sola no sería suficiente.

En un hospital las decisiones son muy importantes porque pueden afectar directamente a los pacientes. Por eso también sería necesario saber por qué el sistema tomó determinada decisión.

Aunque la red neuronal tenga mejores resultados, podría ser un problema si los médicos no pueden entender por qué clasificó a un paciente como prioritario.

También revisaría qué tipo de errores comete cada modelo, ya que no todos los errores tienen la misma importancia. Por ejemplo, podría ser mucho más grave no detectar a un paciente que necesita atención urgente que marcar por error a alguien como prioritario.

Por eso compararía varios aspectos antes de elegir el modelo y no solamente su porcentaje de precisión.

### **Pregunta 9**

**Una empresa de reparto obtiene predicciones diferentes entre un árbol de decisión y una red neuronal. ¿Cómo determinaría cuál de los dos modelos está realizando una mejor predicción?**

Primero tendría que conocer cuál fue el resultado real. Por ejemplo, si un modelo dice que un pedido llegará tarde, tendría que esperar para comprobar si realmente llegó tarde.

Después compararía los resultados de los dos modelos utilizando muchos pedidos, no solamente uno. Así podría saber cuál de los dos acierta más veces.

También utilizaría diferentes métricas para comparar su desempeño, como precisión, recall y una matriz de confusión.

Además, sería importante probarlos con datos que no hayan utilizado durante su entrenamiento. De esta forma se puede comprobar qué tan bien funcionan con casos nuevos.

Por lo tanto, elegiría el modelo que tenga un mejor desempeño general y no simplemente el que haya acertado en un solo pedido.

### **Pregunta 10**

**Una empresa desarrolla dos sistemas para decidir si una persona puede recibir un crédito. El primero utiliza un árbol de decisión y explica claramente por qué una solicitud fue rechazada. El segundo utiliza una red neuronal multicapa y obtiene mejores resultados de predicción, pero es más difícil explicar sus decisiones. Si usted fuera responsable del proyecto, ¿qué consideraría?**

Yo no elegiría solamente el modelo que tenga mayor precisión. También consideraría qué tan importante es poder explicar las decisiones que toma el sistema.

En este caso, el **árbol de decisión** tiene una ventaja importante porque permite saber por qué se rechazó una solicitud. Por ejemplo, podría mostrar que una persona fue rechazada debido a sus ingresos, historial crediticio o nivel de endeudamiento.

La red neuronal podría ser más precisa, pero si no podemos explicar fácilmente sus decisiones, podría ser complicado justificar por qué una persona recibió o no un crédito.

Primero compararía qué tanta diferencia existe realmente entre los dos modelos. Si la red neuronal solamente es un poco más precisa, probablemente elegiría el árbol de decisión por ser más fácil de entender.

Si la diferencia fuera bastante grande, consideraría utilizar la red neuronal, pero buscaría alguna forma de interpretar sus resultados y mantener una revisión por parte de una persona.

En este caso creo que lo más importante sería encontrar un equilibrio entre **precisión, facilidad de entender el modelo y confianza en las decisiones que toma**.

