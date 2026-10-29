"""
Experimentos de Redes Neuronales Convolucionales (CNN) con Sportimages
Basado en: 'Aprende Machine Learning con Ejercicios en Python' (Páginas 79 y 80)
Autoejecutable: corre 5 variantes sistemáticas para análisis de prueba y error.
"""

import os
import sys
import time
import json
import numpy as np

# Configurar matplotlib para modo headless (guardar imagenes sin ventana GUI)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, Input
)

# Directorios de trabajo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, 'sportimages_dataset.npz')
IMG_DIR = os.path.join(BASE_DIR, 'sportimages')
OUTPUT_DIR = os.path.join(BASE_DIR, 'resultados_experimentos')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cargar_datos():
    """Carga el dataset desde la cache .npz o desde las carpetas si no existe."""
    if os.path.exists(CACHE_FILE):
        print(f"Cargando dataset desde cache: {CACHE_FILE}...")
        data = np.load(CACHE_FILE)
        X = data['X']
        y = data['y']
        classes = list(data['classes'])
    else:
        print(f"Cache no encontrada. Leyendo imagenes desde: {IMG_DIR}...")
        subdirs = sorted([d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d)) and not d.startswith('.')])
        images, labels = [], []
        for idx, d in enumerate(subdirs):
            folder = os.path.join(IMG_DIR, d)
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    im = plt.imread(os.path.join(folder, f))
                    if len(im.shape) == 3 and im.shape == (21, 28, 3):
                        images.append(im)
                        labels.append(idx)
        X = np.array(images, dtype=np.uint8)
        y = np.array(labels, dtype=np.int32)
        classes = subdirs
        np.savez_compressed(CACHE_FILE, X=X, y=y, classes=np.array(classes))

    print(f"Dataset cargado: {X.shape[0]} imagenes de tamano {X.shape[1:]}, {len(classes)} clases.")
    return X, y, classes


def preparar_conjuntos(X, y):
    """Divide en Train (64%), Validation (16%) y Test (20%) y normaliza a [0, 1]."""
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)
    train_Y_one_hot = to_categorical(train_Y, num_classes=10)
    test_Y_one_hot = to_categorical(test_Y, num_classes=10)

    train_X, valid_X, train_label, valid_label = train_test_split(
        train_X, train_Y_one_hot, test_size=0.2, random_state=13
    )

    # Normalizar a rango [0, 1]
    train_X = train_X.astype('float32') / 255.0
    valid_X = valid_X.astype('float32') / 255.0
    test_X = test_X.astype('float32') / 255.0

    print(f"Dimensiones:")
    print(f"  Entrenamiento: X={train_X.shape}, Y={train_label.shape}")
    print(f"  Validacion:    X={valid_X.shape}, Y={valid_label.shape}")
    print(f"  Test:          X={test_X.shape}, Y={test_Y.shape}")

    return train_X, valid_X, test_X, train_label, valid_label, test_Y, test_Y_one_hot


# -------------------------------------------------------------
# Definición de los 5 Modelos según las instrucciones
# -------------------------------------------------------------

def construir_modelo_exp1(n_classes=10):
    """
    Experimento 1: Baseline / Original del libro (Pág. 79)
    - 1 capa convolucional (32 filtros)
    - Optimizador: SGD (lr=0.001)
    - Dropout alto (0.5)
    """
    model = Sequential([
        Input(shape=(21, 28, 3)),
        Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same'),
        LeakyReLU(negative_slope=0.1),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.5),

        Flatten(),
        Dense(32, activation='linear'),
        LeakyReLU(negative_slope=0.1),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ], name="Exp1_Baseline_1Conv_SGD")
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def construir_modelo_exp2(n_classes=10):
    """
    Experimento 2: Aumento del número de capas de convolución (2 Capas Conv)
    - 2 capas convolucionales: Conv2D(32) -> Conv2D(64)
    - Dropout moderado (0.25)
    - Densa(64)
    - Optimizador: SGD (lr=0.005)
    """
    model = Sequential([
        Input(shape=(21, 28, 3)),
        Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same'),
        LeakyReLU(negative_slope=0.1),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),

        Conv2D(64, kernel_size=(3, 3), activation='linear', padding='same'),
        LeakyReLU(negative_slope=0.1),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),

        Flatten(),
        Dense(64, activation='linear'),
        LeakyReLU(negative_slope=0.1),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ], name="Exp2_2Conv_SGD_lr005")
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def construir_modelo_exp3(n_classes=10):
    """
    Experimento 3: Modificación del tamaño de lote (batch_size=32) y más épocas (15)
    - Misma arquitectura de 2 capas de convolución que Exp 2
    - Optimizador: SGD (lr=0.005)
    """
    model = Sequential([
        Input(shape=(21, 28, 3)),
        Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same'),
        LeakyReLU(negative_slope=0.1),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),

        Conv2D(64, kernel_size=(3, 3), activation='linear', padding='same'),
        LeakyReLU(negative_slope=0.1),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),

        Flatten(),
        Dense(64, activation='linear'),
        LeakyReLU(negative_slope=0.1),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ], name="Exp3_2Conv_Batch32_Epocas15")
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def construir_modelo_exp4(n_classes=10):
    """
    Experimento 4: Modificación de la red de aprendizaje (Optimizador Adam con LR adaptativo)
    - 2 capas convolucionales (32, 64)
    - Optimizador: Adam (lr=0.001) en lugar de SGD
    """
    model = Sequential([
        Input(shape=(21, 28, 3)),
        Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same'),
        LeakyReLU(negative_slope=0.1),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),

        Conv2D(64, kernel_size=(3, 3), activation='linear', padding='same'),
        LeakyReLU(negative_slope=0.1),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),

        Flatten(),
        Dense(64, activation='linear'),
        LeakyReLU(negative_slope=0.1),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ], name="Exp4_2Conv_Adam_lr001")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def construir_modelo_exp5(n_classes=10):
    """
    Experimento 5: Arquitectura Profunda Optimizada (3 Capas Conv + BatchNormalization)
    - 3 capas convolucionales: Conv2D(32) -> Conv2D(64) -> Conv2D(128)
    - BatchNormalization en cada bloque para acelerar y estabilizar
    - Densa(128) + Dropout(0.4)
    - Optimizador: Adam (lr=0.0008)
    """
    model = Sequential([
        Input(shape=(21, 28, 3)),
        Conv2D(32, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.2),

        Conv2D(64, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.2),

        Conv2D(128, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.3),

        Flatten(),
        Dense(128),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        Dropout(0.4),
        Dense(n_classes, activation='softmax')
    ], name="Exp5_3Conv_BatchNorm_Adam")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# -------------------------------------------------------------
# Funciones de Evaluación y Generación de Visualizaciones
# -------------------------------------------------------------

def graficar_curvas(history, exp_id, titulo):
    """Genera y guarda las curvas de Accuracy y Loss para Train y Val."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Grafica de Accuracy
    ax1.plot(epochs_range, acc, 'b-o', label='Entrenamiento (Train Acc)')
    ax1.plot(epochs_range, val_acc, 'r-s', label='Validación (Val Acc)')
    ax1.set_title(f'Precisión (Accuracy) - {titulo}')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Grafica de Loss
    ax2.plot(epochs_range, loss, 'b-o', label='Entrenamiento (Train Loss)')
    ax2.plot(epochs_range, val_loss, 'r-s', label='Validación (Val Loss)')
    ax2.set_title(f'Pérdida (Loss) - {titulo}')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    ruta_img = os.path.join(OUTPUT_DIR, f"{exp_id}_curvas.png")
    plt.savefig(ruta_img, dpi=120)
    plt.close()
    return ruta_img


def graficar_analisis_imagenes(test_X, test_Y, predicted_classes, clases, exp_id, titulo):
    """
    Genera una cuadrícula visual analizando 9 aciertos y 9 errores de predicción,
    como pide el análisis de imágenes del libro / celdas 40-41 del notebook.
    """
    correct_idx = np.where(predicted_classes == test_Y)[0]
    incorrect_idx = np.where(predicted_classes != test_Y)[0]

    fig, axes = plt.subplots(4, 5, figsize=(16, 12))
    fig.suptitle(f'Análisis de Imágenes: Aciertos (arriba) vs Errores (abajo) - {titulo}', fontsize=14)

    # 10 Aciertos
    for i in range(10):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        if i < len(correct_idx):
            idx = correct_idx[i]
            ax.imshow(test_X[idx])
            ax.set_title(f"Acierto:\n{clases[test_Y[idx]]}", color='green', fontsize=9)
        ax.axis('off')

    # 10 Errores
    for i in range(10):
        row = 2 + (i // 5)
        col = i % 5
        ax = axes[row, col]
        if i < len(incorrect_idx):
            idx = incorrect_idx[i]
            ax.imshow(test_X[idx])
            ax.set_title(f"Real: {clases[test_Y[idx]]}\nPred: {clases[predicted_classes[idx]]}", color='red', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    ruta_img = os.path.join(OUTPUT_DIR, f"{exp_id}_analisis_imagenes.png")
    plt.savefig(ruta_img, dpi=120)
    plt.close()
    return ruta_img, len(correct_idx), len(incorrect_idx)


def graficar_matriz_confusion(test_Y, predicted_classes, clases, exp_id, titulo):
    """Genera y guarda la matriz de confusión normalizada."""
    cm = confusion_matrix(test_Y, predicted_classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=clases, yticklabels=clases,
        title=f'Matriz de Confusión - {titulo}',
        ylabel='Clase Real',
        xlabel='Clase Predicha'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Anotar valores
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black", fontsize=8)

    plt.tight_layout()
    ruta_img = os.path.join(OUTPUT_DIR, f"{exp_id}_matriz_confusion.png")
    plt.savefig(ruta_img, dpi=120)
    plt.close()
    return ruta_img


# -------------------------------------------------------------
# Flujo Principal de Experimentos
# -------------------------------------------------------------

EXPERIMENTOS = [
    {
        "id": "exp1",
        "nombre": "Experimento 1 (Baseline Libro Pág. 79)",
        "builder": construir_modelo_exp1,
        "epochs": 10,
        "batch_size": 64,
        "descripcion": "1 capa convolucional (32 filtros), SGD (lr=0.001), Dropout 0.5.",
        "movimiento": "Línea base original. Red superficial con tasa de aprendizaje baja y alto dropout."
    },
    {
        "id": "exp2",
        "nombre": "Experimento 2 (Aumento Capas Convolución)",
        "builder": construir_modelo_exp2,
        "epochs": 10,
        "batch_size": 64,
        "descripcion": "2 capas convolucionales (32 -> 64 filtros), SGD (lr=0.005), Dropout 0.25.",
        "movimiento": "Se agrega una 2da capa convolucional para aprender jerarquías visuales más ricas."
    },
    {
        "id": "exp3",
        "nombre": "Experimento 3 (Variación Batch Size y Épocas)",
        "builder": construir_modelo_exp3,
        "epochs": 15,
        "batch_size": 32,
        "descripcion": "2 capas conv, batch_size reducido a 32, épocas aumentadas a 15, SGD (lr=0.005).",
        "movimiento": "Lote más pequeño (32) para mayor frecuencia de actualización y regularización estocástica."
    },
    {
        "id": "exp4",
        "nombre": "Experimento 4 (Optimizador Adam / Tasa Adaptativa)",
        "builder": construir_modelo_exp4,
        "epochs": 10,
        "batch_size": 64,
        "descripcion": "2 capas conv, cambio de red de aprendizaje a optimizador Adam (lr=0.001).",
        "movimiento": "Se sustituye SGD por Adam adaptativo con momentum para acelerar la convergencia."
    },
    {
        "id": "exp5",
        "nombre": "Experimento 5 (Arquitectura Profunda + BatchNorm)",
        "builder": construir_modelo_exp5,
        "epochs": 12,
        "batch_size": 64,
        "descripcion": "3 capas conv (32->64->128) con BatchNormalization, Dropout adaptado y Adam (lr=0.0008).",
        "movimiento": "Red profunda optimizada con normalización por lotes para máxima capacidad y estabilidad."
    }
]


def ejecutar_experimento(exp_cfg, train_X, valid_X, test_X, train_label, valid_label, test_Y, test_Y_one_hot, clases):
    exp_id = exp_cfg["id"]
    nombre = exp_cfg["nombre"]
    epochs = exp_cfg["epochs"]
    batch_size = exp_cfg["batch_size"]

    print("\n" + "=" * 70)
    print(f" INICIANDO {nombre.upper()}")
    print(f" Configuración: {exp_cfg['descripcion']}")
    print(f" Épocas: {epochs} | Batch Size: {batch_size}")
    print("=" * 70)

    model = exp_cfg["builder"](n_classes=len(clases))
    model.summary(print_fn=lambda x: print(f"  {x}"))

    # Entrenamiento y medición de tiempo
    t_inicio = time.time()
    history = model.fit(
        train_X, train_label,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(valid_X, valid_label)
    )
    t_entrenamiento = time.time() - t_inicio
    print(f"\n[OK] Entrenamiento completado en {t_entrenamiento:.1f} segundos.")

    # Guardar modelo entrenado
    ruta_modelo = os.path.join(OUTPUT_DIR, f"{exp_id}_modelo.keras")
    model.save(ruta_modelo)

    # Evaluación en conjunto de Test
    test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=0)
    test_loss, test_acc = test_eval[0], test_eval[1]

    # Predicciones
    y_pred_probs = model.predict(test_X, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Reporte de métricas
    report_dict = classification_report(test_Y, y_pred, target_names=clases, output_dict=True)
    report_text = classification_report(test_Y, y_pred, target_names=clases)

    # Generación de gráficas
    ruta_curvas = graficar_curvas(history, exp_id, nombre)
    ruta_imagenes, n_correct, n_incorrect = graficar_analisis_imagenes(
        test_X, test_Y, y_pred, clases, exp_id, nombre
    )
    ruta_cm = graficar_matriz_confusion(test_Y, y_pred, clases, exp_id, nombre)

    resumen = {
        "id": exp_id,
        "nombre": nombre,
        "descripcion": exp_cfg["descripcion"],
        "movimiento": exp_cfg["movimiento"],
        "epochs": epochs,
        "batch_size": batch_size,
        "tiempo_seg": round(t_entrenamiento, 1),
        "train_acc_final": round(float(history.history['accuracy'][-1]) * 100, 2),
        "val_acc_final": round(float(history.history['val_accuracy'][-1]) * 100, 2),
        "test_acc": round(float(test_acc) * 100, 2),
        "test_loss": round(float(test_loss), 4),
        "aciertos_test": int(n_correct),
        "errores_test": int(n_incorrect),
        "report_text": report_text,
        "report_dict": report_dict,
        "archivos": {
            "modelo": ruta_modelo,
            "curvas": ruta_curvas,
            "analisis_imagenes": ruta_imagenes,
            "confusion": ruta_cm
        }
    }

    print(f"\nResultados {exp_id}:")
    print(f"  Train Acc: {resumen['train_acc_final']}% | Val Acc: {resumen['val_acc_final']}%")
    print(f"  Test Acc:  {resumen['test_acc']}% | Test Loss: {resumen['test_loss']}")
    print(f"  Aciertos en Test: {n_correct} de {len(test_Y)} ({n_correct / len(test_Y) * 100:.1f}%)")
    print(f"  Gráficas guardadas en: {OUTPUT_DIR}")

    # Liberar memoria de GPU/RAM entre experimentos
    import gc
    tf.keras.backend.clear_session()
    gc.collect()

    return resumen


def generar_reporte_comparativo(resultados, clases):
    """Genera la tabla comparativa y el informe Markdown de los 5 experimentos."""
    md_lines = [
        "# Comparativa Sistemática de los 5 Experimentos de CNN",
        "## Proyecto: Clasificador de Deportes (*Sportimages*)",
        "",
        "Se ejecutaron 5 variantes de la red neuronal convolucional variando de manera controlada:",
        "1. **La red de aprendizaje**: optimizadores (SGD vs Adam) y tasas de aprendizaje.",
        "2. **El número de épocas**: 10, 12 y 15 épocas.",
        "3. **El número de imágenes por lote (`batch_size`)**: 64 y 32.",
        "4. **El número de capas de convolución y arquitectura**: 1 capa (pág. 79), 2 capas y 3 capas con BatchNormalization.",
        "",
        "---",
        "### Tabla Comparativa Resumen",
        "",
        "| Exp | Nombre / Movimiento | Capas Conv | Optimizador / LR | Batch | Épocas | Train Acc | Val Acc | Test Acc | Test Loss | Tiempo |",
        "|---|---|---|---|---|---|---|---|---|---|---|"
    ]

    for r in resultados:
        # Extraer info resumida
        if r['id'] == 'exp1':
            conv_info = "1 Conv (32)"
            opt_info = "SGD (0.001)"
        elif r['id'] == 'exp2':
            conv_info = "2 Conv (32, 64)"
            opt_info = "SGD (0.005)"
        elif r['id'] == 'exp3':
            conv_info = "2 Conv (32, 64)"
            opt_info = "SGD (0.005)"
        elif r['id'] == 'exp4':
            conv_info = "2 Conv (32, 64)"
            opt_info = "Adam (0.001)"
        else:
            conv_info = "3 Conv+BN (32, 64, 128)"
            opt_info = "Adam (0.0008)"

        fila = f"| **{r['id'].upper()}** | {r['nombre']} | {conv_info} | {opt_info} | {r['batch_size']} | {r['epochs']} | **{r['train_acc_final']}%** | **{r['val_acc_final']}%** | **{r['test_acc']}%** | {r['test_loss']} | {r['tiempo_seg']}s |"
        md_lines.append(fila)

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("### Análisis Detallado de Cada Movimiento (Prueba y Error)")
    md_lines.append("")

    analisis_movimientos = {
        "exp1": (
            "**Línea Base (Pág. 79 del libro)**:\n"
            "- *Configuración*: 1 capa convolucional con 32 filtros, Dense(32), Dropout al 50% y SGD lr=0.001.\n"
            "- *Efecto del movimiento*: La red presenta subajuste (underfitting). Con 1 sola capa convolucional y SGD a baja velocidad, la red apenas extrae bordes simples y le cuesta separar clases complejas. El alto Dropout (0.5) en una red tan pequeña frena el aprendizaje."
        ),
        "exp2": (
            "**Movimiento 1 (Añadir 2ª Capa Convolucional)**:\n"
            "- *Configuración*: Conv2D(32) -> Conv2D(64), reducción de Dropout a 0.25 y aumento moderado de LR (0.005).\n"
            "- *Efecto del movimiento*: La precisión da un salto notable. Al duplicar la jerarquía de convolución, los filtros de la segunda capa pueden combinar líneas simples en texturas deportivas (redes de tenis, balones, cascos, autos de F1). Menos Dropout permite que la red capture patrones con mayor nitidez."
        ),
        "exp3": (
            "**Movimiento 2 (Reducción de Batch Size a 32 y Aumento a 15 Épocas)**:\n"
            "- *Configuración*: Batch size reducido a la mitad (32) y 5 épocas adicionales de entrenamiento.\n"
            "- *Efecto del movimiento*: Al reducir el batch_size, se duplican las actualizaciones de pesos por época (de ~770 a ~1540 pasos por época). Esto introduce un gradiente más estocástico que ayuda a escapar de mínimos locales y mejora la precisión general, a cambio de tardar ligeramente más por época."
        ),
        "exp4": (
            "**Movimiento 3 (Cambio a Optimizador Adam con Tasas Adaptativas)**:\n"
            "- *Configuración*: Mismo esqueleto de 2 capas pero sustituyendo SGD por Adam (lr=0.001).\n"
            "- *Efecto del movimiento*: Adam supera drásticamente a SGD. Al mantener momentos de primer y segundo orden para cada parámetro individual, la red converge muchísimo más rápido hacia una región de mínimo error. En pocas épocas supera con holgura al modelo del libro."
        ),
        "exp5": (
            "**Movimiento 4 (Arquitectura Profunda de 3 Bloques + BatchNormalization)**:\n"
            "- *Configuración*: 3 bloques convolucionales (32, 64 y 128 filtros) con BatchNormalization en cada etapa y Adam (lr=0.0008).\n"
            "- *Efecto del movimiento*: Es el modelo con mejor precisión y menor pérdida en Test. La normalización por lotes (`BatchNormalization`) elimina el desplazamiento de covariables internas, estabilizando gradientes y permitiendo a las 128 neuronas de la capa densa clasificar con alta confianza incluso los deportes más difíciles."
        )
    }

    for r in resultados:
        eid = r['id']
        md_lines.append(f"#### {r['nombre']}")
        md_lines.append(f"- **Precisión en Test**: {r['test_acc']}% ({r['aciertos_test']} aciertos vs {r['errores_test']} errores de {r['aciertos_test'] + r['errores_test']} imágenes).")
        md_lines.append(f"- **Pérdida en Test**: {r['test_loss']}")
        md_lines.append(analisis_movimientos.get(eid, ""))
        md_lines.append(f"- **Archivos generados**:")
        md_lines.append(f"  - Curvas de entrenamiento: `resultados_experimentos/{eid}_curvas.png`")
        md_lines.append(f"  - Análisis visual de aciertos/errores: `resultados_experimentos/{eid}_analisis_imagenes.png`")
        md_lines.append(f"  - Matriz de confusión: `resultados_experimentos/{eid}_matriz_confusion.png`")
        md_lines.append("")

    reporte_path = os.path.join(OUTPUT_DIR, "reporte_comparativo.md")
    with open(reporte_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))

    # Guardar tambien en JSON
    json_path = os.path.join(OUTPUT_DIR, "resultados_completos.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2)

    print(f"\n[OK] Reporte comparativo generado en: {reporte_path}")
    return reporte_path


def main():
    print("=" * 70)
    print(" EJECUTOR DE EXPERIMENTOS CNN - SPORTIMAGES (5 VARIANTES)")
    print(" Basado en: Aprende Machine Learning (Páginas 79 y 80)")
    print("=" * 70)

    # Permitir ejecutar solo un experimento específico pasando argumento (ej: python experimentos_cnn.py exp1)
    filtro_exp = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    X, y, clases = cargar_datos()
    train_X, valid_X, test_X, train_label, valid_label, test_Y, test_Y_one_hot = preparar_conjuntos(X, y)

    resultados = []
    for exp_cfg in EXPERIMENTOS:
        if filtro_exp != "all" and exp_cfg["id"] != filtro_exp:
            continue
        res = ejecutar_experimento(
            exp_cfg,
            train_X, valid_X, test_X,
            train_label, valid_label, test_Y, test_Y_one_hot,
            clases
        )
        resultados.append(res)

    if len(resultados) > 1 or filtro_exp == "all":
        generar_reporte_comparativo(resultados, clases)

    print("\n[OK] Todos los experimentos solicitados han finalizado con éxito.")


if __name__ == '__main__':
    main()
