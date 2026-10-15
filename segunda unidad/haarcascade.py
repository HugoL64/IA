
import cv2 as cv
from pathlib import Path

# Carpeta donde se guardarán las imágenes
ruta = Path(__file__).resolve().parent / "imgs"
ruta.mkdir(exist_ok=True)

# Cargar el clasificador de rostros
rostro = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_alt.xml"
)

if rostro.empty():
    raise RuntimeError("No se pudo cargar el clasificador")

# Activar cámara web
cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo obtener imagen de la cámara")
        break

    # Convertir imagen a escala de grises
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detectar rostros
    rostros = rostro.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in rostros:

        # Recortar rostro sin rectángulo verde
        img = frame[y:y+h, x:x+w].copy()

        # Guardar la imagen
        count += 1
        name = ruta / f"cara{count}.jpg"
        cv.imwrite(str(name), img)

        # Dibujar rectángulo verde con grosor mínimo (1 px)
        cv.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            1
        )

        # Mostrar el rostro detectado
        cv.imshow("cara", img)

    # Mostrar cámara con detección
    cv.imshow("rostros", frame)

    # Presionar ESC para salir
    k = cv.waitKey(1)

    if k == 27:
        break

cap.release()
cv.destroyAllWindows()

print(f"Total de rostros guardados: {count}")
