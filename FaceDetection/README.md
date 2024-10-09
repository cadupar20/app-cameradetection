# flask-realtime-face-detection-opencv-python
Detección de rostros en tiempo real con Python y OpenCV. La cámara web debe estar habilitada para que funcione.

#Cómo usarlo
1. Instale todas las dependencias que tiene
: versión 2.x de OpenCV
: Python 2.7
: Flask

2. Ahora, entrene el algoritmo ejecutando el archivo create_data.py
3. Una vez que se hayan entrenado los datos, puede ejecutar face_recognise.py para que se ejecute
4. Para usarlo en una interfaz basada en la web, ejecute python app.py y abra su host local

- La función cv2.VideoCapture(0) inicia la cámara predeterminada (a menudo, la cámara web principal) para la captura de imágenes. El objeto de captura de video está representado por la variable cap.
- Dentro del bucle, la función lee un solo cuadro de la cámara usando cap.read(). El valor de retorno ret indica si el marco se leyó correctamente y los datos del marco se almacenan en la variable frame.
cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) se utiliza para convertir el marco a escala de grises. Las imágenes en escala de grises son más fáciles de procesar y se utilizan comúnmente en tareas de detección de rostros.
- Para detectar rostros en el marco en escala de grises, el programa utiliza el clasificador face_cascade construido previamente. face_cascade.detectMultiScale() detecta rostros en imágenes a varias escalas. Los rostros reconocidos se devuelven como una lista de rectángulos (x, y, ancho, alto), que se guarda en la variable faces.
- Utilizando cv2.rectangle(), la función dibuja un rectángulo verde alrededor de cada rostro detectado en el marco de color original. Las imágenes de los rostros se recortan del marco en escala de grises y se guardan en el directorio “Faces” con nombres de archivo en el formato “userX.jpg”, donde X es el valor de la variable count. Después de guardar cada imagen, se incrementa el recuento.
- cv2.imshow() se utiliza para mostrar el marco con la detección de rostros y los rectángulos verdes. En la pantalla, el usuario puede ver el proceso de detección de rostros en tiempo real.
- El bucle se puede terminar de dos maneras:
- Si el usuario presiona la tecla 'q', el bucle se interrumpe ya que cv2.waitKey(1) & 0xFF == ord('q') se evalúa como True.

- Detector de rostros: utilizamos el clasificador en cascada Haar para detectar rostros que vamos a capturar en el siguiente paso. El clasificador en cascada Haar es un modelo preentrenado que puede detectar rápidamente objetos, incluidos rostros, en una imagen. La clase CascadeClassifier de OpenCV se utiliza para construir la variable face_cascade. Para reconocer rostros en imágenes, emplea el clasificador en cascada Haar. El archivo XML 'haarcascade_frontalface_default.xml' contiene el modelo preentrenado para la detección de rostros frontales. Este archivo suele estar incluido con OpenCV

#nota:
Puedes omitir la parte de Heroku, lo único que necesitas son Flask y los scripts de Python.

# Funcion main.py (/FaceDetection/main.py)
La segunda funcion de este proyecto es #, donde integra la captura de rostros, almacenandolas en un folder llamado dataset, el entrenamiento del modelo y la detección de rostros en tiempo real. El modelo de entrenamiento utiliza el modelo Haar Cascade, el cual fue descargado del sitio Github https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

## Captura imagenes y las almacena en store folder dataset
1. capture_images()
   Captura de rostros y almacenamiento en dataset folder.

## Entrenamiento del modelo de reconocimiento facial
2. train_model()
   Entrenamiento del modelo de reconocimiento facial utilizando el modelo Haar Cascade.

## Reconocimiento de rostros en tiempo real del modelo entrenado
3. recognize_faces()
   Reconocimiento de rostros en tiempo real del modelo de reconocimiento facial.

