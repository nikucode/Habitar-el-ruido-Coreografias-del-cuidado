CONEXIÓN / PRESENCIA / EXCESO
==============================
Obra audiovisual reactiva con visión por computador/ reconocimiento facial y glitch digital



DEPENDENCIAS E INSTALACIÓN
---------------------------
El proyecto está escrito en Python y requiere las siguientes bibliotecas:

* Python 3.7+
* OpenCV (opencv-python): Para toda la captura, procesamiento de imagen y video.
* NumPy: Para operaciones numéricas eficientes con matrices de imagen.
* (Opcional) Haarcascade: El archivo haarcascade_frontalface_default.xml ya
  está incluido para la detección de rostros.

Pasos para la instalación:

1. Clona el repositorio (o descarga los archivos en una carpeta).
   git clone <URL-de-tu-repositorio>
   cd <nombre-de-la-carpeta>

2. (Recomendado) Crea un entorno virtual para aislar las dependencias.
   python -m venv venv
   source venv/bin/activate  # En Linux/macOS
   venv\Scripts\activate      # En Windows

3. Instala las dependencias.
   pip install opencv-python numpy

4. Prepara el video de fondo.
   * Crea una carpeta llamada "videos" en el directorio raíz del proyecto.
   * Coloca dentro un archivo de video. Por defecto, el código busca un archivo
     llamado "v1.mp4" (datamosh_v2.py) o "Download (3).mp4" (datamosh.py).
     Puedes cambiar el nombre del archivo en la variable VIDEO_PATH dentro de
     cada script.


CÓMO EJECUTAR
-------------
Una vez instaladas las dependencias y colocado el video, simplemente ejecuta
el script deseado desde tu terminal:

python datamosh.py
o
python datamosh_v2.py

* La ventana "conexion" se abrirá mostrando la mezcla de tu cámara y el video.
* Presiona la tecla ESC para salir del programa.


ARCHIVOS DEL PROYECTO
---------------------
* datamosh.py: Versión inicial o simplificada. El glitch se activa en
  intervalos de tiempo fijos (11s normal, 4s glitch), con una transición suave.
  La presencia humana afecta principalmente a la opacidad de la mezcla y la
  velocidad del video.

* datamosh_v2.py: Versión más compleja y sensible al contexto. Introduce un
  sistema de estados emocionales (reposo, conexion, intensidad, saturacion) que
  modulan la intensidad del glitch, la distorsión de la imagen de la cámara
  (blur, ruido, jitter) y otros parámetros. La lógica es más rica y busca una
  experiencia más matizada.

* haarcascade_frontalface_default.xml: Clasificador pre-entrenado de OpenCV
  para la detección de rostros frontales. Esencial para medir la "presencia".

* videos/: Carpeta donde se debe alojar el archivo de video que actuará como
  "flujo de información".



PARÁMETROS CLAVE PARA LA EXPERIMENTACIÓN
----------------------------------------
Dentro de cada script hay variables que puedes ajustar para modificar el
comportamiento de la obra:

* VIDEO_PATH: Ruta al archivo de video.
* BASE_BLOCK_SIZE: Tamaño de los bloques de píxeles que se desplazan durante el
  datamosh. Ajusta la "granularidad" del glitch.
* STICKINESS: Controla la inercia del desplazamiento de los bloques. Valores
  altos crean un glitch más persistente y "pegajoso".
* FREEZE_PROB / UNFREEZE_PROB: Probabilidades de que un bloque se congele o se
  libere, creando texturas estáticas.
* NORMAL_DURATION / DATAMOSH_DURATION (en datamosh.py): Controlan el ritmo
  temporal del ciclo de glitch.
* Umbrales de movimiento y conteo de rostros (en datamosh_v2.py): Definen los
  límites para cambiar entre los estados de reposo, conexion, etc.

Siéntete libre de modificar estos valores para explorar nuevas texturas y
ritmos visuales.


Autoría colectiva / proceso artístico
