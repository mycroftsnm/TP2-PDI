# README
## Trabajo Práctico N°2, Procesamiento de Imágenes 1, de TUIA, FCEIA, UNR.

Este repositorio contiene dos scripts desarrollados en Python para la detección y clasificación de objetos mediante técnicas de visión artificial utilizando la librería OpenCV.

## Ejecución de los scripts
Ambos scripts se pueden ejecutar enteramente en conjunto con las imágenes sobre las que hará las detecciones. 
Los scripts asumen la existencia de los archivos de entrada `monedas.jpg` (problema1.py) y archivos que cumplan el patrón `img*.png` (problema2.py) en el directorio de ejecución.
Todas las imágenes están suministradas en el repositorio mismo. Pueden intercambiarse por otras pero se debe respetar el patrón de nombres mencionado anteriormente.

En `informe.pdf` se presenta en detalle cómo fue el desarrollo del trabajo.
## 1. Detección de Monedas y Dados (`problema1.py`)

Implementa la detección de figuras circulares mediante la **Transformada de Hough Circular (HoughCircles)**. El algoritmo se divide en dos fases lógicas sobre la misma imagen de entrada (`monedas.jpg`):

* **Clasificación de Monedas:** Detecta círculos de radio mayor y los clasifica en tres categorías (10 centavos, 1 peso, 50 centavos) basándose en umbrales de radio predefinidos.
* **Lectura de Dados:** Detecta círculos de radio menor (puntos de los dados). Implementa una lógica de agrupamiento espacial (clustering) basada en la distancia euclidiana para asociar los puntos a un mismo dado y calcular su valor numérico total.
## 2. Detección de Patentes Vehiculares (`problema2.py`)

Implementa un algoritmo de segmentación y reconocimiento de patrones para localizar patentes en imágenes vehiculares (`img*.png`). El proceso utiliza reglas heurísticas:

* **Preprocesamiento:** Conversión a escala de grises y binarización iterativa.
* **Extracción de Candidatos:** Análisis de componentes conectados (`cv2.connectedComponentsWithStats`). Se filtran regiones basándose en la relación de aspecto (AR) y el área.
* **Agrupamiento y Validación:** Los candidatos se agrupan por proximidad horizontal y alineación vertical. Se aplican criterios estadísticos (desviación estándar de altura y posición Y) para validar si un grupo constituye una patente.

## Requisitos

* Python 3.x
* OpenCV (`opencv-python`)
* NumPy
* Matplotlib

