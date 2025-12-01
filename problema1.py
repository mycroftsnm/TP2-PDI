import cv2
import numpy as np
import matplotlib.pyplot as plt


def imshow(img, new_fig=True, title=None, color_img=False, blocking=True, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

mon = cv2.imread('monedas.jpg', cv2.IMREAD_GRAYSCALE)
f_blur = cv2.GaussianBlur(mon, ksize=(5, 5), sigmaX=1.5)

minDist=350
param2=12
minRadius=110
maxRadius=200
circles = cv2.HoughCircles(mon, 
                           method=cv2.HOUGH_GRADIENT, 
                           dp=1, 
                           minDist=minDist,
                           param1=200,
                           param2=25, 
                           minRadius=minRadius,
                           maxRadius=maxRadius)

Nc = circles.shape[1]
# Dibujo los círculos detectados
circles = np.uint16(np.around(circles))
print(circles)


fc = cv2.cvtColor(mon, cv2.COLOR_GRAY2RGB)

def identificar_moneda(radio):
    if radio < 140:
        return (0,0,255), "10 centavos"
    elif radio < 170:
        return (255,0,0), "1 peso"
    else:
        return (0,255,0), "50 centavos"

for c in circles[0,:]:
    radio = c[2]
    color, tipo = identificar_moneda(radio)
    cv2.circle(fc, (c[0],c[1]), radio, color, 10)   # Dibujo el círculo
    cv2.putText(fc, tipo, (c[0]-40, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

# Visualizo
# imshow(fc, colorbar=False, title=f"Círculos detectados ({Nc}), minDist={minDist}, param2={param2}, minRadius={minRadius}, maxRadius={maxRadius}")

# dados
minDist=40
param2=20
minRadius=25
maxRadius=27
circles = cv2.HoughCircles(f_blur, 
                           method=cv2.HOUGH_GRADIENT, 
                           dp=1, 
                           minDist=minDist, 
                           param2=param2,
                           minRadius=minRadius,
                           maxRadius=maxRadius) 

Nc = circles.shape[1]
# Dibujo los círculos detectados
circles = np.uint16(np.around(circles))


class Dado():
    def __init__(self, x, y):
        self.min_x = x
        self.min_y = y
        self.max_x = x
        self.max_y = y
        self.valor = 1

    def actualizar(self, x, y):
        if x < self.min_x:
            self.min_x = x
        if x > self.max_x:
            self.max_x = x
        if y < self.min_y:
            self.min_y = y
        if y > self.max_y:
            self.max_y = y
        self.valor += 1

circulos = sorted(circles[0,:], key=lambda c: c[0])  # ordeno por posición x
dados = []

distancia_anterior = 0
for c in circulos:
    x, y = c[0], c[1]
    distancia = np.hypot(x, y)
    cv2.circle(fc, (c[0],c[1]), c[2], (255,0,255), 2)
    print(distancia, distancia_anterior)
    if abs(distancia - distancia_anterior) > 200:  # si la distancia es suficientemente distinta, es otro dado
        d = Dado(x, y)
        dados.append(d)
        distancia_anterior = distancia
    else:
        d.actualizar(x, y)

for d in dados:
    cx = (d.min_x + d.max_x) // 2
    cy = (d.min_y + d.max_y) // 2
    max_radio = max(d.max_x - d.min_x, d.max_y - d.min_y) // 2
    cv2.rectangle(fc, (d.min_x - max_radio, d.min_y - max_radio), (d.max_x + max_radio, d.max_y + max_radio), (255,0,255), 5)
    cv2.putText(fc, f"Valor {d.valor}", (d.min_x - max_radio, d.min_y - max_radio - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 5)
        
    

# Visualizo
imshow(fc, colorbar=False, title=f"Círculos detectados ({Nc}), minDist={minDist}, param2={param2}, minRadius={minRadius}, maxRadius={maxRadius}")
