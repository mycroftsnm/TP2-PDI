

import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_patente(img):
    """
    Función con procesamiento de imagen con el objetivo de la detección de los caracteres de las patentes.
    Recibe una imagen y devuelve la imagen procesada, el ROI de la patente y los caracteres individuales.
    
    img: imagen en RGB (numpy array)
    return: tuple (img_resultado, roi_patente, caracteres_individuales)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    mejor_grupo = []
    mejor_umbral = -1
    encontrado = False
    
    # PRIMERA PASADA: Búsqueda estricta (exactamente 6 caracteres)
    for th in range(0, 255, 1):
        ret, img_binaria = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY) 
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            img_binaria, connectivity=8
        )
        
        # Filtrar candidatos básicos
        candidatos = []
        h_img, w_img = img.shape[:2]
        
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            if w == 0: continue
            ar = h / float(w)
            
            es_forma_valida = (1.3 <= ar <= 2.8)
            es_tamano_valido = (area > 30) and (h < h_img * 0.6)
            
            if es_forma_valida and es_tamano_valido:
                candidatos.append({'id': i, 'x': x, 'y': y, 'w': w, 'h': h, 'area': area})
        
        # Agrupar candidatos por proximidad
        if candidatos:
            candidatos.sort(key=lambda c: c['x'])
            grupos = []
            grupo_actual = [candidatos[0]]
            
            for i in range(1, len(candidatos)):
                c_prev = candidatos[i-1]
                c_curr = candidatos[i]
                
                distancia = c_curr['x'] - (c_prev['x'] + c_prev['w'])
                umbral_distancia = c_prev['w'] * 2.5 
                desalineacion_y = abs(c_curr['y'] - c_prev['y'])
                
                if distancia < umbral_distancia and desalineacion_y < c_prev['h'] * 0.5:
                    grupo_actual.append(c_curr)
                else:
                    grupos.append(grupo_actual)
                    grupo_actual = [c_curr]
            
            grupos.append(grupo_actual)
            
            # Verificar cada grupo con CONDICIONES ESTRICTAS
            for grupo in grupos:
                if len(grupo) != 6:  # Exactamente 6 caracteres
                    continue
                
                alturas = [c['h'] for c in grupo]
                centros_y = [c['y'] + c['h']/2 for c in grupo]
                
                promedio_h = np.mean(alturas)
                std_h = np.std(alturas)
                std_y = np.std(centros_y)
                
                # Verificar consistencia estricta
                if std_h > promedio_h * 0.35:
                    continue
                if std_y > promedio_h * 0.3:
                    continue
                
                # ¡Patente encontrada con criterio estricto!
                mejor_grupo = grupo
                mejor_umbral = th
                encontrado = True
                break
        
        if encontrado:
            break
    
    # SEGUNDA PASADA: Si no encontró con criterio estricto, intenta con permisivo
    if not encontrado:
        for th in range(0, 255, 1):
            ret, img_binaria = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY) 
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                img_binaria, connectivity=8
            )
            
            # Filtrar candidatos básicos (mismo código)
            candidatos = []
            h_img, w_img = img.shape[:2]
            
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                
                if w == 0: continue
                ar = h / float(w)
                
                es_forma_valida = (1.3 <= ar <= 2.8)
                es_tamano_valido = (area > 30) and (h < h_img * 0.6)
                
                if es_forma_valida and es_tamano_valido:
                    candidatos.append({'id': i, 'x': x, 'y': y, 'w': w, 'h': h, 'area': area})
            
            # Agrupar candidatos (mismo código)
            if candidatos:
                candidatos.sort(key=lambda c: c['x'])
                grupos = []
                grupo_actual = [candidatos[0]]
                
                for i in range(1, len(candidatos)):
                    c_prev = candidatos[i-1]
                    c_curr = candidatos[i]
                    
                    distancia = c_curr['x'] - (c_prev['x'] + c_prev['w'])
                    umbral_distancia = c_prev['w'] * 2.5 
                    desalineacion_y = abs(c_curr['y'] - c_prev['y'])
                    
                    if distancia < umbral_distancia and desalineacion_y < c_prev['h'] * 0.5:
                        grupo_actual.append(c_curr)
                    else:
                        grupos.append(grupo_actual)
                        grupo_actual = [c_curr]
                
                grupos.append(grupo_actual)
                
                # Verificar cada grupo con CONDICIONES PERMISIVAS
                for grupo in grupos:
                    if not (5 <= len(grupo) <= 7):  # 5 a 7 caracteres
                        continue
                    
                    alturas = [c['h'] for c in grupo]
                    centros_y = [c['y'] + c['h']/2 for c in grupo]
                    
                    promedio_h = np.mean(alturas)
                    std_h = np.std(alturas)
                    std_y = np.std(centros_y)
                    
                    # Verificar consistencia más permisiva
                    if std_h > promedio_h * 0.4:  # Más permisivo (antes 0.35)
                        continue
                    if std_y > promedio_h * 0.35:  # Más permisivo (antes 0.3)
                        continue
                    
                    # Patente encontrada con criterio permisivo
                    mejor_grupo = grupo
                    mejor_umbral = th
                    encontrado = True
                    break
            
            if encontrado:
                break
    
    # Visualizar resultado
    img_resultado = img.copy()
    roi_patente = None
    caracteres_individuales = []
    
    if not mejor_grupo:
        h, w = img.shape[:2]
        cv2.putText(img_resultado, "NO DETECTADO", (10, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        # Dibujar caracteres individuales
        for c in mejor_grupo:
            cv2.rectangle(img_resultado, (c['x'], c['y']), 
                         (c['x'] + c['w'], c['y'] + c['h']), (0, 255, 0), 2)
        
        # Calcular ROI envolvente
        xs = [c['x'] for c in mejor_grupo]
        ys = [c['y'] for c in mejor_grupo]
        ws = [c['w'] for c in mejor_grupo]
        hs = [c['h'] for c in mejor_grupo]
        
        min_x, min_y = min(xs), min(ys)
        max_x = max([x + w for x, w in zip(xs, ws)])
        max_y = max([y + h for y, h in zip(ys, hs)])
        
        # Dibujar caja de patente
        padding = 5
        p1 = (max(0, min_x - padding), max(0, min_y - padding))
        p2 = (min(img.shape[1], max_x + padding), min(img.shape[0], max_y + padding))
        
        cv2.rectangle(img_resultado, p1, p2, (0, 0, 255), 3)
        
        # Indicar si fue detectado con criterio estricto o permisivo
        criterio = "ESTRICTO" if len(mejor_grupo) == 6 else "PERMISIVO"
        texto = f"PATENTE-{criterio} ({len(mejor_grupo)} chars, Th:{mejor_umbral})"
        cv2.putText(img_resultado, texto, (p1[0]-100, p1[1]+65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
        
        # Extraer ROI de la patente
        roi_patente = img[p1[1]:p2[1], p1[0]:p2[0]].copy()
        
        # Extraer caracteres individuales
        for c in mejor_grupo:
            char_roi = img[c['y']:c['y']+c['h'], c['x']:c['x']+c['w']].copy()
            caracteres_individuales.append(char_roi)
    
    return img_resultado, roi_patente, caracteres_individuales


def procesar_y_mostrar(ruta_patron='img*.png', cols=3, figsize=(15, 12)):
    """
    Carga todas las imágenes, las procesa con detectar_patente() 
    y las muestra en una grilla.
    Luego muestra una segunda figura con todas las patentes y caracteres en 2 columnas.
    
    ruta_patron: patrón para buscar imágenes (ej: 'img*.png', 'imagenes/*.jpg')
    cols: columnas en la grilla de detecciones
    figsize: tamaño de la figura
    """
    from pathlib import Path
    
    # Cargar imágenes
    paths = sorted(Path('.').glob(ruta_patron))
    imagenes = []
    nombres = []
    
    for path in paths:
        img = cv2.imread(str(path))
        if img is not None:
            imagenes.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            nombres.append(path.name)

    # Procesar cada imagen
    resultados = []
    for img, nombre in zip(imagenes, nombres):
        resultado, roi_patente, caracteres = detectar_patente(img)
        resultados.append((resultado, roi_patente, caracteres, nombre))
    
    # PRIMERA FIGURA: Mostrar todas las detecciones en grilla
    n = len(resultados)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, (img_resultado, roi_patente, caracteres, nombre) in enumerate(resultados):
        if len(img_resultado.shape) == 2:  # Imagen en grises
            axes[i].imshow(img_resultado, cmap='gray')
        else:  # Imagen a color
            axes[i].imshow(img_resultado)
        axes[i].set_title(nombre)
        axes[i].axis('off')
    
    # Ocultar ejes sobrantes
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # SEGUNDA FIGURA: Mostrar todas las patentes y caracteres en 2 columnas y 6 filas
    n_patentes = len(resultados)
    rows_patentes = (n_patentes + 1) // 2  # 6 filas para 12 imágenes
    
    fig, axes = plt.subplots(rows_patentes, 4, figsize=(16, rows_patentes * 2.5))
    axes = axes.flatten()
    
    idx_subplot = 0
    for img_resultado, roi_patente, caracteres, nombre in resultados:
        # Cada patente ocupa 2 subplots (patente + caracteres)
        row = idx_subplot // 2
        col_start = (idx_subplot % 2) * 2
        
        ax_patente = axes[row * 4 + col_start]
        ax_chars = axes[row * 4 + col_start + 1]
        
        if roi_patente is not None and caracteres:
            # Mostrar ROI de la patente
            ax_patente.imshow(roi_patente)
            ax_patente.set_title(f'{nombre}')
            ax_patente.axis('off')
            
            # Mostrar caracteres individuales
            max_h = max([c.shape[0] for c in caracteres])
            
            # Redimensionar caracteres a la misma altura
            chars_resized = []
            for char in caracteres:
                h, w = char.shape[:2]
                new_w = int(w * max_h / h)
                char_resized = cv2.resize(char, (new_w, max_h), interpolation=cv2.INTER_LINEAR)
                chars_resized.append(char_resized)
            
            # Agregar espacios blancos entre caracteres
            espacio = 10
            espacio_blanco = np.ones((max_h, espacio, 3), dtype=np.uint8) * 255
            
            # Concatenar con espacios
            chars_con_espacios = []
            for j, char in enumerate(chars_resized):
                chars_con_espacios.append(char)
                if j < len(chars_resized) - 1:
                    chars_con_espacios.append(espacio_blanco)
            
            chars_concatenados = np.hstack(chars_con_espacios)
            ax_chars.imshow(chars_concatenados)
            ax_chars.set_title(f'Caracteres ({len(caracteres)})')
            ax_chars.axis('off')
        else:
            # No se detectó patente
            ax_patente.text(0.5, 0.5, f'{nombre}\nNO DETECTADO', 
                           ha='center', va='center', fontsize=10, color='red')
            ax_patente.axis('off')
            ax_chars.axis('off')
        
        idx_subplot += 1
    
    # Ocultar subplots sobrantes si hay menos de 12 imágenes
    for i in range(idx_subplot * 2, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


# Ejecutar
procesar_y_mostrar('img*.png')