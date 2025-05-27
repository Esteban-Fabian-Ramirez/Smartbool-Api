import cv2
import pytesseract
import numpy as np
from sympy.logic.boolalg import SOPform
from sympy.abc import A, B, C, D
from PIL import Image
from collections import defaultdict

# Ruta a Tesseract si usas Windows (descomenta si es necesario)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocesar_imagen(ruta_img):
    img = cv2.imread(ruta_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img, thresh

def remover_lineas(roi_bin):
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (roi_bin.shape[1] // 15, 1))
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, roi_bin.shape[0] // 15))

    horizontal_lines = cv2.morphologyEx(roi_bin, cv2.MORPH_OPEN, kernel_horizontal, iterations=1)
    vertical_lines = cv2.morphologyEx(roi_bin, cv2.MORPH_OPEN, kernel_vertical, iterations=1)

    mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    cleaned = cv2.bitwise_and(roi_bin, cv2.bitwise_not(mask))

    return cleaned

def limpiar_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (roi.shape[1]*3, roi.shape[0]*3), interpolation=cv2.INTER_CUBIC)
    bordered = cv2.copyMakeBorder(resized, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)

    adapt = cv2.adaptiveThreshold(bordered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 31, 10)

    # Eliminar ruido de líneas
    limpio = remover_lineas(adapt)

    # Revertir para Tesseract si se invierte (texto blanco en fondo negro)
    invertido = cv2.bitwise_not(limpio)

    return invertido

def agrupar_por_filas(cajas, tolerancia=20):
    filas_dict = defaultdict(list)
    for box in cajas:
        y = box[1]
        asignado = False
        for fila_y in filas_dict:
            if abs(y - fila_y) <= tolerancia:
                filas_dict[fila_y].append(box)
                asignado = True
                break
        if not asignado:
            filas_dict[y].append(box)

    filas_ordenadas = []
    for k in sorted(filas_dict.keys()):
        fila = sorted(filas_dict[k], key=lambda b: b[0])
        filas_ordenadas.extend(fila)
    return filas_ordenadas

def detectar_grilla(thresh_img, original_img, filas, columnas):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 20 < w < 300 and 20 < h < 300:
            cajas.append((x, y, w, h))

    cajas = agrupar_por_filas(cajas)

    img_debug = original_img.copy()
    celdas = []
    idx = 0

    for box in cajas:
        x, y, w, h = box
        roi = original_img[y:y+h, x:x+w]
        roi_proc = limpiar_roi(roi)

        # Guardar para depuración
        cv2.imwrite(f"celda_{idx}.png", roi_proc)
        Image.fromarray(roi_proc).save(f"celda_{idx}_dpi.png", dpi=(300, 300))

        # OCR con configuración más robusta
        config = '--psm 11 -c tessedit_char_whitelist=01'
        text = pytesseract.image_to_string(roi_proc, config=config).strip()

        print(f"[OCR] Texto detectado en celda {idx}: '{text}'")

        if text.isdigit() and int(text) in (0, 1):
            valor = int(text)
            color = (0, 255, 0)
        else:
            print(f"[WARN] Texto no válido en celda {idx}: '{text}', asignando 0.")
            valor = 0
            color = (0, 0, 255)

        # Dibujar sobre imagen para depuración
        cv2.rectangle(img_debug, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_debug, str(valor), (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        celdas.append(valor)
        idx += 1

    cv2.imwrite("debug_cajas_detectadas.png", img_debug)

    # Armar la matriz
    matriz = []
    idx = 0
    for i in range(filas):
        fila = []
        for j in range(columnas):
            if idx < len(celdas):
                fila.append(celdas[idx])
                idx += 1
            else:
                fila.append(0)
        matriz.append(fila)

    return matriz

def kmap_a_minterms(matriz):
    filas, columnas = len(matriz), len(matriz[0])
    minterms = []

    for i in range(filas):
        for j in range(columnas):
            if matriz[i][j] == 1:
                index = i * columnas + j
                minterms.append(index)

    return minterms

def simplificar_funcion(minterms, n_vars):
    variables = [A, B, C, D][:n_vars]
    return SOPform(variables, minterms)

def main():
    ruta_imagen = r"C:\Users\scarb\Downloads\mapak3.png"  # Cambia esta ruta
    filas, columnas = 4, 4
    n_vars = 4

    img, thresh = preprocesar_imagen(ruta_imagen)
    matriz = detectar_grilla(thresh, img, filas, columnas)

    print("\n[INFO] Matriz del K-Map:")
    for fila in matriz:
        print(fila)

    minterms = kmap_a_minterms(matriz)
    print(f"\n[INFO] Minterminos detectados: {minterms}")

    funcion = simplificar_funcion(minterms, n_vars)
    print(f"\n[RESULTADO] Función booleana simplificada:\n{funcion}")

if __name__ == "__main__":
    main()
