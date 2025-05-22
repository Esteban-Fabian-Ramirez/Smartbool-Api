import cv2
import pytesseract
import numpy as np
from sympy.logic.boolalg import SOPform
from sympy.abc import A, B, C, D

# Configurar Tesseract si es necesario (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocesar_imagen(ruta_img):
    img = cv2.imread(ruta_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img, thresh

def limpiar_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_LINEAR)
    blur = cv2.GaussianBlur(resized, (3, 3), 0)
    _, binarizada = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarizada

def detectar_grilla(thresh_img, original_img, filas, columnas):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 20 < w < 200 and 20 < h < 200:
            cajas.append((x, y, w, h))

    # Ordenar por filas y luego por columnas
    cajas = sorted(cajas, key=lambda b: (round(b[1]/10)*10, b[0]))

    celdas = []
    idx = 0
    for box in cajas:
        x, y, w, h = box
        roi = original_img[y:y+h, x:x+w]
        roi_proc = limpiar_roi(roi)

        # Guardar imagen
        cv2.imwrite(f"celda_{idx}.png", roi_proc)

        text = pytesseract.image_to_string(roi_proc, config='--psm 6 -c tessedit_char_whitelist=01').strip()

        if text in ['0', '1']:
            valor = int(text)
        else:
            valor = 0  # Valor por defecto si la lectura falla

        celdas.append(valor)

    # Reconstruir matriz
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
    ruta_imagen = r"C:\Users\scarb\Downloads\mapak3.png"  # Cambia esto por tu imagen
    filas, columnas = 4, 4  # Ajusta aquí según el tamaño del K-map
    n_vars = 4              # 2, 3 o 4 según el mapa

    img, thresh = preprocesar_imagen(ruta_imagen)
    matriz = detectar_grilla(thresh, img, filas, columnas)

    print("[INFO] Matriz del K-Map:")
    for fila in matriz:
        print(fila)

    minterms = kmap_a_minterms(matriz)
    print(f"[INFO] Minterminos detectados: {minterms}")

    funcion = simplificar_funcion(minterms, n_vars)
    print(f"[RESULTADO] Función booleana simplificada:\n{funcion}")

if __name__ == "__main__":
    main()
