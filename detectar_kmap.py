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

def extraer_celdas(thresh_img, original_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    celdas = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20 and w < 200 and h < 200:  # Tamaño típico de celda
            roi = original_img[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(roi_gray, config='--psm 10 digits')
            try:
                valor = int(text.strip())
                celdas.append((x, y, valor))
            except:
                pass  # Ignorar celdas mal leídas

    # Ordenar por posición (fila, columna)
    celdas_ordenadas = sorted(celdas, key=lambda c: (round(c[1]/10)*10, c[0]))
    return celdas_ordenadas

def construir_matriz(celdas, filas, columnas):
    matriz = [[0 for _ in range(columnas)] for _ in range(filas)]
    idx = 0
    for i in range(filas):
        for j in range(columnas):
            if idx < len(celdas):
                matriz[i][j] = celdas[idx][2]
                idx += 1
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
    filas, columnas = 4, 4     # Ajusta según tu K-map

    img, thresh = preprocesar_imagen(ruta_imagen)
    celdas = extraer_celdas(thresh, img)
    print(f"[INFO] Celdas detectadas: {len(celdas)}")

    matriz = construir_matriz(celdas, filas, columnas)
    print("[INFO] Matriz del K-Map:")
    for fila in matriz:
        print(fila)

    minterms = kmap_a_minterms(matriz)
    print(f"[INFO] Minterminos detectados: {minterms}")

    funcion = simplificar_funcion(minterms, n_vars=4)
    print(f"[RESULTADO] Función booleana simplificada:\n{funcion}")

if __name__ == "__main__":
    main()
