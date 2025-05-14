import cv2
import easyocr
import os
import re

# Ruta de la imagen original
image_path = r"C:\Users\scarb\Downloads\compuerta-and-3.webp"

# Cargar imagen y convertir a escala de grises
image = cv2.imread(image_path)
if image is None:
    print(f"Error al cargar la imagen en la ruta: {image_path}")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aumentar contraste
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Aumentar resolución
scale_percent = 250
width = int(thresh.shape[1] * scale_percent / 100)
height = int(thresh.shape[0] * scale_percent / 100)
resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_LINEAR)

# Guardar imagen temporal
processed_path = "temp_processed.png"
cv2.imwrite(processed_path, resized)

# Crear lector OCR
reader = easyocr.Reader(['en', 'es'])

# Leer texto con coordenadas
results = reader.readtext(processed_path)

# Agrupar textos por línea Y
lines = {}
for bbox, text, prob in results:
    top_y = int(bbox[0][1])
    key = top_y // 20
    lines.setdefault(key, []).append((bbox, text, prob))

# Ordenar y extraer tabla como listas
tabla = []
print("\n--- Tabla Reconocida ---")
for key in sorted(lines.keys()):
    linea = sorted(lines[key], key=lambda x: x[0][0][0])
    texto_linea = [t[1] for t in linea]
    print(' '.join(texto_linea))
    # Extraer solo bits (0 o 1) de la línea
    bits = re.findall(r'[01]', ' '.join(texto_linea))
    if len(bits) in [2, 3]:  # NOT (2 bits) o compuertas binarias (3 bits)
        tabla.append(bits)

# Función para identificar la compuerta lógica
def identificar_compuerta(tabla):
    # Convertimos filas a tuplas para que sean comparables en sets
    tabla_set = set(tuple(fila) for fila in tabla if len(fila) >= 2)

    compuertas = {
        "AND": {('0', '0', '0'), ('0', '1', '0'), ('1', '0', '0'), ('1', '1', '1')},
        "OR":  {('0', '0', '0'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '1')},
        "XOR": {('0', '0', '0'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '0')},
        "NAND":{('0', '0', '1'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '0')},
        "NOR": {('0', '0', '1'), ('0', '1', '0'), ('1', '0', '0'), ('1', '1', '0')},
        "XNOR":{('0', '0', '1'), ('0', '1', '0'), ('1', '0', '0'), ('1', '1', '1')},
        "NOT": {('0', '1'), ('1', '0')}
    }

    mejor_match = "Compuerta no reconocida"
    mejor_puntaje = 0

    for nombre, verdad_set in compuertas.items():
        interseccion = tabla_set & verdad_set
        puntaje = len(interseccion) / len(verdad_set)

        if puntaje > mejor_puntaje and puntaje >= 0.75:  # al menos 75% de coincidencia
            mejor_match = nombre
            mejor_puntaje = puntaje

    return mejor_match

def imprimir_tabla_predefinida(nombre):
    compuertas = {
        "AND":  [('0', '0', '0'), ('0', '1', '0'), ('1', '0', '0'), ('1', '1', '1')],
        "OR":   [('0', '0', '0'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '1')],
        "XOR":  [('0', '0', '0'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '0')],
        "NAND": [('0', '0', '1'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '0')],
        "NOR":  [('0', '0', '1'), ('0', '1', '0'), ('1', '0', '0'), ('1', '1', '0')],
        "XNOR": [('0', '0', '1'), ('0', '1', '0'), ('1', '0', '0'), ('1', '1', '1')],
        "NOT":  [('0', '1'), ('1', '0')],
    }

    if nombre in compuertas:
        print(f"\n📋 Tabla de verdad de la compuerta {nombre}:")
        encabezado = "A B Q" if len(compuertas[nombre][0]) == 3 else "A Q"
        print(encabezado)
        for fila in compuertas[nombre]:
            print(' '.join(fila))
    else:
        print("\nNo se puede imprimir la tabla: compuerta desconocida.")

# Combinar líneas incompletas
def limpiar_tabla(tabla_cruda):
    tabla_limpia = []
    buffer = []

    for fila in tabla_cruda:
        buffer += fila
        if len(buffer) == 3:
            tabla_limpia.append(buffer)
            buffer = []
    return tabla_limpia

# Intentar identificar
tabla_limpia = limpiar_tabla(tabla)
nombre_detectado = identificar_compuerta(tabla_limpia)

# Imprimir resultado
imprimir_tabla_predefinida(nombre_detectado)
print(f"\n🔍 Comparta lógica detectada: **{nombre_detectado}**")

# Limpiar temporal
if os.path.exists(processed_path):
    os.remove(processed_path)
