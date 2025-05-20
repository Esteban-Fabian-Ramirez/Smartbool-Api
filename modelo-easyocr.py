import cv2
import easyocr
import os
import re

# Ruta de la imagen original
image_path = r"C:\Users\scarb\Downloads\compuerta-and-1.webp"

# Cargar imagen y verificar
image = cv2.imread(image_path)
if image is None:
    print(f"Error al cargar la imagen en la ruta: {image_path}")
    exit()

# Convertir a escala de grises y aplicar filtro bilateral
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Umbral adaptativo para mejorar OCR
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 6
)

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

# Normalizar caracteres comunes mal leídos
def normalizar_texto(texto):
    texto = texto.upper()
    return texto.replace('O', '0').replace('I', '1').replace('L', '1')

# Agrupar textos por línea Y
lines = {}
for bbox, text, prob in results:
    top_y = int(bbox[0][1])
    key = top_y // 20
    lines.setdefault(key, []).append((bbox, normalizar_texto(text), prob))

# Validar si una línea parece válida (solo bits 0 o 1, 2 o 3 elementos)
def es_linea_valida(texto_linea):
    solo_bits = re.findall(r'[01]', ' '.join(texto_linea))
    return len(solo_bits) in [2, 3] and all(c in '01' for c in solo_bits)

# Extraer tabla
tabla = []
print("\n--- Tabla Reconocida ---")
for key in sorted(lines.keys()):
    linea = sorted(lines[key], key=lambda x: x[0][0][0])
    texto_linea = [t[1] for t in linea]
    if es_linea_valida(texto_linea):
        print(' '.join(texto_linea))
        bits = re.findall(r'[01]', ' '.join(texto_linea))
        tabla.append(bits)

# Combinar fragmentos para formar filas completas
def limpiar_tabla(tabla_cruda):
    tabla_limpia = []
    buffer = []

    for fila in tabla_cruda:
        buffer += fila
        if len(buffer) >= 3:
            tabla_limpia.append(buffer[:3])
            buffer = buffer[3:]
    return tabla_limpia

# Identificar la compuerta lógica
def identificar_compuerta(tabla):
    tabla_set = set(tuple(fila) for fila in tabla if len(fila) >= 2)

    compuertas = {
        "AND":  {('0', '0', '0'), ('0', '1', '0'), ('1', '0', '0'), ('1', '1', '1')},
        "OR":   {('0', '0', '0'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '1')},
        "XOR":  {('0', '0', '0'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '0')},
        "NAND": {('0', '0', '1'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '0')},
        "NOR":  {('0', '0', '1'), ('0', '1', '0'), ('1', '0', '0'), ('1', '1', '0')},
        "XNOR": {('0', '0', '1'), ('0', '1', '0'), ('1', '0', '0'), ('1', '1', '1')},
        "NOT":  {('0', '1'), ('1', '0')}
    }

    mejor_match = "Compuerta no reconocida"
    mejor_puntaje = 0

    for nombre, verdad_set in compuertas.items():
        interseccion = tabla_set & verdad_set
        puntaje = len(interseccion) / len(verdad_set)

        if puntaje > mejor_puntaje and puntaje >= 0.6:
            mejor_match = nombre
            mejor_puntaje = puntaje

    return mejor_match

# Imprimir tabla de verdad esperada
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

# Proceso final
tabla_limpia = limpiar_tabla(tabla)
nombre_detectado = identificar_compuerta(tabla_limpia)

# Imprimir resultado
imprimir_tabla_predefinida(nombre_detectado)
print(f"\n🔍 Compuerta lógica detectada: **{nombre_detectado}**")

# Mostrar diferencias si hay error
compuertas_def = {
    "AND":  [('0', '0', '0'), ('0', '1', '0'), ('1', '0', '0'), ('1', '1', '1')],
    "OR":   [('0', '0', '0'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '1')],
    "XOR":  [('0', '0', '0'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '0')],
    "NAND": [('0', '0', '1'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '0')],
    "NOR":  [('0', '0', '1'), ('0', '1', '0'), ('1', '0', '0'), ('1', '1', '0')],
    "XNOR": [('0', '0', '1'), ('0', '1', '0'), ('1', '0', '0'), ('1', '1', '1')],
    "NOT":  [('0', '1'), ('1', '0')],
}

if nombre_detectado in compuertas_def:
    esperada = set(compuertas_def[nombre_detectado])
    observada = set(tuple(f) for f in tabla_limpia)
    diferencia = observada - esperada
    if diferencia:
        print("\n⚠️ Valores no coincidentes con la compuerta detectada:")
        for fila in diferencia:
            print(f"  Detectado: {fila}")

# Eliminar imagen temporal
if os.path.exists(processed_path):
    os.remove(processed_path)
