# ... (importaciones iniciales sin cambios) ...
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import uvicorn
from sympy import symbols, simplify_logic, to_dnf
from sympy.logic.boolalg import truth_table
from sympy.logic.boolalg import BooleanFunction, Or, And
import pytesseract
import re
import numpy as np
import os
import io
from PIL import ImageOps

# Configuración para usar tensorflow como backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# Inicializar FastAPI
app = FastAPI()

# Configuración de clases
CLASSES = ['and', 'nand', 'nor', 'not', 'or', 'xnor', 'xor']

# Clasificar imagen según contenido
def clasificar_imagen_contenido(img: Image.Image) -> str:
    text = pytesseract.image_to_string(img).lower()
    if "and" in text:
        print("OCR también detecta 'and'")
    if re.search(r'(and|or|not|xor|nand|nor)', text):
        return 'expresion'
    elif re.search(r'[01]\s+[01]', text) and len(text.splitlines()) > 2:
        return 'tabla'
    else:
        return 'diagrama'

# Procesar expresión booleana
def procesar_expresion(expr_texto: str):
    expr_texto = expr_texto.upper().replace("AND", "&").replace("OR", "|").replace("NOT", "~")
    expr = simplify_logic(expr_texto)
    variables = sorted(expr.free_symbols, key=lambda x: str(x))
    tabla = [[*entrada, int(bool(salida))] for entrada, salida in truth_table(expr, variables)]
    kmap = generar_karnaugh(variables, tabla)
    return str(expr), tabla, kmap

# Procesar tabla de verdad
def procesar_tabla(texto_tabla: str):
    filas_raw = [list(map(int, re.findall(r'[01]', linea))) for linea in texto_tabla.strip().splitlines()]
    n_vars = len(filas_raw[0]) - 1
    variables = symbols(f'A:{n_vars}')
    entradas = [tuple(fila[:-1]) for fila in filas_raw]
    salidas = [fila[-1] for fila in filas_raw]

    terms = [And(*[var if val else ~var for var, val in zip(variables, entrada)])
             for entrada, salida in zip(entradas, salidas) if salida == 1]

    expr = simplify_logic(Or(*terms)) if terms else False
    tabla = [list(e) + [int(bool(s))] for e, s in zip(entradas, salidas)]
    kmap = generar_karnaugh(variables, tabla)

    return str(expr), tabla, kmap

# Procesar diagrama (placeholder)
def procesar_diagrama(img: Image.Image):
    expresion = "A & B"
    tabla = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
    return expresion, tabla

@app.post("/analizar")
async def analizar(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return JSONResponse(content={"error": "El archivo no es una imagen."}, status_code=400)

    imagen_bytes = await file.read()
    img = Image.open(io.BytesIO(imagen_bytes)).convert('RGB')

    tipo = clasificar_imagen_contenido(img)

    try:
        if tipo == 'expresion':
            texto = pytesseract.image_to_string(img)
            expresion, tabla, kmap = procesar_expresion(texto)
        elif tipo == 'tabla':
            texto = pytesseract.image_to_string(img)
            expresion, tabla, kmap = procesar_tabla(texto)
        else:
            expresion, tabla = procesar_diagrama(img)
            kmap = generar_karnaugh(symbols("A B"), tabla)

        return {
            "tipo_detectado": tipo,
            "expresion_booleana": expresion,
            "tabla_verdad": tabla,
            "karnaugh": kmap
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Cargar el modelo localmente desde archivo
def cargar_modelo_local():
    print("🔄 Cargando modelo local...")
    model = keras.models.load_model("modelo_compuertas.keras")  # Ruta local
    print("✅ Modelo cargado correctamente.")
    return model

@app.on_event("startup")
def startup_event():
    global model
    try:
        print("🚀 Cargando modelo en startup...")
        model = cargar_modelo_local()
        print("✅ Modelo cargado correctamente.")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")

# Predicción
def predecir_compuerta(imagen_bytes):
    img = Image.open(io.BytesIO(imagen_bytes)).convert('RGB')
    #img = ImageOps.invert(img)  # solo si es necesario
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    clase_idx = np.argmax(pred, axis=1)[0]
    print("🔍 Predicciones brutas:", pred)
    print("🎯 Clase predicha:", CLASSES[clase_idx])
    return CLASSES[clase_idx]

@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return JSONResponse(content={"error": "El archivo no es una imagen."}, status_code=400)
    
    imagen_bytes = await file.read()
    try:
        resultado = predecir_compuerta(imagen_bytes)
        return {"resultado": resultado}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predecir_y_analizar")
async def predecir_y_analizar(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return JSONResponse(content={"error": "El archivo no es una imagen."}, status_code=400)

    imagen_bytes = await file.read()
    try:
        clase = predecir_compuerta(imagen_bytes)
        A, B = symbols('A B')
        expr_map = {
            'and': And(A, B),
            'or': Or(A, B),
            'not': ~A,
            'nand': ~(And(A, B)),
            'nor': ~(Or(A, B)),
            'xor': Or(And(A, ~B), And(~A, B)),
            'xnor': ~Or(And(A, ~B), And(~A, B))
        }

        expr = expr_map.get(clase)
        expr_simplificada = str(simplify_logic(expr)) if expr else "No reconocida"
        variables = (A, B) if clase not in ['not'] else (A,)
        tabla = list(truth_table(expr, variables))
        tabla_lista = [[*inputs, int(bool(salida))] for inputs, salida in tabla]
        kmap = generar_karnaugh(variables, tabla_lista)

        return {
            "compuerta_predicha": clase,
            "expresion_booleana": expr_simplificada,
            "tabla_verdad": tabla_lista,
            "karnaugh": kmap
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def generar_karnaugh(variables, tabla):
    if not tabla:
        return []

    n = len(variables)
    entradas = [tuple(row[:-1]) for row in tabla]
    salidas = [row[-1] for row in tabla]
    mapa = {}

    for entrada, salida in zip(entradas, salidas):
        mapa[entrada] = salida

    if n == 1:
        return [[mapa.get((0,), 0)], [mapa.get((1,), 0)]]
    elif n == 2:
        return [
            [mapa.get((0, 0), 0), mapa.get((0, 1), 0)],
            [mapa.get((1, 0), 0), mapa.get((1, 1), 0)]
        ]
    elif n == 3:
        return [
            [mapa.get((0, 0, 0), 0), mapa.get((0, 0, 1), 0)],
            [mapa.get((0, 1, 0), 0), mapa.get((0, 1, 1), 0)],
            [mapa.get((1, 0, 0), 0), mapa.get((1, 0, 1), 0)],
            [mapa.get((1, 1, 0), 0), mapa.get((1, 1, 1), 0)]
        ]
    elif n == 4:
        return [
            [mapa.get((0, 0, 0, 0), 0), mapa.get((0, 0, 0, 1), 0),
             mapa.get((0, 0, 1, 1), 0), mapa.get((0, 0, 1, 0), 0)],
            [mapa.get((0, 1, 0, 0), 0), mapa.get((0, 1, 0, 1), 0),
             mapa.get((0, 1, 1, 1), 0), mapa.get((0, 1, 1, 0), 0)],
            [mapa.get((1, 1, 0, 0), 0), mapa.get((1, 1, 0, 1), 0),
             mapa.get((1, 1, 1, 1), 0), mapa.get((1, 1, 1, 0), 0)],
            [mapa.get((1, 0, 0, 0), 0), mapa.get((1, 0, 0, 1), 0),
             mapa.get((1, 0, 1, 1), 0), mapa.get((1, 0, 1, 0), 0)]
        ]
    else:
        return [["No soportado para más de 4 variables"]]

@app.post("/calcular_expresion")
async def calcular_expresion(data: dict):
    expr_texto = data.get("expresion", "")
    if not expr_texto:
        return JSONResponse(content={"error": "No se proporcionó una expresión."}, status_code=400)

    try:
        # Convertir la expresión booleana a una forma estándar (con operadores como & y |)
        expr_texto = expr_texto.replace("*", "&").replace("+", "|").replace("^", "^").replace("~", "~")

        # Simplificar la expresión booleana usando sympy
        expr = simplify_logic(expr_texto)

        # Obtener las variables de la expresión y ordenarlas
        variables = sorted(expr.free_symbols, key=lambda x: str(x))
        
        # Generar la tabla de verdad de la expresión booleana
        tabla = [[*entrada, int(bool(salida))] for entrada, salida in truth_table(expr, variables)]

        # Generar el mapa de Karnaugh para la tabla de verdad
        kmap = generar_karnaugh(variables, tabla)

        # Devolver el resultado
        return {
            "expresion_original": data.get("expresion", ""),
            "expresion_convertida": expr_texto,
            "expresion_simplificada": str(expr),
            "tabla_verdad": tabla,
            "karnaugh": kmap
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api_compuertas:app", host="0.0.0.0", port=port)
