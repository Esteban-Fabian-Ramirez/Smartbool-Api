import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
from huggingface_hub import hf_hub_download, login
from PIL import Image

# Configuración para usar tensorflow como backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# Inicializar FastAPI
app = FastAPI()

# Configuración de clases
CLASSES = ['and', 'nand', 'nor', 'not', 'or', 'xnor', 'xor']

# Iniciar sesión con token del entorno (asegúrate de tener HUGGINGFACE_HUB_TOKEN configurado en Render)
login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

# Función para cargar el modelo desde Hugging Face Hub
def cargar_modelo_huggingface():
    print("🔄 Cargando modelo desde Hugging Face...")
    # Descargar el archivo del modelo desde Hugging Face Hub
    model_path = hf_hub_download(repo_id="Estebanxdd/smartbool", filename="modelo_compuertas.keras")
    
    # Cargar el modelo Keras
    model = keras.models.load_model(model_path)
    
    print("✅ Modelo cargado correctamente.")
    return model

# Cargar el modelo en el evento de startup (esto evita bloquear el arranque)
@app.on_event("startup")
def startup_event():
    global model
    model = cargar_modelo_huggingface()

# Función de predicción
def predecir_compuerta(imagen_bytes):
    img = Image.open(io.BytesIO(imagen_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalización
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)
    clase_idx = np.argmax(pred, axis=1)[0]
    
    return CLASSES[clase_idx]

# Endpoint principal para predecir la compuerta
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
