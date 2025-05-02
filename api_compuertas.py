from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import uvicorn
import io
from PIL import Image
from tensorflow import keras
import os
from huggingface_hub import hf_hub_download
from huggingface_hub import login

# Configuración para usar tensorflow como backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# Inicializar FastAPI
app = FastAPI()

# Configuración
CLASSES = ['and', 'nand', 'nor', 'not', 'or', 'xnor', 'xor']

# Iniciar sesión con token del entorno
login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

# Cargar el modelo desde Hugging Face Hub sin descargarlo
def cargar_modelo_huggingface():
    print("🔄 Cargando modelo desde Hugging Face...")

    # Descargar el archivo del modelo desde Hugging Face Hub
    model_path = hf_hub_download(repo_id="Estebanxdd/smartbool", filename="modelo_compuertas.keras")
    
    # Cargar el modelo Keras
    model = keras.models.load_model(model_path)
    
    print("✅ Modelo cargado correctamente.")
    return model

# Cargar el modelo (solo una vez al inicio)
model = cargar_modelo_huggingface()

# Función de predicción
def predecir_compuerta(imagen_bytes):
    img = Image.open(io.BytesIO(imagen_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Cambié img_to_array por np.array
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    clase_idx = np.argmax(pred, axis=1)[0]

    return CLASSES[clase_idx]

# Endpoint principal
@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        return JSONResponse(content={"error": "El archivo no es una imagen."}, status_code=400)

    imagen_bytes = await file.read()
    try:
        resultado = predecir_compuerta(imagen_bytes)
        return {"resultado": resultado}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Para correr directamente con: python api_compuertas.py
if __name__ == "__main__":
    uvicorn.run("api_compuertas:app", host="0.0.0.0", port=8000, reload=True)
