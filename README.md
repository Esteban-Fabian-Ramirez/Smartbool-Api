# 🔍 API de Análisis y Clasificación de Compuertas Lógicas

Este proyecto es una API construida con **FastAPI** que permite:

- 🧠 Clasificar imágenes de compuertas lógicas mediante un modelo CNN (`modelo_compuertas.keras`)
- 🔤 Extraer y analizar expresiones booleanas desde imágenes usando OCR
- 📊 Generar tablas de verdad y mapas de Karnaugh
- 🔀 Simplificar expresiones booleanas mediante `sympy`

---

## 🧰 Tecnologías utilizadas

- [FastAPI](https://fastapi.tiangolo.com/)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [SymPy](https://www.sympy.org/)
- [Pytesseract](https://github.com/madmaze/pytesseract)
- [Uvicorn](https://www.uvicorn.org/)
- [Pillow](https://pillow.readthedocs.io/)

---

## 📦 Requisitos

Asegúrate de tener instalado:

- Python 3.8+
- pip
- Tesseract OCR (instalación requerida aparte)

### 🔧 Instalación de Tesserac

#### En Windows
Descarga desde: https://github.com/tesseract-ocr/tesseract

Instálalo y añade su ruta al sistema (por ejemplo: C:\Program Files\Tesseract-OCR\tesseract.exe)

Si es necesario, descomenta y edita esta línea en tu código:
```bash
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```
#### En Ubuntu/Debian
```bash
sudo apt update
sudo apt install tesseract-ocr
```
#### En macOS (Homebrew)
```bash
brew install tesseract
```
## 🚀 Instrucciones para correr el proyecto
### 1. Clonar el repositorio
```bash
git clone https://github.com/Esteban-Fabian-Ramirez/Smartbool-Api.git
```
### 2. Crear y activar entorno virtual (opcional pero recomendado)
```bash
python -m venv venv
```
#### Activar:
##### En Linux/macOS
```bash
source venv/bin/activate
```
##### En Windows
```bash
venv\Scripts\activate
```
### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```
### 4. Coloca el modelo
Asegúrate de tener el archivo modelo_compuertas.keras en la raíz del proyecto.
## 🔌 Comandos para correr la API
### Ejecutar localmente:
```bash
python api_compuertas.py
```
Esto levantará el servidor en http://127.0.0.1:8000
### 📬 Endpoints disponibles
| Método | Ruta                   | Descripción                                         |
| ------ | ---------------------- | --------------------------------------------------- |
| POST   | `/predecir`            | Clasifica la compuerta lógica en la imagen          |
| POST   | `/analizar`            | Detecta si es expresión, tabla o diagrama y procesa |
| POST   | `/predecir_y_analizar` | Clasifica y genera tabla/expr/mapa                  |
| POST   | `/calcular_expresion`  | Calcula tabla y Karnaugh desde texto plano          |

### 🧪 Ejemplo con curl
```bash
curl -X POST http://127.0.0.1:8000/predecir \
  -F "file=@ruta/a/tu_imagen.png"
```
### 🧠 Ejemplo para /calcular_expresion
```bash
curl -X POST http://127.0.0.1:8000/calcular_expresion \
  -H "Content-Type: application/json" \
  -d '{"expresion": "(A and B) or not C"}'
```
