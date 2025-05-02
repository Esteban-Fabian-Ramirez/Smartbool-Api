from huggingface_hub import HfApi
import os

HF_TOKEN = "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"

file_path = "models/modelo.keras"  # Archivo que quieres subir
repo_id = "Estebanxdd/smartbool"
path_in_repo = "modelo.keras"  # Cómo se llamará en Hugging Face

api = HfApi(token=HF_TOKEN)
api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="model"
)
