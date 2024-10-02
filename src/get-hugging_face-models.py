# construir uma tool para listar os modelos do hugging face
import os
from typing import Optional, Dict
from langchain.tools import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field

import requests
import json
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

API_TOKEN = os.getenv("API_TOKEN")

class RetornaModelosHuggingFaceArgs(BaseModel):
    path: Optional[str] = Field(None, description="Caminho opcional da API para acessar modelos especificos")
    query_params: Optional[Dict[str, str]] = Field(None, description="Parâmetros de busca opcionais como 'search', 'author' etc")
    
def get_hugging_face_models(path: Optional[str] = None, query_params: Optional[Dict[str, str]] = None) -> dict:
    base_url = "https://huggingface.co/api/models"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.get(base_url + (path or ""), params=query_params, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Erro na requisicao: {response.status_code} - {response.text}")
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        raise ValueError(f"A resposta da API não contém um JSON válido")
        

get_huggingface_models_tool = StructuredTool.from_function(
    func=get_hugging_face_models,
    name="Tool Retornar modelos do Hugging Face",
    description="Retorna modelos do Hugging Face",
    args_schema=RetornaModelosHuggingFaceArgs,
    return_direct=True,
)

modelos = get_huggingface_models_tool.run({
    "query_params": {
        "search": "gpt-j"
    }
})

output_file = "./data/modelos.json"
with open(output_file, "w") as file:
    json.dump(modelos, file, indent=4)

print(json.dumps(modelos, indent=4))
print('Modelos salvos')
print(len(modelos))
