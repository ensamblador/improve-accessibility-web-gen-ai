import json
import boto3
import base64
import argparse


# crear un cliente bedrock-runtime para invocar el modelo 
bedrock = boto3.client("bedrock-runtime")
modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
accept = 'application/json'
contentType = 'application/json'


# obtener el archivo imagen desdeargumento  
parser = argparse.ArgumentParser()
parser.add_argument("image_path")
args = parser.parse_args()
image_path = args.image_path

# Lee el contenido del archivo imagen y lo codifica en base64 para enviarlo al modelo.
with open(image_path, "rb") as image_file:
    content_image = base64.b64encode(image_file.read()).decode('utf8')


# Crea un mensaje para el modelo, el rol es 'user' y tres elementos: texto previo, la imagen y texto posterior

previous_text = "...En la imagen se muestra el uso de LLM para entender las imagenes dentro del texto y proveer buqueda más relevante en Bases de Datos Vectoriales."
following_text = "Ahora veamos el caso donde un usuario quiera encontrar documentos donde se muestra la invocación a bedrock..."


message = {
    "role": "user",
    "content": [
        {"type":"text","text":previous_text},
        {
            "type": "image", 
            "source": {
                "type": "base64", 
                "media_type": "image/jpeg", 
                "data": content_image
            }
        },
        {"type":"text","text":following_text}
    ]
}



# La instrucción del modelo para analizar la imagen y obtener un texto alternativo. Ahora considerando contexto
system_prompt = "Tu eres un revisor de articulos web que van a ser publicados, tu misión es ver las imágenes y leer el texto previo y posterior para proporcionar un texto alternativo (que se incluirá como atributo 'alt' para la etiqueta img) que describa su contenido. Considera el texto que viene antes y después de la imagen. Responde en 150 caracteres o menos sin preámbulo"

# Invoca el modelo y obtiene el texto alternativo
body = {
    "system": system_prompt,
    "messages":[message],
    "anthropic_version":"bedrock-2023-05-31",
    "max_tokens":75, "temperature":0
}

response = bedrock.invoke_model(
    body=json.dumps(body), 
    modelId=modelId, accept=accept, contentType=contentType)

response_body = json.loads(response.get('body').read())
alt_text = response_body.get("content")[0].get("text")

# muestra el resultado
print({"image":image_path, "alt_text":alt_text})