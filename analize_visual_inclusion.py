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
previous_text = "En la imagen de abajo se ve las diferentes alarmas del sistema"
following_text = "Podemos ver que existen dos alarmas posibles."

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
system_prompt = """Tu eres un revisor de articulos web que van a ser publicados, tu misión es ver las imágenes y leer el texto para encontrar contenido web no inclusivo para personas con discapacidad visual. A continuación te muestro algunos ejemplos:

Ejemplo de escritura no inclusiva:
Incorrecto:'Como se puede ver en la imagen de arriba, el proceso inicia...' 
Corrección: 'En el diagrama anterior, el proceso inicia...'

Ejemplo de imagen no inclusiva:
Incorrecto: (usando una imagen con colores rojo y verde) 'El color rojo representa un sistema alarmado, y el verde el sistema saludable'
Correcto: (usando una imagen con colores rojo y amarillo y etiquetas de texto para cada color) 'Acá se muestra el sistema alarmado y el sistema saludable'
"""

# Invoca el modelo y obtiene el texto alternativo
body = {
    "system": system_prompt,
    "messages":[message],
    "anthropic_version":"bedrock-2023-05-31",
    "max_tokens":500, "temperature":0
}

response = bedrock.invoke_model(
    body=json.dumps(body), 
    modelId=modelId, accept=accept, contentType=contentType)

response_body = json.loads(response.get('body').read())
recomendacion = response_body.get("content")[0].get("text")

# muestra el resultado
print({"image":image_path, "recomendacion":recomendacion})