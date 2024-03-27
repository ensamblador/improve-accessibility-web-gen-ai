import json
import boto3


# crear un cliente bedrock-runtime para invocar el modelo 
bedrock = boto3.client("bedrock-runtime")
modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
accept = 'application/json'
contentType = 'application/json'

# Crea un mensaje para el modelo, el rol es 'user' y solo un contenido: la imagen
message = {
    "role": "user",
    "content": [{"type":"text","text":"cual es la capital de Chile?. Responde solo con el nombre."}]
}

# Invoca el modelo y obtiene el texto alternativo
body = {
    "messages":[message],
    "anthropic_version":"bedrock-2023-05-31",
    "max_tokens":100, "temperature":0
}

response = bedrock.invoke_model(
    body=json.dumps(body), 
    modelId=modelId, accept=accept, contentType=contentType)

response_body = json.loads(response.get('body').read())
# muestra el resultado
print(response_body.get("content")[0].get("text"))