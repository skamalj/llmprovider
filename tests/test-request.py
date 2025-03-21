# @! create code to send request to api endpoint at API_ENDPOINT env variable with api key set in header x-api-key, a post request with body as json

import requests
import json
import os
from langchain.schema import HumanMessage, AIMessage

API_ENDPOINT = os.getenv('API_ENDPOINT')
API_KEY = os.getenv('API_GW_KEY')

headers = {'x-api-key': API_KEY}
print(headers)
body = {
    "provider": "openai",    
    "model_name": "gpt-4o",  
    "params": {              
        "temperature": 0.7,
        "max_tokens": 100
    },
     "messages": [{"content": "talk to me about avengers movie", "role": "human"}]
}

response = requests.post(API_ENDPOINT, headers=headers, json=body)
print(response.headers.get("Content-Type"))
print(AIMessage(**(response.json())))