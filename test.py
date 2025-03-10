import json
import pickle
import os
import requests

# Load API Gateway URL & API Key from environment variables
BASE_APIGW_URL = os.environ["BASE_APIGW_URL"]

# Step 1: Request the model from API Gateway
model_request_payload = {
    "provider": "openai",    
    "model_name": "gpt-4o",  
    "params": {              
        "temperature": 0.7,
        "max_tokens": 100
    }
}

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "*/*",  # Explicitly request JSON response
    "User-Agent": "Application",
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': '*',
    'Access-Control-Allow-Methods': 'POST'
}

print(f"{BASE_APIGW_URL}/model")
response = requests.get(f"{BASE_APIGW_URL}/model", headers=HEADERS, json=model_request_payload)
print(response)
response_data = response.json()

if "model_pickle" not in response_data:
    print("Error fetching model:", response_data.get("error"))
    exit(1)

# Step 2: Deserialize the model
model_pickle = response_data["model_pickle"]
model = pickle.loads(bytes.fromhex(model_pickle))

# Step 3: Use the model to generate a response
prompt = "What is the capital of France?"
model_response = model.invoke(prompt)

# Print the final result
print("Model Response:", model_response)
