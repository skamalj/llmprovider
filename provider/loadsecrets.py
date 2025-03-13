import os
import requests
import json

def get_secret(secret_name):
    
    headers = {"X-Aws-Parameters-Secrets-Token": os.environ.get('AWS_SESSION_TOKEN')}

    secrets_extension_http_port = os.environ["PARAMETERS_SECRETS_EXTENSION_HTTP_PORT"]
    secrets_extension_endpoint = f"http://localhost:{secrets_extension_http_port}/secretsmanager/get?secretId={secret_name}"
  
    r = requests.get(secrets_extension_endpoint, headers=headers)
  
    secret = json.loads(r.text)["SecretString"]
    return secret

# @! create a function which accepts a dictionary of env variable and secret to load, and then sets sets env variable to secrets value using the function defined above

def load_secrets(env_secret_map):
    for env_var, secret_name in env_secret_map.items():
        secret_value = get_secret(secret_name)
        os.environ[env_var] = secret_value