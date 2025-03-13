import os
import requests
import json
import boto3

# @! create a function to get secret from aws secret manager 

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

# @! create a function which accepts a dictionary of env variable and secret to load, and then sets sets env variable to secrets value using the function defined above

def load_secrets(env_secret_map):
    for env_var, secret_name in env_secret_map.items():
        secret_value = get_secret(secret_name)
        os.environ[env_var] = secret_value