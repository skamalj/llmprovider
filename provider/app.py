import pickle
import json
import os
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_aws import BedrockLLM
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

def get_model(provider: str, model_name: str, base_url: str, **kwargs):
    """
    Creates and returns a pickled model from LangChain for OpenAI, Gemini, Anthropic, AWS Bedrock, Ollama, or Hugging Face.
    
    Args:
        provider (str): The LLM provider ("openai", "gemini", "anthropic", "bedrock", "ollama", "huggingface").
        model_name (str): The name of the model (e.g., "gpt-4", "gemini-pro", "claude-3", "amazon.titan-tg1", "mistral", "meta-llama").
        **kwargs: Additional parameters to pass into the model instantiation.
        
    Returns:
        Pickled model instance
    """
    if provider == "openai":
        model = ChatOpenAI(model_name=model_name, base_url=base_url, **kwargs)
    elif provider == "google":
        model = ChatGoogleGenerativeAI(model=model_name, base_url=base_url,**kwargs)
    elif provider == "anthropic":
        model = ChatAnthropic(model=model_name, base_url=base_url, **kwargs)
    elif provider == "bedrock":
        model = BedrockLLM(model_id=model_name, base_url=base_url, **kwargs)
    elif provider == "ollama":
        model = ChatOllama(model=model_name, base_url=base_url, **kwargs)
    elif provider == "huggingface":
        endpoint = HuggingFaceEndpoint(repo_id=model_name, base_url=base_url, **kwargs)
        model = ChatHuggingFace(endpoint=endpoint)
    else:
        raise ValueError("Unsupported provider")
    
    return pickle.dumps(model)

base_url = os.environ["BASE_APIGW_URL"] 
default_headers = {"x-api-key": os.environ["APIGW_KEY"]}

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        provider = body.get("provider")
        model_name = body.get("model_name")
        additional_params = body.get("params", {})
        
        if not provider or not model_name:
            raise ValueError("Both 'provider' and 'model_name' are required.")
        
        model_pickle = get_model(provider, model_name,base_url=base_url, default_headers=default_headers, **additional_params)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Model created successfully",
                "model_pickle": model_pickle.hex()
            }),
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }
