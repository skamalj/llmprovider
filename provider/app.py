import dill
import json
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_ollama import ChatOllama
from nemoguardrails import RailsConfig, LLMRails


def get_model(provider: str, model_name: str, messages, base_url: str, **kwargs):
    """
    Creates and returns a pickled model from LangChain for OpenAI, Gemini, Anthropic, AWS Bedrock, Ollama, or Hugging Face.
    
    Args:
        provider (str): The LLM provider ("openai", "gemini", "anthropic", "bedrock", "ollama").
        model_name (str): The name of the model (e.g., "gpt-4", "gemini-pro", "claude-3", "amazon.titan-tg1", "mistral", "meta-llama").
        **kwargs: Additional parameters to pass into the model instantiation.
        
    Returns:
        Pickled model instance
    """
    config = RailsConfig.from_path("./config")
    
    if provider == "openai":
        model = ChatOpenAI(model_name=model_name, base_url=f"{base_url}/openai", **kwargs)
    elif provider == "google":
        model = ChatGoogleGenerativeAI(model=model_name, base_url=f"{base_url}/google",**kwargs)
    elif provider == "anthropic":
        model = ChatAnthropic(model=model_name, base_url=f"{base_url}/anthropic", **kwargs)
    elif provider == "bedrock":
        model = ChatBedrock(model_id=model_name, base_url=base_url, **kwargs)
    elif provider == "ollama":
        model = ChatOllama(model=model_name, base_url=base_url, **kwargs)
    else:
        raise ValueError("Unsupported provider")
    
    rails = LLMRails(config, llm=model)

    return rails.generate(messages=messages)

base_url = os.environ["BASE_APIGW_URL"] 
default_headers = {"x-api-key": os.environ["APIGW_KEY"]}

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        provider = body.get("provider")
        model_name = body.get("model_name")
        messages = body.get("messages")
        additional_params = body.get("params", {})
        
        if not provider or not model_name or not messages:
            raise ValueError("Both 'provider', 'model_name' and 'messages' are required.")
        
        response = get_model(provider, model_name,messages, base_url=base_url, default_headers=default_headers, **additional_params)
        
        return {
            "statusCode": 200,
            "body": json.dumps(response)
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }
