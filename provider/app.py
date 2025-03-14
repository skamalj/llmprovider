import dill
import json
import os
import traceback
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrockConverse
from langchain_ollama import ChatOllama
from nemoguardrails import RailsConfig, LLMRails
from loadsecrets import load_secrets

def get_model(provider: str, model_name: str, messages, **kwargs):
    """
    Creates and returns a pickled model from LangChain for OpenAI, Gemini, Anthropic, AWS Bedrock, Ollama, or Hugging Face.
    
    Args:
        provider (str): The LLM provider ("openai", "gemini", "anthropic", "bedrock", "ollama").
        model_name (str): The name of the model (e.g., "gpt-4", "gemini-pro", "claude-3", "amazon.titan-tg1", "mistral", "meta-llama").
        **kwargs: Additional parameters to pass into the model instantiation.
        
    Returns:
        Pickled model instance
    """
    try:
        config = RailsConfig.from_path("./config")

        if provider == "openai":
            model = ChatOpenAI(model_name=model_name, **kwargs)
        elif provider == "google":
            model = ChatGoogleGenerativeAI(model=model_name, **kwargs)
        elif provider == "anthropic":
            model = ChatAnthropic(model=model_name,**kwargs)
        elif provider == "bedrock":
            model = ChatBedrockConverse(model_id=model_name, **kwargs)
        elif provider == "ollama":
            model = ChatOllama(model=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        rails = LLMRails(config, llm=model)
        print(messages)


        return rails.generate(json.dumps(messages))

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in get_model: {error_details}")  # Log stack trace
        raise RuntimeError(f"Failed to generate response: {str(e)}") from e

env_secret_map = {
    "ANTHROPIC_API_KEY": "AnthropicAPIKey",
    "OPENAI_API_KEY":"OpenAIAPIKey",
    "GOOGLE_API_KEY": "GeminiAPIKey"
}
load_secrets(env_secret_map)

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        provider = body.get("provider")
        model_name = body.get("model_name")
        messages = body.get("messages")
        additional_params = body.get("params", {})

        if not provider or not model_name or not messages:
            raise ValueError("Both 'provider', 'model_name', and 'messages' are required.")

        response = get_model(provider, model_name, messages, **additional_params)

        return {
            "statusCode": 200,
            "body": json.dumps(response)
        }
    
    except json.JSONDecodeError:
        error_msg = "Invalid JSON in request body."
        print(f"JSONDecodeError: {error_msg}\nEvent Body: {event.get('body')}")
        return {"statusCode": 400, "body": json.dumps({"error": error_msg})}

    except KeyError as e:
        error_msg = f"Missing environment variable: {str(e)}"
        print(f"KeyError: {error_msg}")
        return {"statusCode": 500, "body": json.dumps({"error": error_msg})}

    except ValueError as e:
        error_msg = f"ValueError: {str(e)}"
        print(error_msg)
        return {"statusCode": 400, "body": json.dumps({"error": error_msg})}

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Unhandled Exception: {error_details}")  # Log full stack trace
        return {"statusCode": 500, "body": json.dumps({"error": "Internal Server Error. Check logs for details."})}
