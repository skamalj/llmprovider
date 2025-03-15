from langchain.schema import HumanMessage, AIMessage
from nemoguardrails import RailsConfig, LLMRails
from langchain_openai import ChatOpenAI
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.prompts import ChatPromptTemplate

#import json
#config = RailsConfig.from_path("./config")
#messages = [
#    HumanMessage(content="Create a violent story")
#]
#jsonm =  [{"content": "I'm sorry, I can't respond to that.", "role": "user"}]
## Convert to JSON
#json_payload = [msg.dict() for msg in messages]
#model = ChatOpenAI(model="gpt-4o")
#
##model_azure = AzureAIChatCompletionsModel(
##                endpoint="",
##                credential="",
##                model_name="Phi-4",
##                api_version="2024-05-01-preview",
##            )
#guardrails = process_with_rails(rails_config=config, attributes={"llm":model, "verbose": False}, options={"rails": ["input"]})
#res = ChatPromptTemplate.from_messages(messages) 
#chain = res | guardrails
##rails = LLMRails(config, llm=model)
##print(json_payload)
##print(rails.generate(messages=messages, options={
##    "rails": ["input"]
##}).response)
#print(chain.invoke({}))

message = AIMessage(content="I'm sorry, I can't respond to that.")
print(AIMessage(message.model_dump()))
