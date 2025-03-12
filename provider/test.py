from langchain.schema import HumanMessage, AIMessage
from nemoguardrails import RailsConfig, LLMRails
from langchain_openai import ChatOpenAI

import json
config = RailsConfig.from_path("./config")
messages = [
    HumanMessage(content="Who all starred in avengers 2 movie")
]

# Convert to JSON
json_payload = json.dumps([msg.dict() for msg in messages])
model = ChatOpenAI(model="gpt-4o")
rails = LLMRails(config, llm=model)
print(json_payload)
#print(rails.generate(json_payload))
