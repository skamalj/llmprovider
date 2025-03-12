from langchain.schema import HumanMessage, AIMessage
from nemoguardrails import RailsConfig, LLMRails
from langchain_openai import ChatOpenAI

import json
config = RailsConfig.from_path("./config")
messages = [
    HumanMessage(content="How many players play cricket test match per side")
]
jsonm =  [{"content": "Who all starred in avengers 2 movie", "type": "human"}]
# Convert to JSON
json_payload = json.dumps([msg.dict() for msg in messages])
model = ChatOpenAI(model="gpt-4o", base_url="", 
                   default_headers={"x-api-key":""})
rails = LLMRails(config, llm=model)
print(json_payload)
print(rails.generate(json.dumps(jsonm)))
