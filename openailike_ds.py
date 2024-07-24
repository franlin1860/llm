import os
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(model="deepseek-chat",
                 api_base="https://api.deepseek.com/v1",
                 api_key=os.getenv("DEEPSEEK_API_KEY"),
                 is_chat_model=True)

response = llm.complete("Hello World!")
print(str(response))
