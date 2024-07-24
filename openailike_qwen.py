import os
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(model="qwen2-1.5b-instruct",
                 api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 api_key=os.getenv("DASHSCOPE_API_KEY"),
                 is_chat_model=True)

response = llm.complete("Hello World!")
print(str(response))
