import os
import sys
import logging
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import SimpleDirectoryReader
from llama_index.core import PropertyGraphIndex
from llama_index.core import StorageContext, load_index_from_storage

# eg. llama_index/docs/docs/examples/property_graph/property_graph_basic.ipynb

# 定义DeepSpeed model, is_chat_model needed
llm = OpenAILike(model="deepseek-chat",
                 api_base="https://api.deepseek.com/v1",
                 api_key=os.getenv("DEEPSEEK_API_KEY"),
                 is_chat_model=True,
                 temperatue=0.1)

# 配置环境
Settings.llm = llm

# 设置嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.embed_model = embed_model

index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="./storage")
)

'''
retriever = index.as_retriever(
    include_text=False,  # include source text, default True
)

nodes = retriever.retrieve("On which machine did Paul Graham first try writing programs?")

for node in nodes:
    print(node.text)

'''

query_engine = index.as_query_engine(
    include_text=True,
)

response = query_engine.query("On which machine did Paul Graham first try writing programs?")

print("\nQery result: ", response, "\n\n")
