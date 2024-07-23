import os
import logging
import sys
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
import pandas as pd
from llama_index.core import Document
from typing import Literal
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# 参考：https://docs.llamaindex.ai/en/stable/examples/property_graph/property_graph_neo4j/

# 配置日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# 定义通义千问2 qwen2-1.5b-instruct
llm = DashScope(temperature=0,
                model="qwen2-72b-instruct",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# 配置环境
Settings.llm = llm

# 设置嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 512

# 读取Neo4j配置文件
config = {}
with open("../neo4j_config.txt", "r") as config_file:
    for line in config_file:
        if line.strip() and not line.startswith("#"):
            key, value = line.strip().split("=")
            config[key] = value

neo4j_url = config.get("NEO4J_URI")
neo4j_username = config.get("NEO4J_USERNAME")
neo4j_password = config.get("NEO4J_PASSWORD")

# 初始化Neo4j图存储 PropertyGraph not Graph
graph_store = Neo4jPropertyGraphStore(
    username=neo4j_username,
    password=neo4j_password,
    url=neo4j_url
)

# 使用SimpleDirectoryReader读取TXT文件
documents = SimpleDirectoryReader(input_files=['../data/paul_graham/paul_graham_essay.txt']).load_data()

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=embed_model,
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=llm
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

# Querying and Retrieval
retriever = index.as_retriever(
    include_text=False,  # include source text in returned nodes, default True
)

nodes = retriever.retrieve("What happened at Interleaf and Viaweb?")

for node in nodes:
    print(node.text)

query_engine = index.as_query_engine(include_text=True)

response = query_engine.query("What happened at Interleaf and Viaweb?")

print(str(response))
