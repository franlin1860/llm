import os
import sys
import logging
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.llms.openai_like import OpenAILike
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import LLMSynonymRetriever, VectorContextRetriever

# 定义DeepSpeed model, is_chat_model needed
llm = OpenAILike(model="deepseek-chat",
                 api_base="https://api.deepseek.com/v1",
                 api_key=os.getenv("DEEPSEEK_API_KEY"),
                 is_chat_model=True)

# 配置环境
Settings.llm = llm

# 设置嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.embed_model = embed_model

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

# 创建存储上下文
storage_context = StorageContext.from_defaults(graph_store=graph_store)

llm_synonym = LLMSynonymRetriever(
    graph_store,
    include_text=False,
)

vector_context = VectorContextRetriever(
    graph_store,
    include_text=False,
)

query_engine = RetrieverQueryEngine.from_args(
    llm_synonym,
)

response = query_engine.query(
    "What happened at Interleaf and Viaweb?"
)

print("\nKG query reslut:", response, "\n\n")

vector_query_engine = RetrieverQueryEngine.from_args(
        vector_context,
        )

response = vector_query_engine.query(
        "What happened at Interleaf and Viaweb?"
        )

print("\nVector query result:", response, "\n\n")
