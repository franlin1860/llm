import os
import sys
import logging
from llama_index.core import Settings
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage

# 配置日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# 定义通义千问2
llm = DashScope(temperature=0.2,
                model="qwen2-1.5b-instruct",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# 配置环境
Settings.llm = llm

# 设置嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.embed_model = embed_model

# 读取Neo4j配置文件
config = {}
with open("neo4j_config.txt", "r") as config_file:
    for line in config_file:
        if line.strip() and not line.startswith("#"):
            key, value = line.strip().split("=")
            config[key] = value

neo4j_url = config.get("NEO4J_URI")
neo4j_username = config.get("NEO4J_USERNAME")
neo4j_password = config.get("NEO4J_PASSWORD")

# 初始化Neo4j图存储
graph_store = Neo4jGraphStore(
    username=neo4j_username,
    password=neo4j_password,
    url=neo4j_url
)

# 创建存储上下文
storage_context = StorageContext.from_defaults(graph_store=graph_store)

graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(
    graph_rag_retriever,
)


response = query_engine.query(
    "Tell me about Peter Quill."
)
print(response)

# restore from graph and vector storee
storage_context_kg = StorageContext.from_defaults(persist_dir='./storage_graph', graph_store=graph_store)
kg_index = load_index_from_storage(
    storage_context=storage_context_kg,
    max_triplets_per_chunk=10,
    include_embeddings=True,
)

storage_context_vector = StorageContext.from_defaults(persist_dir='./storage_vector')
vector_index = load_index_from_storage(
    storage_context=storage_context_vector
)

kg_rag_query_engine = kg_index.as_query_engine(
    include_text=False,
    retriever_mode="keyword",
    response_mode="tree_summarize",
)

vector_rag_query_engine = vector_index.as_query_engine()

response_graph_rag = kg_rag_query_engine.query("Tell me about Peter Quill.")
print("KG query result: ", response_graph_rag)

response_vector_rag = vector_rag_query_engine.query("Tell me about Peter Quill.")
print("Vector query result: ", response_vector_rag)
