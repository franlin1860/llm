import os
import logging
import sys
from llama_index.core import Settings
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.core import StorageContext, KnowledgeGraphIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.core import SimpleDirectoryReader

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

# 读取已解析的文档
reader = SimpleDirectoryReader(
    input_files=["./data/parsed_documents.txt"]
)

# 将读取的文本内容转换为文档对象
documents = reader.load_data()

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

# 清理Neo4j数据库中的旧数据
graph_store.query(
    """
    MATCH (n) DETACH DELETE n
    """
)

# 构建知识图谱索引
index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    include_embeddings=True,
    storage_context=storage_context,
)
