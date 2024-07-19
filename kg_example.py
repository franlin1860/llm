import os
import logging
import sys
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope

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
Settings.chunk_size = 512

# 使用SimpleDirectoryReader读取TXT文件
documents = SimpleDirectoryReader(input_files=['./data/paul_graham/paul_graham_essay.txt']).load_data()

# 初始化存储上下文
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# 创建KnowledgeGraphIndex
index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
    include_embeddings=True,
)

# 初始化查询引擎
query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)

# 查询知识图谱
query = "显示所有实体和关系"
response = query_engine.query(query)
print(response)

response = query_engine.query(
    "Tell me more about Interleaf",
)

print(response)
