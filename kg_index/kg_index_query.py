import os
import logging
import sys
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.core import load_index_from_storage

# 参考：https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphDemo/

# 配置日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# 定义通义千问2
llm = DashScope(temperature=0,
                model="qwen2-1.5b-instruct",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# 配置环境
Settings.llm = llm

# 设置嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 512

# restore from graph and vector storee
graph_store = SimpleGraphStore()
storage_context_kg = StorageContext.from_defaults(persist_dir='./data/storage_graph', graph_store=graph_store)
index = load_index_from_storage(
    storage_context=storage_context_kg,
    max_triplets_per_chunk=10,
    include_embeddings=True,
)

# 初始化查询引擎
query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)

response = query_engine.query(
    "Tell me more about Interleaf",
)
print("图谱查询: ", response, "/n/n")

# 不同查询方式
query_engine = index.as_query_engine(
    include_text=True, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf",
)
print("图谱查询结果: ", response, "/n/n")

# Query with embedding
# query using top 3 triplets plus keywords (duplicate triplets are removed)
query_engine = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)
response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf",
)
print("嵌入模式查询知识图谱: ", response, "/n/n")

# create graph
from pyvis.network import Network

# 获取 NetworkX 图
g = index.get_networkx_graph()

# 创建 PyVis 网络图对象
net = Network(notebook=False, directed=True)

# 将 NetworkX 图转换为 PyVis 图
net.from_nx(g)
net.show('kg_index_graph.html', notebook=False)
