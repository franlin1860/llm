import os
import logging
import sys
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope

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

# 使用SimpleDirectoryReader读取TXT文件
documents = SimpleDirectoryReader(input_files=['./data/paul_graham/paul_graham_essay.txt']).load_data()

# 初始化存储上下文
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# 创建KnowledgeGraphIndex
index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=10,
    storage_context=storage_context,
    include_embeddings=True,
)

# persist KG Index(Only MetaData will be persisted, KG is in NEO4J)
index.storage_context.persist(persist_dir='./data/storage_graph')

# 初始化查询引擎
query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)

response = query_engine.query(
    "Tell me more about Interleaf",
)
print("图谱查询: ", response)

# 不同查询方式
query_engine = index.as_query_engine(
    include_text=True, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf",
)
print("嵌入模式图谱查询结果: ", response)
