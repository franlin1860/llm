import os
from llama_index.llms.dashscope import DashScope
from llama_index.core import Settings
from llama_index.readers.file import PDFReader
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging
import sys

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


# PDF Reader with `SimpleDirectoryReader`
parser = PDFReader()
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    "./pdf_data", file_extractor=file_extractor
).load_data()

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# 构建知识图谱索引
index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
)

# Querying the Knowledge Graph
query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)

response = query_engine.query("the ethical considerations and regulatory implications of using LLMs in finance")
print(response)

# NOTE: can take a while!
new_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    include_embeddings=True,
)

# query using top 3 triplets plus keywords (duplicate triplets are removed)
query_engine = new_index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)
response = query_engine.query(
    "the ethical considerations and regulatory implications of using LLMs in finance",
)
print(response)
