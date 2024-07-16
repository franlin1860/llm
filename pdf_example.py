import os
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.core import Settings
from llama_index.readers.file import PDFReader
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# 定义通义千问2 大模型 72B
llm = DashScope(temperature=0.2,
                model="qwen2-72b-instruct",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# 配置环境
Settings.llm = llm

# 配置向量模型
Settings.embed_model = DashScopeEmbedding()


# PDF Reader with `SimpleDirectoryReader`
parser = PDFReader()
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    "./pdf_data", file_extractor=file_extractor
).load_data()

index = VectorStoreIndex.from_documents(
    documents,
)

index.storage_context.persist(persist_dir="<persist_pdf_dir>")

# 配置检索器
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# 配置响应合成器
response_synthesizer = get_response_synthesizer()

# 组装查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

# 查询
response = query_engine.query("What are the ethical considerations and regulatory implications of using LLMs in finance, particularly regarding legal responsibility and safety?")
print(response)
