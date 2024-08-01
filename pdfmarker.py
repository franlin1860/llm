import os
import logging
import sys
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.readers.pdf_marker import PDFMarkerReader
from pathlib import Path

# 配置日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# 定义DeepSpeed model
llm = OpenAILike(model="deepseek-chat",
                 api_base="https://api.deepseek.com/v1",
                 api_key=os.getenv("DEEPSEEK_API_KEY"),
                 temperature=0.6,
                 is_chat_model=True)
# 配置环境
Settings.llm = llm

# 设置嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 768

# PDF Reader with `pdf maker`
path = Path("./pdf_data/mjzf_1.pdf")
if not path.exists():
    print("文件不存在，程序即将退出")
    sys.exit()
reader = PDFMarkerReader()
documents = reader.load_data(file=path, max_pages=1)

index = VectorStoreIndex.from_documents(
    documents,
)

index.storage_context.persist(persist_dir="<persist_pdf_dir>")

# 配置检索器
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
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
response = query_engine.query("互联网药店直付用药流程?")
print(response, "\n")

response = query_engine.query("治疗期间发生的必需且合理的医疗费用包括什么？?")
print(response, "\n")

response = query_engine.query("门急诊费用医疗保险C款的保险费约定支付日?")
print(response, "\n")
