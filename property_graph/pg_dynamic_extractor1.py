import os
import sys
import logging
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import SimpleDirectoryReader
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor,
    SchemaLLMPathExtractor,
    DynamicLLMPathExtractor,
)
# eg. llama_index/docs/docs/examples/property_graph/Dynamic_KG_Extraction.ipynb

# 定义DeepSpeed model, is_chat_model needed
llm = OpenAILike(model="deepseek-chat",
                 api_base="https://api.deepseek.com/v1",
                 api_key=os.getenv("DEEPSEEK_API_KEY"),
                 is_chat_model=True,
                 temperatue=0.1)

# 配置环境
Settings.llm = llm

# 设置嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

kg_extractor = DynamicLLMPathExtractor(
    llm=llm,
    max_triplets_per_chunk=50,  # 根据文档长度和复杂性调整
    num_workers=8,  # 增加并行处理的数量
    allowed_entity_types=["PERSON", "TECHNOLOGY", "PROGRAMMING_LANGUAGE"],
    allowed_relation_types=["WROTE", "USED", "LEARNED", "PRESIDENT_OF", "MEMBER_OF"],
    allowed_relation_props=["description", "time", "location", "technology"],
    allowed_entity_props=["description", "age", "occupation", "education"],
)

index = PropertyGraphIndex.from_documents(
    documents,
    llm=llm,
    embed_kg_nodes=False,
    kg_extractors=[kg_extractor],
    show_progress=True,
)

'''
retriever = index.as_retriever(
    include_text=False,  # include source text, default True
)

nodes = retriever.retrieve("On which machine did Paul Graham first try writing programs?")

for node in nodes:
    print(node.text)

'''

query_engine = index.as_query_engine(
    include_text=True,
)

response = query_engine.query("On which machine did Paul Graham first try writing programs?")

print("\nQery result: ", response, "\n\n")

index.storage_context.persist(persist_dir="./storage_dynamic")

index.property_graph_store.save_networkx_graph(
    name="./DynamicGraph1.html"
)

'''
dynamic_index.property_graph_store.get_triplets(
    entity_names=["Barack Obama", "Obama"]
)
'''
