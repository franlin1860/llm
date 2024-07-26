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
from typing import Literal

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

# best practice to use upper-case
entities = Literal[
    "PERSON",
    "TECHNOLOGY",
    "PROGRAMMING_LANGUAGE",
    "LOCATION",
    "ORGANIZATION",
    "PRODUCT",
    "EVENT"
]

relations = Literal[
    "WROTE",
    "USED",
    "LEARNED",
    "SUPPLIER_OF",
    "COMPETITOR",
    "PARTNERSHIP",
    "ACQUISITION",
    "WORKS_AT",
    "SUBSIDIARY",
    "BOARD_MEMBER",
    "CEO",
    "PROVIDES",
    "HAS_EVENT",
    "IN_LOCATION"
]

# define which entities can have which relations
validation_schema = {
    "PERSON": ["WROTE", "USED", "LEARNED", "WORKS_AT", "BOARD_MEMBER", "CEO", "HAS_EVENT"],
    "TECHNOLOGY": ["USED"],
    "PROGRAMMING_LANGUAGE": ["LEARNED"],
    "ORGANIZATION": [
        "SUPPLIER_OF",
        "COMPETITOR",
        "PARTNERSHIP",
        "ACQUISITION",
        "WORKS_AT",
        "SUBSIDIARY",
        "BOARD_MEMBER",
        "CEO",
        "PROVIDES",
        "HAS_EVENT",
        "IN_LOCATION",
    ],
    "PRODUCT": ["PROVIDES"],
    "EVENT": ["HAS_EVENT", "IN_LOCATION"],
    "LOCATION": ["HAPPENED_AT", "IN_LOCATION"],
}

kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    # if false, allows for values outside of the schema
    # useful for using the schema as a suggestion
    strict=True,
    max_triplets_per_chunk=50,  # 根据文档长度和复杂性调整
    num_workers=8,  # 增加并行处理的数量
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
    name="./SchemaGraph.html"
)

'''
dynamic_index.property_graph_store.get_triplets(
    entity_names=["Barack Obama", "Obama"]
)
'''
