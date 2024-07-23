import os
import logging
import sys
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
import pandas as pd
from llama_index.core import Document
from typing import Literal
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

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

# 读取Neo4j配置文件
config = {}
with open("../neo4j_config.txt", "r") as config_file:
    for line in config_file:
        if line.strip() and not line.startswith("#"):
            key, value = line.strip().split("=")
            config[key] = value

neo4j_url = config.get("NEO4J_URI")
neo4j_username = config.get("NEO4J_USERNAME")
neo4j_password = config.get("NEO4J_PASSWORD")

# 初始化Neo4j图存储
# 注意 是 PropertyGraphStore not GraphStore
graph_store = Neo4jPropertyGraphStore(
    username=neo4j_username,
    password=neo4j_password,
    url=neo4j_url
)

news = pd.read_csv("./data/news_articles.csv")
documents = [Document(text=f"{row['title']}: {row['text']}") for i, row in news.iterrows()]
news.head()

# best practice to use upper-case
entities = Literal["PERSON", "LOCATION", "ORGANIZATION", "PRODUCT", "EVENT"]
relations = Literal[
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
]

# define which entities can have which relations
validation_schema = {
    "Person": ["WORKS_AT", "BOARD_MEMBER", "CEO", "HAS_EVENT"],
    "Organization": [
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
    "Product": ["PROVIDES"],
    "Event": ["HAS_EVENT", "IN_LOCATION"],
    "Location": ["HAPPENED_AT", "IN_LOCATION"],
}

kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    # if false, allows for values outside of the schema
    # useful for using the schema as a suggestion
    strict=True,
)

NUMBER_OF_ARTICLES = 250

index = PropertyGraphIndex.from_documents(
    documents[:NUMBER_OF_ARTICLES],
    kg_extractors=[kg_extractor],
    llm=llm,
    embed_model=embed_model,
    property_graph_store=graph_store,
    show_progress=True,
)

query_engine = index.as_query_engine(include_text=True)
response = query_engine.query("What do you know about Maliek Collins or Darragh O’Brien?")

print(str(response))
