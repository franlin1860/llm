# pg_schema_extractor.py

import os
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import SimpleDirectoryReader
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from typing import List, Dict
from typing import Literal


class PGSchemaExtractor:
    def __init__(self):
        self.llm = OpenAILike(
            model="deepseek-chat",
            api_base="https://api.deepseek.com/v1",
            api_key="sk-4944cc4af34148f6a53f74244a48bf8b",
            is_chat_model=True,
            temperature=0.1
        )

        Settings.llm = self.llm

        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
        Settings.embed_model = self.embed_model

        self.schema_schema = {}
        self.schema_attributes = []
        self.schema_relation_extraction_ops = []
        self.kg_extractor = None
        self.schema_type = ""

    def set_schema(self, schema_type):
        print("In set_schema :\n")

        self.schema_type = schema_type

    def create_kg_extractor(self):

        schema_type = self.schema_type

        # 打印调试信息
        print("In create_kg_extractor\n")

        # 更新或创建 kg_extractor
        if self.kg_extractor is None:
            print("kg_extractor is None\n")
            if schema_type == 'schema1':
                self.kg_extractor = SchemaLLMPathExtractor(
                    llm=self.llm,
                    strict=False,  # Set to False to showcase why it's not going to be the same as DynamicLLMPathExtractor
                    possible_entities=None,  # USE DEFAULT ENTITIES (PERSON, ORGANIZATION... etc)
                    possible_relations=None,  # USE DEFAULT RELATIONSHIPS
                    possible_entity_props=entities1,
                    possible_relation_props=relations1,
                    kg_validation_schema=validation_schema1,
                    max_triplets_per_chunk=50,
                )
            else:
                self.kg_extractor = SchemaLLMPathExtractor(
                    llm=self.llm,
                    strict=False,  # Set to False to showcase why it's not going to be the same as DynamicLLMPathExtractor
                    possible_entities=None,  # USE DEFAULT ENTITIES (PERSON, ORGANIZATION... etc)
                    possible_relations=None,  # USE DEFAULT RELATIONSHIPS
                    possible_entity_props=entities2,
                    possible_relation_props=relations2,
                    kg_validation_schema=validation_schema2,
                    max_triplets_per_chunk=50,
                )
        else:
            print("kg_extractor is renewed\n")
            if schema_type == 'schema1':
                self.kg_extractor.kg_validation_schema = validation_schema1
                self.kg_extractor.possible_entity_props = entities1
                self.kg_extractor.possible_relation_props = relations1
            else:
                self.kg_extractor.kg_validation_schema = validation_schema2
                self.kg_extractor.possible_entity_props = entities2
                self.kg_extractor.possible_relation_props = relations2

        # Debug info
        print("After reset in create_kg_extractor\n")
        print(f"Schema:  {self.kg_extractor.kg_validation_schema}\n")
        print(f"Entities: {self.kg_extractor.possible_entity_props}\n")
        print(f"Relations: {self.kg_extractor.possible_relation_props}\n")

    def extract(self, data_dir):
        documents = SimpleDirectoryReader(data_dir).load_data()

        self.create_kg_extractor()

        index = PropertyGraphIndex.from_documents(
            documents,
            llm=self.llm,
            embed_kg_nodes=False,
            kg_extractors=[self.kg_extractor],
            show_progress=True,
        )

        return index

    def query(self, index, query_text):
        query_engine = index.as_query_engine(include_text=True)
        response = query_engine.query(query_text)
        return response


# 定义两个不同的schema
entities1 = [
    "PERSON", "TECHNOLOGY", "PROGRAMMING_LANGUAGE", "LOCATION",
    "ORGANIZATION", "PRODUCT", "EVENT"
    ]

relations1 = [
    "WROTE", "USED", "LEARNED", "SUPPLIER_OF", "COMPETITOR",
    "PARTNERSHIP", "ACQUISITION", "WORKS_AT", "SUBSIDIARY",
    "BOARD_MEMBER", "CEO", "PROVIDES", "HAS_EVENT", "IN_LOCATION"
    ]

validation_schema1 = {
    "PERSON": ["WROTE", "USED", "LEARNED", "WORKS_AT", "BOARD_MEMBER", "CEO", "HAS_EVENT"],
    "TECHNOLOGY": ["USED"],
    "PROGRAMMING_LANGUAGE": ["LEARNED"],
    "ORGANIZATION": [
        "SUPPLIER_OF", "COMPETITOR", "PARTNERSHIP", "ACQUISITION",
        "WORKS_AT", "SUBSIDIARY", "BOARD_MEMBER", "CEO", "PROVIDES",
        "HAS_EVENT", "IN_LOCATION",
    ],
    "PRODUCT": ["PROVIDES"],
    "EVENT": ["HAS_EVENT", "IN_LOCATION"],
    "LOCATION": ["IN_LOCATION"],
}

entities2 = [
    "PERSON",
    "COMPUTER",
    "PROGRAMMING_LANGUAGE",
    "LOCATION",
    "ORGANIZATION",
    "TIME_PERIOD",
    "CONCEPT",
    "SKILL"
]

relations2 = [
    "WORKED_ON",
    "USED",
    "LEARNED",
    "LOCATED_IN",
    "STUDIED_AT",
    "DEVELOPED",
    "EXPERIENCED",
    "HAD_DIFFICULTY_WITH",
    "INVOLVED_IN",
    "DURING"
]

validation_schema2 = {
    "PERSON": ["WORKED_ON", "USED", "LEARNED", "LOCATED_IN", "STUDIED_AT", "DEVELOPED", "EXPERIENCED", "HAD_DIFFICULTY_WITH"],
    "COMPUTER": ["USED", "LOCATED_IN"],
    "PROGRAMMING_LANGUAGE": ["USED", "LEARNED", "DEVELOPED"],
    "LOCATION": ["LOCATED_IN"],
    "ORGANIZATION": ["LOCATED_IN", "DEVELOPED"],
    "TIME_PERIOD": ["DURING"],
    "CONCEPT": ["LEARNED", "DEVELOPED", "INVOLVED_IN"],
    "SKILL": ["LEARNED", "DEVELOPED", "EXPERIENCED"]
}
