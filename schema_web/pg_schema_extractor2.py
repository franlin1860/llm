import os
import json
import sys
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import SimpleDirectoryReader
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from typing import Literal

# 定义两个不同的schema
entities1 = Literal[
    "PERSON", "TECHNOLOGY", "PROGRAMMING_LANGUAGE", "LOCATION",
    "ORGANIZATION", "PRODUCT", "EVENT"
]

relations1 = Literal[
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

entities2 = Literal[
    "PERSON",
    "COMPUTER",
    "PROGRAMMING_LANGUAGE",
    "LOCATION",
    "ORGANIZATION",
    "TIME_PERIOD",
    "CONCEPT",
    "SKILL"
]

relations2 = Literal[
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

class PGSchemaExtractor:
    def __init__(self, api_key):
        self.llm = OpenAILike(
            model="deepseek-chat",
            api_base="https://api.deepseek.com/v1",
            api_key=api_key,
            is_chat_model=True,
            temperature=0.1
        )

        Settings.llm = self.llm

        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
        Settings.embed_model = self.embed_model

        self.kg_extractor = None
        self.schema_type = ""

    def set_schema(self, schema_type: str):
        self.schema_type = schema_type
        self._update_kg_extractor()

    def _update_kg_extractor(self):
        if self.schema_type == 'schema1':
            possible_entities = entities1
            possible_relations = relations1
            kg_validation_schema = validation_schema1
        else:
            possible_entities = entities2
            possible_relations = relations2
            kg_validation_schema = validation_schema2

        if self.kg_extractor is None:
            self.kg_extractor = SchemaLLMPathExtractor(
                llm=self.llm,
                possible_entities=possible_entities,
                possible_relations=possible_relations,
                possible_entity_props=None,
                possible_relation_props=None,
                kg_validation_schema=kg_validation_schema,
                max_triplets_per_chunk=10,
            )
        else:
            self.kg_extractor.update_config(
                possible_entities=possible_entities,
                possible_relations=possible_relations,
                kg_validation_schema=kg_validation_schema
            )

    def extract(self, data_dir: str):
        documents = SimpleDirectoryReader(data_dir).load_data()

        if self.kg_extractor is None:
            raise ValueError("Schema not set. Please call set_schema() before extracting.")

        index = PropertyGraphIndex.from_documents(
            documents,
            llm=self.llm,
            embed_kg_nodes=False,
            kg_extractors=[self.kg_extractor],
            show_progress=True,
        )

        return index

    def query(self, index: PropertyGraphIndex, query_text: str):
        query_engine = index.as_query_engine(include_text=True)
        response = query_engine.query(query_text)
        return response

def main():
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("API key not found. Set the DEEPSEEK_API_KEY environment variable.")

    args = json.loads(sys.argv[1])
    schema_type = args['schema']
    query_text = args['query']
    data_dir = args['data_dir']

    extractor = PGSchemaExtractor(api_key)
    extractor.set_schema(schema_type)
    index = extractor.extract(data_dir)
    response = extractor.query(index, query_text)
    
    print(json.dumps({"message": str(response)}))

if __name__ == "__main__":
    main()
