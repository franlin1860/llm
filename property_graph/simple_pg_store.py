from llama_index.core.graph_stores import (
    SimplePropertyGraphStore,
    EntityNode,
    Relation,
)
from llama_index.core.schema import TextNode

graph_store = SimplePropertyGraphStore()

entities = [
    EntityNode(name="llama", label="ANIMAL", properties={"key": "val"}),
    EntityNode(name="index", label="THING", properties={"key": "val"}),
]

relations = [
    Relation(
        label="HAS",
        source_id=entities[0].id,
        target_id=entities[1].id,
        properties={},
    )
]

graph_store.upsert_nodes(entities)
graph_store.upsert_relations(relations)

# optionally, we can also insert text chunks
source_chunk = TextNode(id_="source", text="My llama has an index.")

# create relation for each of our entities
source_relations = [
    Relation(
        label="HAS_SOURCE",
        source_id=entities[0].id,
        target_id="source",
    ),
    Relation(
        label="HAS_SOURCE",
        source_id=entities[1].id,
        target_id="source",
    ),
]
graph_store.upsert_llama_nodes([source_chunk])
graph_store.upsert_relations(source_relations)
