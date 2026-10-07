import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

# 1. Dynamic Path Resolution
CURRENT_FILE = Path(__file__).resolve()
VECTOR_DB_DIR = CURRENT_FILE.parent
BACKEND_DIR = VECTOR_DB_DIR.parent
QDRANT_STORAGE_PATH = VECTOR_DB_DIR / "qdrant_storage"
ENV_PATH = BACKEND_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

# 2. Initialize Neo4j
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# 3. Initialize Local Qdrant
print("Initializing Local Qdrant Database...")
qdrant_client = QdrantClient(path=str(QDRANT_STORAGE_PATH))
COLLECTION_NAME = "irdai_legal_rules"

# 4. Initialize the Hybrid Models
print("Loading Snowflake Arctic v2.0 & SPLADE Models (Runs locally)...")
# Snowflake model for Dense Semantic Meaning (1024 dimensions)
dense_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")

# FastEmbed SPLADE for Sparse Keyword Matching
sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

def setup_qdrant_collection():
    if qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' exists. Dropping for a fresh sync...")
        qdrant_client.delete_collection(COLLECTION_NAME)
        
    print(f"Creating Hybrid Collection '{COLLECTION_NAME}'...")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=1024, # Updated to match Snowflake Arctic's 1024 dimensions
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )

def fetch_all_rules_from_neo4j():
    print("Fetching legal rules and relationships from Neo4j...")
    query = """
    MATCH (d:Document)-[:CONTAINS]->(r:Rule)
    OPTIONAL MATCH (r)-[:MANDATES]->(t:Threshold)
    OPTIONAL MATCH (e:Exception)-[:OVERRIDES]->(r)
    RETURN d.id AS doc_id, 
           r.id AS rule_id, 
           r.clause AS clause, 
           r.title AS title, 
           r.summary AS summary, 
           r.verbatim_text AS verbatim, 
           r.domain AS domain,
           collect(DISTINCT t.raw_expression) AS thresholds,
           collect(DISTINCT e.text) AS exceptions
    """
    
    with neo4j_driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]

def format_for_embedding(record):
    thresholds_text = ", ".join([t for t in record["thresholds"] if t])
    exceptions_text = " | ".join([e for e in record["exceptions"] if e])
    
    semantic_text = f"Title: {record['title']}\n"
    semantic_text += f"Domain: {record['domain']}\n"
    semantic_text += f"Summary: {record['summary']}\n"
    semantic_text += f"Verbatim: {record['verbatim']}\n"
    
    if thresholds_text:
        semantic_text += f"Time/Numerical Limits: {thresholds_text}\n"
    if exceptions_text:
        semantic_text += f"Exceptions/Conditions: {exceptions_text}\n"
        
    return semantic_text

def ingest_to_qdrant():
    setup_qdrant_collection()
    records = fetch_all_rules_from_neo4j()
    
    print(f"Formatting {len(records)} rules and calculating embeddings...")
    documents = [format_for_embedding(record) for record in records]
    
    # Prefix required by Snowflake for document indexing
    snowflake_docs = [f" {doc}" for doc in documents] 
    
    print("Generating Dense Embeddings (Snowflake Arctic v2.0)...")
    dense_embeddings = dense_model.encode(snowflake_docs)
    
    print("Generating Sparse Embeddings (SPLADE)...")
    sparse_embeddings = list(sparse_model.embed(documents))
    
    print("Uploading vectorized payloads to Qdrant...")
    points = []
    for idx, record in enumerate(records):
        sparse_vector = models.SparseVector(
            indices=sparse_embeddings[idx].indices.tolist(),
            values=sparse_embeddings[idx].values.tolist()
        )
        
        points.append(
            models.PointStruct(
                id=idx,
                vector={
                    "dense": dense_embeddings[idx].tolist(),
                    "sparse": sparse_vector
                },
                payload={
                    "neo4j_rule_id": record["rule_id"],
                    "document_id": record["doc_id"],
                    "clause": record["clause"],
                    "domain": record["domain"],
                    "text_content": documents[idx]
                }
            )
        )
        
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    print(f"✅ Hybrid Vector Sync Complete! Embedded {len(records)} rules locally in {QDRANT_STORAGE_PATH}")

if __name__ == "__main__":
    try:
        ingest_to_qdrant()
    finally:
        neo4j_driver.close()
        qdrant_client.close()   