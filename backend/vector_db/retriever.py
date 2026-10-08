import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

# 1. Path Resolution
CURRENT_FILE = Path(__file__).resolve()
VECTOR_DB_DIR = CURRENT_FILE.parent
BACKEND_DIR = VECTOR_DB_DIR.parent
QDRANT_STORAGE_PATH = VECTOR_DB_DIR / "storage"
ENV_PATH = BACKEND_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

# 2. Initialize Databases
qdrant_client = QdrantClient(path=str(QDRANT_STORAGE_PATH))
COLLECTION_NAME = "irdai_legal_rules"

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# 3. Load Embedding Models
device = "cuda" if torch.cuda.is_available() else "cpu"
dense_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0", device=device)
sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

def get_graph_context(rule_ids: list):
    """
    Takes the Rule IDs found by the Vector Search and traverses the Neo4j Graph
    to find attached Thresholds, Exceptions, and Cross-References.
    """
    query = """
    MATCH (r:Rule)
    WHERE r.id IN $rule_ids
    OPTIONAL MATCH (r)-[:MANDATES]->(t:Threshold)
    OPTIONAL MATCH (e:Exception)-[:OVERRIDES]->(r)
    OPTIONAL MATCH (r)-[rel]->(cr:RuleRef)
    RETURN r.id AS rule_id, 
           r.clause AS clause,
           r.domain AS domain,
           r.verbatim_text AS verbatim,
           collect(DISTINCT t.raw_expression) AS thresholds,
           collect(DISTINCT e.text) AS exceptions,
           collect(DISTINCT cr.name) AS cross_references
    """
    with neo4j_driver.session() as session:
        result = session.run(query, rule_ids=rule_ids)
        return {record["rule_id"]: record.data() for record in result}

def search_and_enrich(user_query: str, top_k: int = 3):
    print(f"\n🔍 User Query: '{user_query}'")
    
    # 1. Vector Search (Find the most relevant rules)
    snowflake_query = f"Represent this sentence for searching relevant passages: {user_query}"
    dense_vector = dense_model.encode(snowflake_query, normalize_embeddings=True).tolist()
    
    sparse_result = next(sparse_model.query_embed(user_query))
    sparse_vector = models.SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist()
    )
    
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vector, using="dense", limit=10),
            models.Prefetch(query=sparse_vector, using="sparse", limit=10)
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    )
    
    # 2. Extract IDs and query Graph Database
    rule_ids = [point.payload['neo4j_rule_id'] for point in search_results.points]
    graph_data = get_graph_context(rule_ids)
    
    # 3. Output the fully enriched GraphRAG context
    print("\n" + "="*70)
    print("⚖️  GRAPH-ENRICHED LEGAL CONTEXT")
    print("="*70)
    
    for idx, point in enumerate(search_results.points, 1):
        rule_id = point.payload['neo4j_rule_id']
        context = graph_data.get(rule_id, {})
        
        print(f"\n[{idx}] Clause: {context.get('clause', 'N/A')} ({context.get('domain', 'N/A')})")
        print(f"    Confidence Score: {point.score:.4f}")
        
        thresholds = [t for t in context.get('thresholds', []) if t]
        if thresholds:
            print(f"    ⏳ Thresholds: {', '.join(thresholds)}")
            
        exceptions = [e for e in context.get('exceptions', []) if e]
        if exceptions:
            print(f"    ⚠️ Exceptions: {', '.join(exceptions)}")
            
        refs = [r for r in context.get('cross_references', []) if r]
        if refs:
            print(f"    🔗 Cross-Refs: {', '.join(refs)}")
            
        print("-" * 70)

if __name__ == "__main__":
    try:
        search_and_enrich("What is the free look period for a health policy?")
        search_and_enrich("The insurance company rejected my claim because I didn't tell them about a previous illness. Can they do this?")
    finally:
        # Clean teardown to prevent Windows sys.meta_path errors
        qdrant_client.close()
        neo4j_driver.close()