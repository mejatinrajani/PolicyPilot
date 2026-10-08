import os
import torch
from pathlib import Path
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

class FederatedRetriever:
    def __init__(self):
        backend_dir = Path(__file__).resolve().parent.parent
        self.qdrant_client = QdrantClient(path=str(backend_dir / "vector_db" / "storage"))
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dense_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0", device=device)
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        self.collection = "irdai_legal_rules"

    def _get_graph_context(self, rule_ids: list) -> dict:
        query = """
        MATCH (r:Rule) WHERE r.id IN $rule_ids
        OPTIONAL MATCH (r)-[:MANDATES]->(t:Threshold)
        OPTIONAL MATCH (e:Exception)-[:OVERRIDES]->(r)
        RETURN r.id AS rule_id, r.clause AS clause, r.domain AS domain,
               r.verbatim_text AS verbatim,
               collect(DISTINCT t.raw_expression) AS thresholds,
               collect(DISTINCT e.text) AS exceptions
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, rule_ids=rule_ids)
            return {record["rule_id"]: record.data() for record in result}

    def execute_search(self, sub_queries: list, top_k: int = 3) -> str:
        """Phase 3: Executes RRF search and Graph enrichment across sub-queries."""
        all_rule_ids = set()
        search_results = []
        
        # Fan out across the decomposed sub-queries
        for query in sub_queries:
            snowflake_query = f"Represent this sentence for searching relevant passages: {query}"
            dense_vector = self.dense_model.encode(snowflake_query, normalize_embeddings=True).tolist()
            
            sparse_result = next(self.sparse_model.query_embed(query))
            sparse_vector = models.SparseVector(
                indices=sparse_result.indices.tolist(),
                values=sparse_result.values.tolist()
            )
            
            res = self.qdrant_client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(query=dense_vector, using="dense", limit=5),
                    models.Prefetch(query=sparse_vector, using="sparse", limit=5)
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k
            )
            search_results.extend(res.points)
            
        # Deduplicate results
        unique_points = []
        for point in search_results:
            rule_id = point.payload['neo4j_rule_id']
            if rule_id not in all_rule_ids:
                all_rule_ids.add(rule_id)
                unique_points.append(point)
                
        # Bridge to Neo4j
        graph_data = self._get_graph_context(list(all_rule_ids))
        
        # Assemble Payload
        context_str = ""
        for point in unique_points:
            rule_id = point.payload['neo4j_rule_id']
            context = graph_data.get(rule_id, {})
            
            context_str += f"Clause: {context.get('clause')} (Domain: {context.get('domain')})\n"
            context_str += f"Verbatim: {context.get('verbatim')}\n"
            
            if context.get('thresholds'):
                context_str += f"Thresholds: {', '.join(t for t in context['thresholds'] if t)}\n"
            if context.get('exceptions'):
                context_str += f"Exceptions: {', '.join(e for e in context['exceptions'] if e)}\n"
            context_str += "---\n"
            
        return context_str