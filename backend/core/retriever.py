import os
import torch
from pathlib import Path
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer, CrossEncoder

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
        
        # NEW: Load the Cross-Encoder for logical reranking
        print("Booting Reranker Engine...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
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

    def execute_search(self, original_query: str, sub_queries: list, domain_filters: list, final_top_k: int = 5) -> str:
        all_rule_ids = set()
        search_results = []
        
        # 1. Broad Fetch: Get Top 15 instead of Top 3 to avoid missing the "needle"
        for query in sub_queries:
            snowflake_query = f"Represent this sentence for searching relevant passages: {query}"
            dense_vector = self.dense_model.encode(snowflake_query, normalize_embeddings=True).tolist()
            sparse_result = next(self.sparse_model.query_embed(query))
            sparse_vector = models.SparseVector(
                indices=sparse_result.indices.tolist(), values=sparse_result.values.tolist()
            )
            
            res = self.qdrant_client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(query=dense_vector, using="dense", limit=15),
                    models.Prefetch(query=sparse_vector, using="sparse", limit=15)
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=15
            )
            search_results.extend(res.points)
            
        unique_rule_ids = list(set(point.payload['neo4j_rule_id'] for point in search_results))
        graph_data = self._get_graph_context(unique_rule_ids)
        
        # 2. Hard Domain Gating
        filtered_candidates = []
        for rule_id, context in graph_data.items():
            domain = context.get('domain', 'UNIVERSAL')
            # If the LLM specified domains, strictly drop mismatched domains (allow UNIVERSAL fallback)
            if domain_filters and domain not in domain_filters and domain != 'UNIVERSAL':
                continue
            filtered_candidates.append(context)
            
        if not filtered_candidates:
            return "No relevant regulatory context found for the specified domain."

        # 3. Cross-Encoder Reranking
        # Pair the original user query against every retrieved clause's text
        cross_inp = [[original_query, c['verbatim']] for c in filtered_candidates]
        scores = self.cross_encoder.predict(cross_inp)
        
        for idx, c in enumerate(filtered_candidates):
            c['rerank_score'] = float(scores[idx])
            
        # Sort by logical relevance and slice the Top 5
        filtered_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        top_candidates = filtered_candidates[:final_top_k]
        
        # 4. Context Assembly
        context_str = ""
        for context in top_candidates:
            context_str += f"Clause: {context.get('clause')} (Domain: {context.get('domain')})\n"
            context_str += f"Verbatim: {context.get('verbatim')}\n"
            if context.get('thresholds'):
                context_str += f"Thresholds: {', '.join(t for t in context['thresholds'] if t)}\n"
            if context.get('exceptions'):
                context_str += f"Exceptions: {', '.join(e for e in context['exceptions'] if e)}\n"
            context_str += "---\n"
            
        return context_str