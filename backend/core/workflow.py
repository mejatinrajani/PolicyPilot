from core.laya_client import LayaEngine
from core.groq_client import GroqEngine
from core.retriever import FederatedRetriever

class AdjudicationWorkflow:
    def __init__(self):
        self.laya = LayaEngine()
        self.groq = GroqEngine()
        self.retriever = FederatedRetriever()

    async def run(self, original_query: str) -> dict:
        metrics = {"loops": 0, "status": "success", "laya_scores": {}}
        
        # Phase 1: Gatekeeper
        if not self.laya.gatekeeper_check(original_query):
            return {
                "final_ruling": self.groq.fast_fallback(),
                "execution_metrics": {"status": "out_of_scope"}
            }
            
        # Phase 2: Planner & Strategy
        planner_output = self.groq.decompose_query(original_query)
        sub_queries = planner_output.get("queries", [original_query])
        domain_filters = planner_output.get("domain_filters", [])
        
        strategy = self.laya.select_strategy(sub_queries)
        
        # Phase 3: Retrieval (Now passes the original query and domain filters)
        context = self.retriever.execute_search(original_query, sub_queries, domain_filters)
        
        # Phase 4 & 5: Synthesis & Guardrail Loop
        max_retries = 2
        critique = None
        final_ruling = ""
        
        for attempt in range(max_retries + 1):
            metrics["loops"] = attempt
            final_ruling = self.groq.synthesize_adjudication(original_query, context, critique)
            
            # Local Validation via Laya
            validation = self.laya.guardrail_check(final_ruling, context)
            metrics["laya_scores"] = {
                "faithfulness": validation["faith_score"],
                "completeness": validation["complete_score"]
            }
            
            if validation["passed"]:
                break
            else:
                if validation["faith_score"] < 0.85:
                    critique = "The previous draft included hallucinated facts not present in the context. Ensure strict adherence."
                elif validation["complete_score"] < 0.80:
                    critique = "The previous draft did not fully address all aspects of the user's specific scenario. Expand the legal reasoning."
                    
        return {
            "final_ruling": final_ruling,
            "execution_metrics": metrics
        }