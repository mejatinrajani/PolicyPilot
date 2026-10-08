from core.jev_client import JevEngine
from core.groq_client import GroqEngine
from core.retriever import FederatedRetriever

class AdjudicationWorkflow:
    def __init__(self):
        self.jev = JevEngine()
        self.groq = GroqEngine()
        self.retriever = FederatedRetriever()

    async def run(self, original_query: str) -> dict:
        metrics = {"loops": 0, "status": "success", "jev_scores": {}}
        
        # Phase 1: Gatekeeper
        if not self.jev.gatekeeper_check(original_query):
            return {
                "final_ruling": self.groq.fast_fallback(),
                "execution_metrics": {"status": "out_of_scope"}
            }
            
        # Phase 2: Planner & Strategy
        sub_queries = self.groq.decompose_query(original_query)
        strategy = self.jev.select_strategy(sub_queries)
        
        # Phase 3: Retrieval
        context = self.retriever.execute_search(sub_queries)
        
        # Phase 4 & 5: Synthesis & Guardrail Loop
        max_retries = 2
        critique = None
        final_ruling = ""
        
        for attempt in range(max_retries + 1):
            metrics["loops"] = attempt
            final_ruling = self.groq.synthesize_adjudication(original_query, context, critique)
            
            # Jev Validation
            validation = self.jev.guardrail_check(final_ruling, context)
            metrics["jev_scores"] = {
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