import json
from laya import Router

class LayaEngine:
    def __init__(self):
        print("Booting Local Laya System-1 Engine...")
        # Preload loads the model into RAM/VRAM to eliminate cold-start latency
        self.router = Router(preload=True) 
        print("Laya Ready.")

    def gatekeeper_check(self, user_prompt: str) -> bool:
        """Phase 1: Binary check to ensure the query is in-scope."""
        questions = {
            "intent": {
                "type": "choice",
                "instructions": "Classify the intent of this input.",
                "criteria": {
                    "in_scope": "Inquiry about Indian insurance, IRDAI regulations, legal clauses, or claim disputes.",
                    "out_of_scope": "General chat, greetings, code generation, or non-insurance topics."
                }
            }
        }
        
        # Laya evaluates the state in a single forward pass without generating text
        result = self.router.predict(user_prompt, questions)
        return result["answers"]["intent"]["choice"] == "in_scope"

    def select_strategy(self, sub_queries: list) -> str:
        """Phase 2: Determines if we need parallel federated retrieval."""
        questions = {
            "strategy": {
                "type": "choice",
                "instructions": "Determine the optimal retrieval strategy for these queries.",
                "criteria": {
                    "single": "The queries represent one highly specific legal rule.",
                    "federated": "The queries span multiple facets (e.g., medical, timeline, and precedent)."
                }
            }
        }
        
        state = json.dumps(sub_queries)
        result = self.router.predict(state, questions)
        return result["answers"]["strategy"]["choice"]

    def guardrail_check(self, drafted_response: str, retrieved_context: str) -> dict:
        """Phase 5: Self-Reflection loop for hallucination prevention."""
        state = f"Context:\n{retrieved_context}\n\nDraft:\n{drafted_response}"
        
        questions = {
            "is_faithful": {
                "type": "noul",
                "instructions": "The drafted response relies strictly on facts, numbers, and timelines present in the context, without introducing outside information."
            },
            "is_complete": {
                "type": "noul",
                "instructions": "The drafted response fully addresses the specific scenario outlined by the user."
            }
        }
        
        result = self.router.predict(state, questions)
        
        # 'noul' returns a calibrated probability float between 0.0 and 1.0
        faith_score = result["answers"]["is_faithful"]["noul"]
        complete_score = result["answers"]["is_complete"]["noul"]
        
        passed = (faith_score >= 0.85) and (complete_score >= 0.80)
        
        return {
            "passed": passed, 
            "faith_score": faith_score, 
            "complete_score": complete_score
        }