import os
import json
from typesafe_sdk import TypeSafeClient, Choice, Noul

class JevEngine:
    def __init__(self):
        # We initialize the client to be used cleanly across the pipeline
        self.api_key = os.environ.get("TYPESAFE_API_KEY")

    def gatekeeper_check(self, user_prompt: str) -> bool:
        """Phase 1: Binary check to ensure the query is in-scope."""
        question = Choice(
            instruction="Classify the intent of this input.",
            criteria={
                "in_scope": "Inquiry about Indian insurance, IRDAI regulations, legal clauses, or claim disputes.",
                "out_of_scope": "General chat, greetings, code generation, or non-insurance topics."
            }
        )
        with TypeSafeClient(api_key=self.api_key) as client:
            response = client.system_one(
                model="jev-latest",
                state=user_prompt,
                questions={"intent": question}
            )
            return response.answers["intent"].choice == "in_scope"

    def select_strategy(self, sub_queries: list) -> str:
        """Phase 2: Determines if we need parallel federated retrieval."""
        question = Choice(
            instruction="Determine the optimal retrieval strategy for these queries.",
            criteria={
                "single": "The queries represent one highly specific legal rule.",
                "federated": "The queries span multiple facets (e.g., medical, timeline, and precedent)."
            }
        )
        with TypeSafeClient(api_key=self.api_key) as client:
            response = client.system_one(
                model="jev-latest",
                state=json.dumps(sub_queries),
                questions={"strategy": question}
            )
            return response.answers["strategy"].choice

    def guardrail_check(self, drafted_response: str, retrieved_context: str) -> dict:
        """Phase 5: Self-Reflection loop for hallucination prevention."""
        state = f"Context:\n{retrieved_context}\n\nDraft:\n{drafted_response}"
        
        hallucination_check = Noul(
            instruction="The drafted response relies strictly on facts, numbers, and timelines present in the context, without introducing outside information."
        )
        completeness_check = Noul(
            instruction="The drafted response fully addresses the specific scenario outlined by the user."
        )
        
        with TypeSafeClient(api_key=self.api_key) as client:
            response = client.system_one(
                model="jev-latest",
                state=state,
                questions={
                    "is_faithful": hallucination_check,
                    "is_complete": completeness_check
                }
            )
            
            faith_score = response.answers["is_faithful"].noul
            complete_score = response.answers["is_complete"].noul
            
            # Require high confidence to pass the guardrail
            passed = (faith_score >= 0.85) and (complete_score >= 0.80)
            
            return {
                "passed": passed, 
                "faith_score": faith_score, 
                "complete_score": complete_score
            }