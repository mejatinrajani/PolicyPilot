import os
import json
from groq import Groq

class GroqEngine:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY_1"))
        self.fast_model = "openai/gpt-oss-20b"
        self.heavy_model = "openai/gpt-oss-120b"

    def fast_fallback(self) -> str:
        """Phase 1 Fallback: Fast exit for out-of-scope queries."""
        response = self.client.chat.completions.create(
            model=self.fast_model,
            messages=[{
                "role": "system", 
                "content": "Politely explain that PolicyPilot only handles Indian insurance regulatory matters. Provide support@policypilot.in for assistance. Keep it under 2 sentences."
            }]
        )
        return response.choices[0].message.content

    def decompose_query(self, user_prompt: str) -> list:
        """Phase 2: Breaks messy scenarios into atomic legal searches."""
        system_prompt = """
        You are a legal analyst. Deconstruct the user's scenario into a JSON array of 1 to 3 atomic search queries optimized for a vector database.
        Output ONLY raw JSON format with a 'queries' key.
        """
        response = self.client.chat.completions.create(
            model=self.heavy_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = json.loads(response.choices[0].message.content)
        return content.get("queries", [user_prompt])

    def synthesize_adjudication(self, original_query: str, context: str, critique: str = None) -> str:
        """Phase 4: Drafts the final ruling. Re-runs if Phase 5 injects a critique."""
        system_prompt = """
        You are a Legal Adjudicator for Indian Insurance Regulatory Law (IRDAI).
        Draft a courtroom-level ruling using ONLY the provided context.
        1. Explicitly cite the Rule/Clause ID.
        2. Highlight explicit thresholds (e.g., 30-day windows).
        3. Provide actionable recourse steps.
        """
        
        if critique:
            system_prompt += f"\nCRITICAL CORRECTION FROM GUARDRAIL: {critique}. Rewrite the adjudication and fix this error."
            
        response = self.client.chat.completions.create(
            model=self.heavy_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nUser Question: {original_query}"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content