import os
import json
import logging
import threading
from groq import Groq, RateLimitError

logger = logging.getLogger(__name__)

class GroqEngine:
    def __init__(self):
        # 1. Strictly load exactly two API keys
        key_1 = os.getenv("GROQ_API_KEY_1")
        key_2 = os.getenv("GROQ_API_KEY_2")
        
        self.clients = []
        if key_1:
            self.clients.append(Groq(api_key=key_1))
        if key_2:
            self.clients.append(Groq(api_key=key_2))
            
        if not self.clients:
            raise ValueError("No Groq API keys found. Please set GROQ_API_KEY_1 and/or GROQ_API_KEY_2 in your environment.")

        self.current_idx = 0
        self.lock = threading.Lock()
        
        # 2. Use Groq's official production GPT-OSS models
        self.fast_model = "openai/gpt-oss-20b"
        self.heavy_model = "openai/gpt-oss-120b"

    def _get_next_client(self):
        """Thread-safe round-robin selection between the two Groq clients."""
        with self.lock:
            client = self.clients[self.current_idx]
            self.current_idx = (self.current_idx + 1) % len(self.clients)
            return client, self.current_idx

    def _execute_with_rotation(self, **kwargs):
        """
        Executes the API call using the active key.
        If a RateLimitError occurs, it immediately fails over to the second key.
        """
        max_attempts = len(self.clients)
        
        for attempt in range(max_attempts):
            client, client_id = self._get_next_client()
            try:
                return client.chat.completions.create(**kwargs)
            
            except RateLimitError:
                key_name = f"GROQ_API_KEY_{client_id + 1}"
                logger.warning(f"Rate limit hit on {key_name}. Failing over to the alternate key...")
                
                # If we've tried both keys and they are both rate-limited, fail gracefully
                if attempt == max_attempts - 1:
                    logger.error("Both Groq API keys are currently rate-limited.")
                    raise
                    
        raise RuntimeError("Request failed across both available keys.")

    def fast_fallback(self) -> str:
        response = self._execute_with_rotation(
            model=self.fast_model,
            messages=[{
                "role": "system", 
                "content": "Politely explain that PolicyPilot only handles Indian insurance regulatory matters. Provide support@policypilot.in for assistance. Keep it under 2 sentences."
            }]
        )
        return response.choices[0].message.content

    def decompose_query(self, user_prompt: str) -> dict:
        system_prompt = """
        You are a legal analyst. Deconstruct the user's scenario.
        Output ONLY raw JSON format with two keys:
        1. 'domain_filters': An array of applicable insurance domains. Choose ONLY from: ["LIFE", "HEALTH", "MOTOR", "GENERAL", "UNIVERSAL"]. If unsure, include "UNIVERSAL".
        2. 'queries': An array of 1 to 3 atomic search queries optimized for a vector database.
        """
        response = self._execute_with_rotation(
            model=self.heavy_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def synthesize_adjudication(self, original_query: str, context: str, critique: str = None) -> str:
        system_prompt = """
        You are a Legal Adjudicator for Indian Insurance Regulatory Law (IRDAI).
        Draft a courtroom-level ruling using ONLY the provided context.
        1. Explicitly cite the Rule/Clause ID.
        2. Highlight explicit thresholds (e.g., 30-day windows).
        3. Provide actionable recourse steps.
        
        STRICT ANTI-HALLUCINATION RULE: You are FORBIDDEN from including any statutory timelines, limits, or days (e.g., 15 days, 30 days, 2 years) that are not explicitly written in the provided context. Do not add them with a disclaimer. If a timeline is not in the context, write "Not specified in retrieved context."
        """
        
        if critique:
            system_prompt += f"\nCRITICAL CORRECTION FROM GUARDRAIL: {critique}. Rewrite the adjudication and fix this error."
            
        response = self._execute_with_rotation(
            model=self.heavy_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nUser Question: {original_query}"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content