import os
# MUST BE AT THE VERY TOP: Fixes Windows [WinError 1314] Symlink crash
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from groq import Groq
from neo4j import GraphDatabase
from docling.document_converter import DocumentConverter
from schema import LegalExtractionBatch

CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
ENV_PATH = BACKEND_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

print("Initializing Docling AI Models (Symlinks Disabled)...")
doc_converter = DocumentConverter()

GROQ_KEYS = [os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2")]
GROQ_KEYS = [k for k in GROQ_KEYS if k]
if not GROQ_KEYS:
    raise ValueError("No Groq API keys found in .env.")

current_key_idx = 0

def get_next_groq_client():
    global current_key_idx
    key = GROQ_KEYS[current_key_idx]
    current_key_idx = (current_key_idx + 1) % len(GROQ_KEYS)
    return Groq(api_key=key)

EXTRACTION_PROMPT = """
You are a Senior Legal Ontologist specializing in Indian Insurance Regulatory Law (IRDAI).
Extract ALL operational rules, statutory timelines, waiting periods, and consumer rights into a strict JSON structure.

CRITICAL INSTRUCTIONS:
1. Numerical/time limits (e.g., 36 months, 30 days) MUST be mapped to 'thresholds'.
2. Conditional overrides ('Provided that', 'Except') MUST be extracted verbatim into 'exceptions'.
3. Cross-references to other Acts MUST be extracted to 'cross_references'.
4. Ensure domain is classified accurately ('HEALTH', 'MOTOR', 'LIFE', 'GENERAL', or 'UNIVERSAL').
5. ONLY return valid JSON matching the exact provided schema. Do not wrap in ```json markers.
"""

def extract_entities_with_retry(markdown_chunk: str, doc_id: str, max_retries=3) -> LegalExtractionBatch | None:
    schema_json = LegalExtractionBatch.model_json_schema()
    client = get_next_groq_client()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system", 
                        "content": f"{EXTRACTION_PROMPT}\n\nRequired Schema:\n{json.dumps(schema_json)}"
                    },
                    {"role": "user", "content": f"Document ID: {doc_id}\n\nLegal Text:\n{markdown_chunk}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            raw_json = response.choices[0].message.content
            return LegalExtractionBatch.model_validate_json(raw_json)
            
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "rate limit" in err_msg or "413" in err_msg:
                wait_time = 65  # Groq TPM limits reset every 60 seconds
                print(f"   [!] Groq Limit Hit. Sleeping for {wait_time}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
                client = get_next_groq_client() 
            else:
                print(f"   [!] Extraction error: {e}")
                return None
                
    print("   [!] Max retries exhausted for this chunk.")
    return None

def write_to_neo4j(tx, batch: LegalExtractionBatch):
    for rule in batch.rules:
        safe_clause = rule.clause_identifier.lower().replace(" ", "_").replace(".", "_").replace("(", "_").replace(")", "")
        rule_id = f"{batch.document_id}:{safe_clause}"

        tx.run("""
            MERGE (d:Document {id: $doc_id})
            MERGE (r:Rule {id: $rule_id})
            ON CREATE SET r.clause = $clause, r.title = $title, r.summary = $summary, 
                          r.verbatim_text = $verbatim, r.domain = $domain
            MERGE (d)-[:CONTAINS]->(r)
        """, doc_id=batch.document_id, rule_id=rule_id, clause=rule.clause_identifier,
             title=rule.title, summary=rule.summary, verbatim=rule.verbatim_text, domain=rule.domain)

        for th in rule.thresholds:
            th_id = f"{rule_id}:th_{th.name.lower().replace(' ', '_')}"
            tx.run("""
                MATCH (r:Rule {id: $rule_id})
                MERGE (t:Threshold {id: $th_id})
                ON CREATE SET t.metric = $metric, t.numeric_value = $val, t.unit = $unit, 
                              t.raw_expression = $raw, t.comparator = $comp
                MERGE (r)-[:MANDATES]->(t)
            """, rule_id=rule_id, th_id=th_id, metric=th.metric, val=th.numeric_value,
                 unit=th.unit, raw=th.raw_expression, comp=th.comparator)

        for idx, ex_text in enumerate(rule.exceptions):
            ex_id = f"{rule_id}:ex_{idx}"
            tx.run("""
                MATCH (r:Rule {id: $rule_id})
                MERGE (e:Exception {id: $ex_id})
                ON CREATE SET e.text = $text
                MERGE (e)-[:OVERRIDES]->(r)
            """, rule_id=rule_id, ex_id=ex_id, text=ex_text)

        for ref in rule.cross_references:
            tx.run(f"""
                MATCH (r:Rule {{id: $rule_id}})
                MERGE (target:RuleRef {{name: $target_name}})
                MERGE (r)-[:{ref.relationship_type}]->(target)
            """, rule_id=rule_id, target_name=ref.target_clause_or_act)

def process_file(file_path: Path, doc_id: str):
    print(f"\n==================================================")
    print(f"Processing: {file_path.name}")
    print(f"==================================================")

    if file_path.suffix.lower() == ".md":
        print(f"-> Reading native Markdown file...")
        with open(file_path, "r", encoding="utf-8") as f:
            full_markdown = f.read()
    else:
        print(f"-> Parsing locally via Docling...")
        conv_result = doc_converter.convert(str(file_path))
        full_markdown = conv_result.document.export_to_markdown()

    # Step 1: Structural Split by Headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
    )
    initial_splits = header_splitter.split_text(full_markdown)

    # Step 2: Safety Split for Token Limits (Max ~3000 tokens per chunk)
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=500)
    final_splits = []
    
    for split in initial_splits:
        if len(split.page_content) > 12000:
            # FIXED: Used plural split_documents and wrapped split in a list
            final_splits.extend(char_splitter.split_documents([split]))
        else:
            final_splits.append(split)

    print(f"-> Created {len(final_splits)} safe chunks.")

    with neo4j_driver.session() as session:
        for idx, chunk in enumerate(final_splits):
            print(f"\n   Extracting chunk {idx + 1}/{len(final_splits)}...")
            extracted_batch = extract_entities_with_retry(chunk.page_content, doc_id)
            
            if extracted_batch and extracted_batch.rules:
                session.execute_write(write_to_neo4j, extracted_batch)
                print(f"   [+] Committed {len(extracted_batch.rules)} rules to Neo4j.")
            else:
                print(f"   [-] Chunk {idx + 1} produced no actionable rules.")
            
            # Base 3-second sleep between standard API calls to pace the free tier
            time.sleep(3)

# REMOVED the 3 successfully processed files to save tokens
DOCUMENT_REGISTRY = {
    "Master_Circular_on_Health_Insurance_Business_29052024.pdf": "IRDAI_MC_HEALTH_2024",
    "Insurance_Act_1938.pdf": "ACT_INSURANCE_1938",
    "Motor Vehicles Act, 1988 (Amended 2019).pdf": "ACT_MOTOR_VEHICLES_1988",
    "MC_Life_Insurance_Products.pdf": "IRDAI_MC_LIFE_2024",
    "MC_Protection_of_Policyholders_interests_2024.pdf": "IRDAI_MC_PROTECTION_2024",
    "MC_Operations_and_Allied_Matters_of_Insurers.pdf": "IRDAI_MC_OPERATIONS_2024",
    "Supreme_Court_Precedents.md": "SC_PRECEDENTS_CANONICAL"
}

if __name__ == "__main__":
    for filename, doc_id in DOCUMENT_REGISTRY.items():
        target_path = RAW_DATA_DIR / filename
        
        if not target_path.exists():
            print(f"\n⚠️ Warning: Could not find {filename}. Skipping.")
            continue
            
        try:
            process_file(target_path, doc_id)
        except Exception as e:
            print(f"\n❌ Failed to process {filename}: {e}")

    neo4j_driver.close()
    print("\n✅ Full Ingestion pipeline complete.")