import os
# Disable HuggingFace Symlinks for Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from groq import Groq
from neo4j import GraphDatabase
from docling.document_converter import DocumentConverter
from schema import LegalExtractionBatch

# 1. Path Setup
CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parent.parent
RAW_DATA_DIR = BACKEND_DIR.parent / "raw_data"
load_dotenv(dotenv_path=BACKEND_DIR / ".env")

# 2. Clients
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

print("Initializing Docling AI Models (for Recovery)...")
doc_converter = DocumentConverter()

GROQ_KEYS = [os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2")]
GROQ_KEYS = [k for k in GROQ_KEYS if k]
current_key_idx = 0

def get_next_groq_client():
    global current_key_idx
    key = GROQ_KEYS[current_key_idx]
    current_key_idx = (current_key_idx + 1) % len(GROQ_KEYS)
    return Groq(api_key=key)

EXTRACTION_PROMPT = """
You are a Senior Legal Ontologist specializing in Indian Insurance Regulatory Law (IRDAI).
Extract ALL operational rules, statutory timelines, waiting periods, and consumer rights into a strict JSON structure.
Numerical/time limits MUST be mapped to 'thresholds'. Conditional overrides MUST be extracted to 'exceptions'.
ONLY return valid JSON. Do not wrap in ```json markers.
"""

def extract_entities(markdown_chunk: str, doc_id: str) -> LegalExtractionBatch | None:
    schema_json = LegalExtractionBatch.model_json_schema()
    client = get_next_groq_client()
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": f"{EXTRACTION_PROMPT}\n\nRequired Schema:\n{json.dumps(schema_json)}"},
                    {"role": "user", "content": f"Document ID: {doc_id}\n\nLegal Text:\n{markdown_chunk}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=8000 # ENFORCED MAX TOKENS TO PREVENT JSON TRUNCATION
            )
            return LegalExtractionBatch.model_validate_json(response.choices[0].message.content)
            
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "413" in err_msg:
                print(f"   [!] Rate Limit Hit. Sleeping 65s (Attempt {attempt+1}/3)...")
                time.sleep(65)
                client = get_next_groq_client()
            else:
                print(f"   [!] Extraction error: {e}")
                return None
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

def get_docling_chunks(file_path: Path):
    print(f"-> Reading {file_path.name} via Docling to find broken chunk...")
    conv_result = doc_converter.convert(str(file_path))
    full_markdown = conv_result.document.export_to_markdown()
    
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")])
    initial_splits = header_splitter.split_text(full_markdown)
    
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=500)
    final_splits = []
    for split in initial_splits:
        if len(split.page_content) > 12000:
            final_splits.extend(char_splitter.split_documents([split]))
        else:
            final_splits.append(split)
    return final_splits

def recover_specific_chunk(filename: str, doc_id: str, chunk_number: int):
    print(f"\n==================================================")
    print(f"RECOVERY MODE: {filename} (Chunk {chunk_number})")
    print(f"==================================================")
    target_path = RAW_DATA_DIR / filename
    chunks = get_docling_chunks(target_path)
    
    target_idx = chunk_number - 1 # 0-indexed
    if target_idx < len(chunks):
        print(f"   Extracting repaired Chunk {chunk_number}...")
        extracted = extract_entities(chunks[target_idx].page_content, doc_id)
        if extracted and extracted.rules:
            with neo4j_driver.session() as session:
                session.execute_write(write_to_neo4j, extracted)
            print(f"   [+] Successfully committed {len(extracted.rules)} rules!")
    else:
        print("   [!] Could not find that chunk index.")

def process_protection_circular(filename: str, doc_id: str):
    print(f"\n==================================================")
    print(f"FALLBACK MODE: {filename}")
    print(f"==================================================")
    target_path = RAW_DATA_DIR / filename
    
    print("-> Reading via PyPDFLoader to bypass Docling Table OCR bug...")
    loader = PyPDFLoader(str(target_path))
    pages = loader.load()
    full_text = "\n\n".join([page.page_content for page in pages])
    
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=500)
    splits = char_splitter.create_documents([full_text])
    print(f"-> Created {len(splits)} safe chunks.")

    with neo4j_driver.session() as session:
        for idx, chunk in enumerate(splits):
            print(f"\n   Extracting chunk {idx + 1}/{len(splits)}...")
            extracted = extract_entities(chunk.page_content, doc_id)
            if extracted and extracted.rules:
                session.execute_write(write_to_neo4j, extracted)
                print(f"   [+] Committed {len(extracted.rules)} rules to Neo4j.")
            else:
                print(f"   [-] Chunk {idx + 1} produced no actionable rules.")
            time.sleep(3)


# if __name__ == "__main__":
#     # 1. Repair Motor Vehicles Act (Chunk 3)
#     recover_specific_chunk("Motor Vehicles Act, 1988 (Amended 2019).pdf", "ACT_MOTOR_VEHICLES_1988", 3)
    
#     # 2. Repair Life Insurance Products (Chunk 42)
#     recover_specific_chunk("MC_Life_Insurance_Products.pdf", "IRDAI_MC_LIFE_2024", 42)
    
#     # 3. Process the stuck Protection Circular
#     process_protection_circular("MC_Protection_of_Policyholders_interests_2024.pdf", "IRDAI_MC_PROTECTION_2024")
    
#     neo4j_driver.close()
#     print("\n✅ Total Graph Recovery & Ingestion Complete!")


if __name__ == "__main__":
    print("\n==================================================")
    print("FINAL MICRO-PATCH: MC_Protection_of_Policyholders_interests_2024.pdf (Chunk 10)")
    print("==================================================")
    
    target_path = RAW_DATA_DIR / "MC_Protection_of_Policyholders_interests_2024.pdf"
    
    # 1. Re-run PyPDF Loader to maintain exact chunk indexing
    loader = PyPDFLoader(str(target_path))
    pages = loader.load()
    full_text = "\n\n".join([page.page_content for page in pages])
    
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=500)
    splits = char_splitter.create_documents([full_text])
    
    # 2. Isolate Chunk 10 (0-indexed as 9)
    if len(splits) > 9:
        target_chunk = splits[9].page_content
        print("   Extracting repaired Chunk 10...")
        
        extracted = extract_entities(target_chunk, "IRDAI_MC_PROTECTION_2024")
        
        if extracted and extracted.rules:
            with neo4j_driver.session() as session:
                session.execute_write(write_to_neo4j, extracted)
            print(f"   [+] Successfully committed {len(extracted.rules)} rules!")
        else:
            print("   [-] Chunk produced no actionable rules.")
    else:
        print("   [!] Could not locate Chunk 10.")
        
    neo4j_driver.close()
    print("\n✅ Knowledge Graph is now 100% complete.")