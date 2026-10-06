import os
import json
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter
from openai import OpenAI
from neo4j import GraphDatabase
from schema import LegalExtractionBatch

load_dotenv()

# 1. Initialize Neo4j Driver
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# 2. Initialize Groq via OpenAI SDK for ultra-fast, compatible inference
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# 3. Initialize LlamaParse
parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",
    verbose=True
)

EXTRACTION_PROMPT = """
You are a Senior Legal Analyst and Ontologist specializing in the Insurance Regulatory and Development Authority of India (IRDAI) framework, the Insurance Act of 1938, and the Consumer Protection Act of 2019.

Your task is to analyze the provided markdown text of an insurance regulation and extract ALL operational rules, statutory timelines, waiting periods, and consumer rights into a strict JSON structure. 

CRITICAL EXTRACTION GUIDELINES:
1. Identify all numerical or time limits as 'Thresholds' (e.g., 36 months, 30 days free-look period).
2. Extract all condition modifiers like 'Provided that', 'Except', or 'However' as 'Exceptions'.
3. Map any mention of other laws, sections, or Master Circulars under 'cross_references'.
4. Set the domain accurately based on the context ('HEALTH', 'MOTOR', 'LIFE', 'GENERAL', or 'UNIVERSAL').
5. You MUST return ONLY a valid JSON object matching the requested schema. Do not include markdown formatting like ```json.
"""

def extract_entities(markdown_chunk: str, doc_id: str) -> LegalExtractionBatch | None:
    try:
        # Pass the Pydantic schema to Groq to enforce exact JSON structure
        schema_json = LegalExtractionBatch.model_json_schema()
        
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system", 
                    "content": f"{EXTRACTION_PROMPT}\n\nRequired JSON Schema:\n{json.dumps(schema_json)}"
                },
                {
                    "role": "user", 
                    "content": f"Document ID: {doc_id}\n\nLegal Text to Process:\n{markdown_chunk}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0 # Zero temperature for absolute deterministic legal extraction
        )
        
        # Validate the Groq JSON response directly through our Pydantic model
        raw_json = response.choices[0].message.content
        return LegalExtractionBatch.model_validate_json(raw_json)
        
    except Exception as e:
        print(f"Extraction error on chunk: {e}")
        return None

def write_to_neo4j(tx, batch: LegalExtractionBatch):
    for rule in batch.rules:
        # Generate safe, deterministic Graph IDs
        safe_clause = (
            rule.clause_identifier.lower()
            .replace(" ", "_")
            .replace(".", "_")
            .replace("(", "_")
            .replace(")", "")
        )
        rule_id = f"{batch.document_id}:{safe_clause}"

        # Merge Document and Rule
        tx.run("""
            MERGE (d:Document {id: $doc_id})
            MERGE (r:Rule {id: $rule_id})
            ON CREATE SET 
                r.clause = $clause, 
                r.title = $title, 
                r.summary = $summary, 
                r.verbatim_text = $verbatim, 
                r.domain = $domain
            MERGE (d)-[:CONTAINS]->(r)
        """, doc_id=batch.document_id, rule_id=rule_id, clause=rule.clause_identifier,
             title=rule.title, summary=rule.summary, verbatim=rule.verbatim_text, domain=rule.domain)

        # Merge Thresholds
        for th in rule.thresholds:
            th_id = f"{rule_id}:th_{th.name.lower().replace(' ', '_')}"
            tx.run("""
                MATCH (r:Rule {id: $rule_id})
                MERGE (t:Threshold {id: $th_id})
                ON CREATE SET 
                    t.metric = $metric, 
                    t.numeric_value = $val, 
                    t.unit = $unit, 
                    t.raw_expression = $raw, 
                    t.comparator = $comp
                MERGE (r)-[:MANDATES]->(t)
            """, rule_id=rule_id, th_id=th_id, metric=th.metric, val=th.numeric_value,
                 unit=th.unit, raw=th.raw_expression, comp=th.comparator)

        # Merge Exceptions
        for idx, ex_text in enumerate(rule.exceptions):
            ex_id = f"{rule_id}:ex_{idx}"
            tx.run("""
                MATCH (r:Rule {id: $rule_id})
                MERGE (e:Exception {id: $ex_id})
                ON CREATE SET e.text = $text
                MERGE (e)-[:OVERRIDES]->(r)
            """, rule_id=rule_id, ex_id=ex_id, text=ex_text)

        # Merge Cross References
        for ref in rule.cross_references:
            tx.run(f"""
                MATCH (r:Rule {{id: $rule_id}})
                MERGE (target:RuleRef {{name: $target_name}})
                MERGE (r)-[:{ref.relationship_type}]->(target)
            """, rule_id=rule_id, target_name=ref.target_clause_or_act)

def process_document(file_path: str, doc_id: str):
    print(f"\n[1/3] Uploading {file_path} to LlamaParse...")
    documents = parser.load_data(file_path)
    full_markdown = "\n\n".join([doc.text for doc in documents])
    
    print("[2/3] Parsing Layout & Splitting by Legal Boundaries...")
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
    )
    splits = splitter.split_text(full_markdown)

    print(f"[3/3] Initiating Groq Extraction for {len(splits)} chunks...")
    with neo4j_driver.session() as session:
        for idx, chunk in enumerate(splits):
            print(f"  -> Processing Chunk {idx + 1}/{len(splits)} via gpt-oss-120b...")
            extracted_batch = extract_entities(chunk.page_content, doc_id)
            if extracted_batch and extracted_batch.rules:
                session.execute_write(write_to_neo4j, extracted_batch)
                print(f"Committed {len(extracted_batch.rules)} rules to Neo4j.")
            else:
                print(f"Chunk {idx + 1} yielded no valid rules.")

if __name__ == "__main__":
    # Test execution on the core health circular
    process_document("Master_Circular_on_Health_Insurance_Business.pdf", "IRDAI_MC_HEALTH_2024")
    neo4j_driver.close()