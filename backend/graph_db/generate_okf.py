import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

# 1. Path Resolution
CURRENT_FILE = Path(__file__).resolve()
GRAPH_DB_DIR = CURRENT_FILE.parent
BACKEND_DIR = GRAPH_DB_DIR.parent
OKF_DIR = BACKEND_DIR / "generated_okf_exports"
ENV_PATH = BACKEND_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

# Ensure the export directory exists
OKF_DIR.mkdir(parents=True, exist_ok=True)

# 2. Initialize Neo4j Driver
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

def fetch_all_domains():
    """Retrieves all unique legal domains currently present in the graph."""
    query = "MATCH (r:Rule) RETURN DISTINCT r.domain AS domain"
    with neo4j_driver.session() as session:
        result = session.run(query)
        return [record["domain"] for record in result if record["domain"]]

def generate_okf_for_domain(domain: str):
    """
    Extracts all rules, thresholds, exceptions, and cross-references for a specific domain
    and formats them into a standardized Open Knowledge Format (Markdown) document.
    """
    query = """
    MATCH (d:Document)-[:CONTAINS]->(r:Rule {domain: $domain})
    OPTIONAL MATCH (r)-[:MANDATES]->(t:Threshold)
    OPTIONAL MATCH (e:Exception)-[:OVERRIDES]->(r)
    OPTIONAL MATCH (r)-[rel]->(cr:RuleRef)
    RETURN d.id AS doc_id, r.clause AS clause, r.title AS title, 
           r.summary AS summary, r.verbatim_text AS verbatim,
           collect(DISTINCT t.raw_expression) AS thresholds,
           collect(DISTINCT e.text) AS exceptions,
           collect(DISTINCT cr.name) AS cross_references
    ORDER BY d.id, r.clause
    """
    
    with neo4j_driver.session() as session:
        result = session.run(query, domain=domain)
        records = [record.data() for record in result]
        
    if not records:
        return

    file_path = OKF_DIR / f"{domain}.md"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"# IRDAI Legal Knowledge Base: {domain}\n\n")
        f.write(f"> Auto-generated Open Knowledge Format (OKF) export from PolicyPilot Graph DB.\n")
        f.write(f"> Total Rules Indexed: {len(records)}\n\n")
        
        current_doc = ""
        for record in records:
            # Group by Source Document
            if record["doc_id"] != current_doc:
                current_doc = record["doc_id"]
                f.write(f"\n## Source Document: {current_doc}\n\n")
            
            f.write(f"### Clause {record['clause']}: {record['title']}\n")
            f.write(f"**Summary:** {record['summary']}\n\n")
            
            thresholds = [t for t in record["thresholds"] if t]
            if thresholds:
                f.write("**Statutory Thresholds & Timelines:**\n")
                for t in thresholds:
                    f.write(f"- {t}\n")
                f.write("\n")
                
            exceptions = [e for e in record["exceptions"] if e]
            if exceptions:
                f.write("**Exceptions & Overrides:**\n")
                for e in exceptions:
                    f.write(f"- {e}\n")
                f.write("\n")
                
            refs = [cr for cr in record["cross_references"] if cr]
            if refs:
                f.write("**Cross-References:**\n")
                for ref in refs:
                    f.write(f"- {ref}\n")
                f.write("\n")
                
            f.write("**Verbatim Regulatory Text:**\n")
            # Format verbatim text cleanly as a blockquote
            clean_verbatim = str(record['verbatim']).replace('\n', '\n> ')
            f.write(f"> {clean_verbatim}\n\n")
            
            f.write("---\n\n")
            
    print(f"  [+] Generated {domain}.md ({len(records)} rules mapped)")

if __name__ == "__main__":
    try:
        print("==================================================")
        print("Extracting Graph DB to Open Knowledge Format (OKF)")
        print("==================================================")
        
        domains = fetch_all_domains()
        print(f"\nFound {len(domains)} distinct legal domains. Generating markdown files...\n")
        
        for domain in domains:
            generate_okf_for_domain(domain)
            
        print(f"\n✅ OKF Export Complete! All files safely stored in {OKF_DIR}")
        
    finally:
        neo4j_driver.close()