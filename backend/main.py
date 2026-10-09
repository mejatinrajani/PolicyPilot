import os
# MUST BE AT THE VERY TOP: Fixes Windows [WinError 1314] Symlink crash for HuggingFace / Laya downloads
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env before initializing the workflow so Groq keys are available
load_dotenv()

from core.workflow import AdjudicationWorkflow

app = FastAPI(title="PolicyPilot Agentic GraphRAG")

# Initializes Laya, Qdrant, Neo4j, and Groq once in the main process
workflow = AdjudicationWorkflow()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    ruling: str
    metrics: dict

@app.post("/adjudicate", response_model=QueryResponse)
async def process_legal_query(request: QueryRequest):
    try:
        result = await workflow.run(request.query)
        return QueryResponse(
            ruling=result["final_ruling"],
            metrics=result["execution_metrics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # FIXED: Passed 'app' directly and removed reload=True to prevent Windows file-lock crashes
    uvicorn.run(app, host="0.0.0.0", port=8000)