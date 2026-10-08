from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from core.workflow import AdjudicationWorkflow

load_dotenv()

app = FastAPI(title="PolicyPilot Agentic GraphRAG")
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)