from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from .pipeline import answer_question

app = FastAPI(title="SEC RAG API", version="1.0")

class QueryIn(BaseModel):
    query: str

@app.post("/answer")
def answer(q: QueryIn):
    return answer_question(q.query)
