from fastapi import FastAPI
from httpx import request
from pydantic import BaseModel

app = FastAPI()


class Query(BaseModel):
    video_url: str
    question: str
    start_time: int
    end_time: int


@app.post("/")
def process_query(query: Query):
    return {"message": "Query received", "query": query}