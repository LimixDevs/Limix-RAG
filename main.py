from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from SmartContractRAG import chat_with_model, detect_vulnerabilities

app = FastAPI()
# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the Smart Contract RAG API!"}


@app.post("/chat")
def chat(query_request: QueryRequest):
    user_query = query_request.query
    response = chat_with_model(user_query)
    return {"response": response}


@app.post("/detect_vulnerabilities")
def detect_vulnerabilities_endpoint(query_request: QueryRequest):
    user_query = query_request.query
    vulnerabilities = detect_vulnerabilities(user_query)
    return {"vulnerabilities": vulnerabilities}
