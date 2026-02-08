from pydantic import BaseModel, EmailStr
from datetime import datetime




class UserCreate(BaseModel):
    email : EmailStr
    password : str 


class UserResponse(BaseModel):
    email : EmailStr
    password : str

    class Config:
        orm_mode = True


class QueryRequest(BaseModel):
    question : str


class QueryResponse(BaseModel):
    answer : str
    latency_ms: float


class HistoryItem(BaseModel):
    question: str
    answer: str
    created_at: datetime


class HistoryResponse(BaseModel):
    history: list[HistoryItem]


class HealthResponse(BaseModel):
    status: str