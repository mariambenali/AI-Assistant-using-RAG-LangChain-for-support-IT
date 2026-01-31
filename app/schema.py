from pydantic import BaseModel, EmailStr
from datetime import datetime




class UserCreate(BaseModel):
    email : EmailStr
    password : str


class UserResponse(BaseModel):
    id : int
    email : EmailStr

    class Config:
        orm_mode = True


class Query(BaseModel):
    question : str


class QueryResponse(BaseModel):
    answer : str


class HistoryItem(BaseModel):
    question: str
    answer: str
    created_at: datetime


class HistoryResponse(BaseModel):
    history: list[HistoryItem]


class HealthResponse(BaseModel):
    status: str