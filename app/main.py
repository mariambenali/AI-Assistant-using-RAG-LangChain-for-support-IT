from .models import User, Query
from .schema import UserCreate, UserResponse, Query, QueryResponse, HistoryItem, HistoryResponse, HealthResponse
from .security import hash_password, verify_password
from .database import SessionLocal, engine, Base
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from jose import jwt
from dotenv import load_dotenv
import os


load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")



app = FastAPI()

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/register")
def register(user:UserCreate, db: Session = Depends(get_db)):
    hash_pwd = hash_password(user.password) 
    new_user=User (
        password= hash_pwd,
        created_at= datetime.today()
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user

    
@app.get("/login")
def login(user:UserResponse, db: Session = Depends(get_db)):
    user_login = db.query(User).filter(User.username == user.username).first()

    if not user_login:
        raise HTTPException(status_code = 404, detail="User not found")
    elif not verify_password(user.password, user_login.password):
        raise HTTPException(status_code=401, detail="Invalid credentials") 
    
    payload = {"sub": user_login.username}
    token = jwt.encode(payload,SECRET_KEY,algorithm=ALGORITHM)

    return token
    
