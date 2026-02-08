from sqlalchemy import Column, String, Integer,Boolean, DateTime, ForeignKey
from .database import Base, engine
from datetime import datetime
from sqlalchemy.orm import relationship


Base.metadata.create_all(bind=engine)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email =Column(String, unique=True, nullable=False)
    hashedpassword=Column(String, nullable=False)
    isactive =Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    #relation 1 user ---> many queries
    queries = relationship("Query", back_populates="user")


class Query(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    question = Column(String)
    answer = Column(String)
    cluster = Column(Integer)
    latency_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    #relation many queries ---> 1 user
    user = relationship("User",back_populates= "queries")
