# backend/app/db_models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DB_PATH = os.path.join("data", "uttam_history.sqlite3")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class PredictionHistory(Base):
    __tablename__ = "predictions_history"
    id = Column(Integer, primary_key=True, index=True)
    zone_id = Column(String)
    generated_at = Column(DateTime, default=datetime.utcnow)
    saved_at = Column(DateTime, default=datetime.utcnow)
    risk_score = Column(Float)
    eta_minutes = Column(Float)
    alert_level = Column(String)
    reason = Column(String)
    raw_json = Column(String)

def init_db():
    Base.metadata.create_all(bind=engine)
