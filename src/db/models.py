from sqlalchemy import Column, Integer, Float, DateTime, JSON, ARRAY, Text
from sqlalchemy.sql import func
from src.db.database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    features = Column(Text, nullable=False)    
    predicted_class = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())