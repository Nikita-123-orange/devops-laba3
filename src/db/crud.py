from sqlalchemy.orm import Session
from src.db import models

def create_prediction(db: Session, features, predicted_class: int, confidence: float):
    """
    Сохраняет новое предсказание в базу данных.
    
    Args:
        db: сессия SQLAlchemy
        features: список признаков (преобразуется в JSON)
        predicted_class: целочисленный предсказанный класс
        confidence: уверенность модели (0..1)
    
    Returns:
        созданный объект Prediction
    """
    db_prediction = models.Prediction(
        features=features,
        predicted_class=predicted_class,
        confidence=confidence
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction