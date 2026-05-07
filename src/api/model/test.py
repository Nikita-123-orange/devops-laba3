from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """Модель ответа предсказания."""
    status: str
    datetime: str
    model: str
    message: str = ""
    scores: dict


class TestResponse(BaseModel):
    """Модель ответа тестирования."""
    status: str
    datetime: str
    model: str
    test_type: str
    scores: dict