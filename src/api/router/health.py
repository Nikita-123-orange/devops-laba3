from typing import Any
from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check() -> dict[str, Any]:
    """Проверка здоровья приложения.
    
    Returns:
        dict: Статус приложения
    """
    return {"status": "healthy", "service": "ml-prediction-api"}
