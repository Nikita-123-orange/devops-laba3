from fastapi import FastAPI
import uvicorn

from src.api.router.health import router as health_router
from src.api.router.test import router as test_router

app = FastAPI(
    title="ML Prediction API",
    description="API для проверки здоровья приложения и тестирования ML моделей",
    version="2.0.0"
)

app.include_router(health_router)
app.include_router(test_router)


@app.get("/")
async def root():
    """Главная страница API."""
    return {
        "message": "Welcome to ML Prediction API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "test_smoke": "/test/smoke",
            "test_func": "/test/func",
            "predict": "/test/predict"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
