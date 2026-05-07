FROM python:3.14-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app 

# Сначала копируем только requirements.txt (для кэширования слоя с зависимостями)
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь остальной код
COPY . .

CMD ["sh", "-c", "python -c 'from src.db.database import init_db; init_db()' && python -m main]