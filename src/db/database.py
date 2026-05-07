import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Пытаемся получить полный URL, либо собираем из компонентов
from src.db.vault_client import get_db_credentials

creds = get_db_credentials()

POSTGRES_USER = creds["user"]
POSTGRES_PASSWORD = creds["password"]
POSTGRES_DB = creds["database"]
POSTGRES_HOST = creds["host"]
POSTGRES_PORT = creds["port"]
DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
    f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from src.db import models
    Base.metadata.create_all(bind=engine)