# ML API для Fashion MNIST (с Vault)

FastAPI + PostgreSQL + Vault: предсказание класса изображения/CSV, логирование в БД, управление секретами через HashiCorp Vault.

## Быстрый старт

```bash
pip install -r requirements.txt
python src/preprocess.py   # распаковка, split данных
python src/train.py        # обучение модели + scaler
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Управление секретами (HashiCorp Vault)

Для защиты учётных данных PostgreSQL используется Vault в dev-режиме.

**В `docker-compose.yml`** поднимаются:
- `vault` — сервер (порт 8200)
- `vault-init` — инициализирует KV-движок и записывает secrets `secret/db` (username, password, host, port, database)
- `db` — PostgreSQL, использующий те же credentials
- `web` — приложение, которое через `vault_client.py` читает секреты

**`vault_client.py`**:
```python
client = hvac.Client(url=VAULT_ADDR, token=VAULT_TOKEN)
secret = client.secrets.kv.v2.read_secret_version(path="db", mount_point="secret")
```
Данные из Vault передаются в `database.py` для построения `DATABASE_URL`.

**Переменные окружения** (в контейнере web):
- `VAULT_ADDR=http://vault:8200`
- `VAULT_TOKEN=root-token`

**Локальная разработка** (без Docker):
- Вручную запустить Vault (dev) и заполнить secrets командой:  
  `vault secrets enable -path=secret kv-v2`  
  `vault kv put secret/db username=postgres password=admin host=localhost port=5432 database=mydb`
- Установить переменные окружения `VAULT_ADDR` и `VAULT_TOKEN`

## Docker-запуск (со всеми сервисами)

```bash
docker-compose up --build
```

Vault инициализируется автоматически (сервис `vault-init`), приложение получает секреты и подключается к БД.

## Тесты и CI/CD

- Юнит-тесты: `coverage run -m src.unit_tests.test_preprocess`
- Jenkins: клон, сборка, тесты, push в Docker Hub, подпись Cosign.

Конфиг: `config.ini`, логи: `logfile.log`.