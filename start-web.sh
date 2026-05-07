#!/bin/bash
set -e

python -c 'from src.db.database import init_db; init_db()'
python -m src.predict
coverage run -a -m src.unit_tests.test_endpoints
coverage report -m || true
python -m main