#!/bin/sh

export VAULT_ADDR=http://vault:8200
export VAULT_TOKEN=root-token

echo "Waiting for Vault..."

sleep 15

vault secrets enable -path=secret kv-v2 || true

vault kv put secret/db \
    POSTGRES_USER=app_user \
    POSTGRES_PASSWORD=app_password \
    POSTGRES_DB=ml_db \
    POSTGRES_HOST=db \
    POSTGRES_PORT=5432

echo "Vault initialized"