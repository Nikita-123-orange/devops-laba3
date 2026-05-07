#!/bin/sh

echo 'Waiting for Vault...'
while ! vault status >/dev/null 2>&1; do sleep 2; done

echo 'Vault is ready. Initializing secrets...'
vault secrets enable -path=secret kv-v2 || true

vault kv put secret/db \
    username="${POSTGRES_USER}" \
    password="${POSTGRES_PASSWORD}" \
    host="${POSTGRES_HOST}" \
    port="${POSTGRES_PORT}" \
    database="${POSTGRES_DB}"

echo 'Vault initialized successfully'
