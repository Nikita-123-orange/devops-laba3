#!/bin/sh

export VAULT_ADDR=http://vault:8200
export VAULT_TOKEN=root-token

echo "Waiting for Vault..."

sleep 15

vault status

vault secrets enable -path=secret kv-v2 || true

vault kv put secret/db \
    username=postgres \
    password=admin \
    host=db \
    port=5432 \
    database=mydb

echo "Vault initialized"