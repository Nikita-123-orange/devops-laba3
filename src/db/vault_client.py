import os

import hvac


VAULT_ADDR = os.getenv("VAULT_ADDR", "http://vault:8200")
VAULT_TOKEN = os.getenv("VAULT_TOKEN", "root-token")


client = hvac.Client(
    url=VAULT_ADDR,
    token=VAULT_TOKEN
)


def get_db_credentials():
    secret = client.secrets.kv.v2.read_secret_version(
        path="db",
        mount_point="secret"
    )

    data = secret["data"]["data"]

    return {
        "user": data["POSTGRES_USER"],
        "password": data["POSTGRES_PASSWORD"],
        "db": data["POSTGRES_DB"],
        "host": data["POSTGRES_HOST"],
        "port": data["POSTGRES_PORT"]
    }