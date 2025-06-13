#!/usr/bin/env python3
"""
Query R&W AI Companion information from Cassandra and Elasticsearch.
"""
import os
from pathlib import Path
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from elasticsearch import Elasticsearch
from datetime import datetime
from dotenv import load_dotenv

# --------------------------------------------------
# Load .env from project root (3 levels up)
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

AGENT_NAME = os.getenv("AGENT_NAME", "R&W AI Companion")


def get_cassandra_session():
    host = os.getenv("CASSANDRA_HOST", "127.0.0.1")
    port = int(os.getenv("CASSANDRA_PORT", "9042"))
    keyspace = os.getenv("CASSANDRA_KEYSPACE", "rw_agent")
    user = os.getenv("CASSANDRA_USER")
    pw = os.getenv("CASSANDRA_PASS")

    print(f"📡 Cassandra Config - Host: {host}, Port: {port}, Keyspace: {keyspace}")

    auth_provider = PlainTextAuthProvider(username=user, password=pw) if user and pw else None
    cluster = Cluster([host], port=port, auth_provider=auth_provider, protocol_version=4)
    return cluster.connect(keyspace)


def get_elasticsearch_client():
    es_url = os.getenv("ELASTICSEARCH_URL", "http://127.0.0.1:9201")
    print(f"🔗 Elasticsearch URL: {es_url}")
    return Elasticsearch([es_url], verify_certs=False)


def print_info_block(title):
    print(f"\n{title}")
    print("-" * len(title))


def query_agent_info():
    print("🔍 Querying R&W AI Companion Info")

    try:
        cass_session = get_cassandra_session()
        cass_result = cass_session.execute(
            "SELECT * FROM agent_registry WHERE name = %s ALLOW FILTERING", [AGENT_NAME]
        )

        if not cass_result.current_rows:
            print(f"❌ Agent '{AGENT_NAME}' not found in Cassandra.")
            return

        cass_data = cass_result.one()

        print_info_block("🗄️  Cassandra Agent Registry")
        print(f"Agent ID     : {cass_data.agent_id}")
        print(f"Name         : {cass_data.name}")
        print(f"Description  : {cass_data.description}")
        print(f"Version      : {cass_data.version}")
        print(f"Status       : {cass_data.status}")
        print(f"Created      : {cass_data.created_at}")
        print(f"Updated      : {cass_data.updated_at}")

        es = get_elasticsearch_client()
        es_doc = es.get(index="agent_registry", id=str(cass_data.agent_id))
        source = es_doc['_source']

        print_info_block("🔎 Elasticsearch Agent Record")
        print(f"Document ID  : {es_doc['_id']}")
        print(f"Name         : {source.get('name')}")
        print(f"Description  : {source.get('description')}")
        print(f"Version      : {source.get('version')}")
        print(f"Last Updated : {source.get('last_updated')}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    query_agent_info()

