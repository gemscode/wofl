#!/usr/bin/env python3
"""
Query R&W AI Companion information from Cassandra and Elasticsearch.
"""
import os
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from elasticsearch import Elasticsearch
from tabulate import tabulate

AGENT_NAME = "R&W AI Companion"

def get_cassandra_session():
    """Connect to Cassandra with credentials if provided"""
    host = os.getenv("CASSANDRA_HOST", "127.0.0.1")
    keyspace = os.getenv("CASSANDRA_KEYSPACE", "rw_agent")
    user = os.getenv("CASSANDRA_USER")
    pw = os.getenv("CASSANDRA_PASS")
    
    auth_provider = PlainTextAuthProvider(username=user, password=pw) if user and pw else None
    cluster = Cluster([host], auth_provider=auth_provider, protocol_version=4)
    return cluster.connect(keyspace)

def get_elasticsearch_client():
    """Connect to Elasticsearch with SSL verification disabled"""
    return Elasticsearch([os.getenv("ELASTICSEARCH_URL", "http://10.4.96.3:9200")], verify_certs=False)

def print_table(title, data, headers):
    print(f"\n{title}")
    print(tabulate(data, headers=headers, tablefmt='fancy_grid'))

def query_agent_info():
    print("🤖 R&W AI Companion Information")

    # Query Cassandra
    cass_session = get_cassandra_session()
    cass_result = cass_session.execute(
        "SELECT * FROM agent_registry WHERE name = %s ALLOW FILTERING", [AGENT_NAME]
    )
    
    if not cass_result.current_rows:
        print(f"❌ Agent '{AGENT_NAME}' not found in Cassandra.")
        return
        
    cass_data = cass_result.one()
    print_table("🗄️  Cassandra Agent Registry", [
        ["Agent ID", str(cass_data.agent_id)],
        ["Name", cass_data.name],
        ["Description", cass_data.description],
        ["Version", cass_data.version],
        ["Status", cass_data.status],
        ["Created", cass_data.created_at],
        ["Updated", cass_data.updated_at]
    ], ["Field", "Value"])

    # Query Elasticsearch by document ID
    es = get_elasticsearch_client()
    try:
        es_doc = es.get(
            index="agent_registry", 
            id=str(cass_data.agent_id)
        )
        source = es_doc['_source']
        print_table("🔎 Elasticsearch Agent Record", [
            ["Document ID", es_doc['_id']],
            ["Name", source.get('name')],
            ["Description", source.get('description')],
            ["Version", source.get('version')],
            ["Last Updated", source.get('last_updated')]
        ], ["Field", "Value"])
    except Exception as e:
        print(f"⚠️  Elasticsearch error: {str(e)}")

if __name__ == "__main__":
    query_agent_info()
