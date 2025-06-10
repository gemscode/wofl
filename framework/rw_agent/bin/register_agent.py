#!/usr/bin/env python3
"""
R&W AI Companion Initial Registration (Idempotent First-Run Setup)
"""
import os
import uuid
from datetime import datetime
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy
from elasticsearch import Elasticsearch
from pathlib import Path
from dotenv import load_dotenv

# --------------------------------------------------
# Load environment from .env at project root
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# --------------------------------------------------
# FirstRunRegistrar
# --------------------------------------------------
class FirstRunRegistrar:
    def __init__(self):
        self.cassandra = None
        self.elastic = None
        self.agent_id = str(uuid.uuid4())
        self.agent_name = os.getenv("AGENT_NAME", "R&W AI Companion")

        self._init_cassandra()
        self._init_elasticsearch()
        self.ensure_elasticsearch_index()

    def _init_cassandra(self):
        try:
            auth_provider = None
            if os.getenv("CASSANDRA_USER") and os.getenv("CASSANDRA_PASS"):
                auth_provider = PlainTextAuthProvider(
                    os.getenv("CASSANDRA_USER"),
                    os.getenv("CASSANDRA_PASS")
                )

            cluster = Cluster(
                contact_points=[os.getenv("CASSANDRA_HOST", "127.0.0.1")],
                port=int(os.getenv("CASSANDRA_PORT", "9042")),
                auth_provider=auth_provider,
                load_balancing_policy=DCAwareRoundRobinPolicy(),
                protocol_version=4
            )

            keyspace = os.getenv("CASSANDRA_KEYSPACE", "rw_agent")
            session = cluster.connect()
            session.execute(f"USE {keyspace}")
            self.cassandra = session
            print(f"✅ Connected to Cassandra keyspace: {keyspace}")
        except Exception as e:
            print(f"❌ Cassandra init failed: {e}")
            raise

    def _init_elasticsearch(self):
        try:
            es_url = os.getenv("ELASTICSEARCH_URL", "http://127.0.0.1:9200")
            self.elastic = Elasticsearch([es_url], verify_certs=False)
            print(f"✅ Connected to Elasticsearch at {es_url}")
        except Exception as e:
            print(f"❌ Elasticsearch init failed: {e}")
            raise

    def ensure_elasticsearch_index(self):
        index_name = "agent_registry"
        try:
            if self.elastic.indices.exists(index=index_name):
                print(f"🟡 Elasticsearch index '{index_name}' already exists")
            else:
                self.elastic.indices.create(
                    index=index_name,
                    body={
                        "mappings": {
                            "properties": {
                                "agent_id": {"type": "keyword"},
                                "name": {"type": "text"},
                                "description": {"type": "text"},
                                "created_at": {"type": "date"}
                            }
                        }
                    }
                )
                print(f"✅ Elasticsearch index '{index_name}' created successfully")
        except Exception as e:
            print(f"⚠️ Elasticsearch index setup failed: {e}")

    def register_agent(self):
        agent_data = {
            "agent_id": self.agent_id,
            "name": self.agent_name,
            "description": "AI Code Generator with Enterprise capabilities",
            "version": "1.0.0",
            "status": "active",
            "created_at": datetime.utcnow()
        }

        # Cassandra Check
        cass_registered = False
        try:
            cass_result = self.cassandra.execute(
                "SELECT agent_id FROM agent_registry WHERE name=%s ALLOW FILTERING",
                [self.agent_name]
            )
            if cass_result.one():
                cass_registered = True
                print("🟡 Agent already exists in Cassandra")
        except Exception as e:
            print(f"⚠️ Cassandra lookup failed: {e}")

        # Elasticsearch Check
        es_registered = False
        try:
            es_result = self.elastic.search(index="agent_registry", body={
                "query": {"match": {"name": self.agent_name}}
            })
            if es_result["hits"]["total"]["value"] > 0:
                es_registered = True
                print("🟡 Agent already exists in Elasticsearch")
        except Exception as e:
            print(f"⚠️ Elasticsearch lookup failed: {e}")

        if cass_registered and es_registered:
            print("🟡 Skipping registration (agent exists in both Cassandra and Elasticsearch)")
            return

        if not cass_registered:
            try:
                self.cassandra.execute("""
                    INSERT INTO agent_registry (agent_id, name, description, version, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    uuid.UUID(agent_data["agent_id"]),
                    agent_data["name"],
                    agent_data["description"],
                    agent_data["version"],
                    agent_data["status"],
                    agent_data["created_at"]
                ))
                print("✅ Registered agent in Cassandra")
            except Exception as e:
                print(f"❌ Failed to register agent in Cassandra: {e}")
                raise

        if not es_registered:
            try:
                self.elastic.index(
                    index="agent_registry",
                    id=agent_data["agent_id"],
                    body={
                        "agent_id": agent_data["agent_id"],
                        "name": agent_data["name"],
                        "description": agent_data["description"],
                        "created_at": agent_data["created_at"].isoformat()
                    }
                )
                print("✅ Registered agent in Elasticsearch")
            except Exception as e:
                print(f"❌ Failed to register agent in Elasticsearch: {e}")
                raise

        print(f"✅ Agent '{self.agent_name}' registered with ID: {self.agent_id}")

# --------------------------------------------------
# Main Entry
# --------------------------------------------------
if __name__ == "__main__":
    print("🚀 Registering R&W Framework Agent...")
    try:
        registrar = FirstRunRegistrar()
        registrar.register_agent()
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        exit(1)

