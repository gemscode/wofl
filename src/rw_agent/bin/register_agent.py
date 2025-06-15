#!/usr/bin/env python3
"""
R&W AI Companion Initial Registration (First-Run Setup)
"""
import os
import uuid
from datetime import datetime, timezone
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy
from elasticsearch import Elasticsearch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirstRunRegistrar:
    def __init__(self):
        self.cassandra = None
        self.elastic = None
        self.agent_id = str(uuid.uuid4())
        
        # Initialize infrastructure
        self._init_cassandra()
        self._init_elasticsearch()
        
        # Create schemas
        self._create_cassandra_schema()
        self._create_elastic_indices()

    def _init_cassandra(self):
        """Initialize Cassandra connection and keyspace"""
        try:
            # Handle authentication only if credentials exist
            auth_provider = None
            if os.getenv("CASSANDRA_USER") and os.getenv("CASSANDRA_PASS"):
                auth_provider = PlainTextAuthProvider(
                    username=os.getenv("CASSANDRA_USER"),
                    password=os.getenv("CASSANDRA_PASS")
                )

            # Connect without keyspace first
            cluster = Cluster(
                contact_points=[os.getenv("CASSANDRA_HOST", "127.0.0.1")],
                auth_provider=auth_provider,
                load_balancing_policy=DCAwareRoundRobinPolicy(),
                protocol_version=4
            )
            
            # Create keyspace if needed
            session = cluster.connect()
            keyspace = os.getenv("CASSANDRA_KEYSPACE", "rw_agent")
            session.execute(f"""
                CREATE KEYSPACE IF NOT EXISTS {keyspace}
                WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
            """)
            
            # Reconnect to keyspace
            self.cassandra = cluster.connect(keyspace)
            logger.info("Cassandra initialized successfully")

        except Exception as e:
            logger.error(f"Cassandra initialization failed: {e}")
            raise

    def _create_cassandra_schema(self):
        """Create required Cassandra tables"""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS agent_registry (
                agent_id UUID PRIMARY KEY,
                name TEXT,
                description TEXT,
                version TEXT,
                status TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_agent_name ON agent_registry (name)"
        ]
        
        for table in tables:
            self.cassandra.execute(table)
        logger.info("Cassandra tables created")

    def _init_elasticsearch(self):
        """Initialize Elasticsearch connection"""
        host = os.getenv("ELASTICSEARCH_HOST", "127.0.0.1")
        port = os.getenv("ELASTICSEARCH_HTTP_PORT", "9200")
        es_url = f"http://{host}:{port}"
        
        self.elastic = Elasticsearch(
            [es_url],
            verify_certs=False
        )
        logger.info(f"Elasticsearch connection established at {es_url}")

    def _create_elastic_indices(self):
        """Create Elasticsearch indices with mappings"""
        if not self.elastic.indices.exists(index="agent_registry"):
            self.elastic.indices.create(
                index="agent_registry",
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
            logger.info("Elasticsearch index created")

    def register_agent(self):
        """Register the agent in both databases"""
        agent_data = {
            "agent_id": self.agent_id,
            "name": "R&W AI Companion",
            "description": "AI Code Generator with Enterprise capabilities",
            "version": "1.0.0",
            "status": "active",
            "created_at": datetime.now(timezone.utc)  # Timezone-aware
        }

        # Insert into Cassandra
        self.cassandra.execute("""
            INSERT INTO agent_registry 
            (agent_id, name, description, version, status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            uuid.UUID(agent_data["agent_id"]),
            agent_data["name"],
            agent_data["description"],
            agent_data["version"],
            agent_data["status"],
            agent_data["created_at"]
        ))

        # Index in Elasticsearch
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

        logger.info(f"Agent registered with ID: {self.agent_id}")
        print("✅ First-run registration completed successfully!")

if __name__ == "__main__":
    print("🚀 Starting first-run registration...")
    try:
        registrar = FirstRunRegistrar()
        registrar.register_agent()
    except Exception as e:
        print(f"❌ Critical error during first-run setup: {e}")
        raise

