from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (3 levels up from utils directory)
ROOT_DIR = Path(__file__).resolve().parents[2]  # Adjusted path for utils location
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

class CassandraManager:
    def __init__(self):
        self.cluster = None
        self._session = None
        self.keyspace = os.getenv("CASSANDRA_KEYSPACE", "rw_agent")
        self._connect()

    def _connect(self):
        """Establish Cassandra connection with .env config"""
        cassandra_host = os.getenv("CASSANDRA_HOST", "localhost")
        cassandra_port = int(os.getenv("CASSANDRA_PORT", "9042"))  # Added port handling
        cassandra_user = os.getenv("CASSANDRA_USER")
        cassandra_pass = os.getenv("CASSANDRA_PASS")

        auth_provider = None
        if cassandra_user and cassandra_pass:
            auth_provider = PlainTextAuthProvider(
                username=cassandra_user,
                password=cassandra_pass
            )

        self.cluster = Cluster(
            [cassandra_host],
            port=cassandra_port,  # Added port parameter
            auth_provider=auth_provider,
            protocol_version=4
        )

        self._session = self.cluster.connect(self.keyspace)

    @property
    def session(self):
        return self._session

    def close(self):
        if self.cluster:
            self.cluster.shutdown()
            self.cluster = None
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

