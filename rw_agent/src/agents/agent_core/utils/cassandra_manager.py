from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import os

class CassandraManager:
    def __init__(self):
        self.cluster = None
        self._session = None  # Changed to protected attribute
        self.keyspace = os.getenv("CASSANDRA_KEYSPACE", "rw_agent")
        self._connect()

    def _connect(self):
        """Establish Cassandra connection"""
        cassandra_host = os.getenv("CASSANDRA_HOST", "localhost")
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
            auth_provider=auth_provider,
            protocol_version=4
        )
        
        self._session = self.cluster.connect(self.keyspace)  # Assign to _session

    @property
    def session(self):
        """Get Cassandra session (read-only property)"""
        return self._session

    def close(self):
        """Close connection"""
        if self.cluster:
            self.cluster.shutdown()
            self.cluster = None
            self._session = None  # Update _session directly

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

