from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from uuid import UUID
import os

class CassandraManager:
    def __init__(self):
        auth = PlainTextAuthProvider(
            username=os.getenv("CASSANDRA_USER"),
            password=os.getenv("CASSANDRA_PASS")
        )
        self.cluster = Cluster(
            [os.getenv("CASSANDRA_HOST")], 
            auth_provider=auth
        )
        self.session = self.cluster.connect(os.getenv("CASSANDRA_KEYSPACE"))

    def create_project(self, user_id: UUID, project_name: str, agents: list):
        project_id = UUID(os.urandom(16).hex())
        
        # Insert project
        self.session.execute("""
            INSERT INTO projects 
            (project_id, user_id, name, created_at, status)
            VALUES (%s, %s, %s, toTimestamp(now()), 'active')
        """, (project_id, user_id, project_name))
        
        # Insert agents
        for agent in agents:
            is_required = agent in ['agent_core', 'agent_security']
            self.session.execute("""
                INSERT INTO project_agents 
                (project_id, agent_name, is_required, is_selected)
                VALUES (%s, %s, %s, %s)
            """, (project_id, agent, is_required, True))
        
        return project_id

    # Add other CRUD operations...

