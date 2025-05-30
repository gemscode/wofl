from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy

def get_cassandra_session():
    # Configure connection parameters without authentication
    cluster = Cluster(
        contact_points=['127.0.0.1'],
        load_balancing_policy=DCAwareRoundRobinPolicy(
            local_dc='datacenter1'  # Match your Cassandra setup
        ),
        protocol_version=4          # For Cassandra 3.11.x
    )

    session = cluster.connect()
    
    # Initialize schema (unchanged)
    session.execute(
        "CREATE KEYSPACE IF NOT EXISTS auth_system WITH replication = "
        "{'class': 'SimpleStrategy', 'replication_factor': 1}"
    )
    session.set_keyspace('auth_system')
    
    session.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "email text PRIMARY KEY, "
        "password text, "
        "created_at timestamp)"
    )
    
    return session

