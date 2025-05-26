from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

def get_cassandra_session():
    cluster = Cluster(['127.0.0.1'])
    session = cluster.connect()
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

