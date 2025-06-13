#!/usr/bin/env python3
import json
import os
from pathlib import Path
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from dotenv import load_dotenv

# Load .env from project root (two levels up from framework/utils/)
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

def get_cassandra_session():
    host = os.getenv("CASSANDRA_HOST", "127.0.0.1")
    port = int(os.getenv("CASSANDRA_PORT", "9042"))
    user = os.getenv("CASSANDRA_USER")
    pw = os.getenv("CASSANDRA_PASS")
    keyspace = os.getenv("CASSANDRA_KEYSPACE", "rw_agent")

    print(f"📡 Cassandra Config - Host: {host}, Port: {port}, Keyspace: {keyspace}")

    auth_provider = PlainTextAuthProvider(username=user, password=pw) if user and pw else None
    cluster = Cluster([host], port=port, auth_provider=auth_provider, protocol_version=4)
    session = cluster.connect()
    session.set_keyspace(keyspace)
    return session

def generate_metadata():
    session = get_cassandra_session()

    # Fetch the single project row
    project_row = session.execute("SELECT * FROM project_metadata LIMIT 1").one()
    if not project_row:
        print("❌ No project_metadata found.")
        return

    # Extract agent names from the dependencies map
    agent_names = sorted(project_row.dependencies.keys()) if project_row.dependencies else []

    # Build metadata dictionary
    metadata = {
        "project_id": str(project_row.project_id),
        "name": project_row.project_name or "R&W AI Companion",
        "version": "1.0.0",
        "agents": agent_names,
        "storage": {
            "cassandra": { "enabled": True },
            "elasticsearch": { "enabled": True }
        },
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-rwagent",
                "ms-agent.python"
            ]
        }
    }

    # Write to rwagent.json in framework/
    out_path = Path(__file__).resolve().parent.parent / "rwagent.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")  # Ensure newline at EOF
        f.flush()
        f.truncate()   # Truncate remaining content if rewriting old file

    print(f"✅ Created rwagent.json at {out_path}")

if __name__ == "__main__":
    try:
        generate_metadata()
    except Exception as e:
        print(f"❌ Error: {e}")

