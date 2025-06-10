import hashlib
from pathlib import Path
from datetime import datetime
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import BatchStatement, ConsistencyLevel
import uuid
REQUIRED_FILES = {
    "root": [
        "README.md",
        "requirements.txt",
        "setup.py",
        ".env",
        "bin/register_agent.py",
        "bin/query_agent_info.py"
    ],
    "src": [
        "src/__init__.py",
        "src/app.py",
        "src/templates/docker/Dockerfile.j2"
    ],
    "agents": [
        "src/agents/agent_core/__init__.py",
        "src/agents/agent_core/core.py",
        "src/agents/agent_ai/__init__.py",
        "src/agents/agent_ai/core.py", 
        "src/agents/agent_storage/__init__.py",
        "src/agents/agent_storage/core.py"
    ],
    "utils": [
        "src/utils/__init__.py",
        "src/utils/cassandra_manager.py",
        "src/utils/elasticsearch_manager.py"
    ]
}
class IntegrityManager:
    def __init__(self, session):
        self.session = session
        print("DEBUG: IntegrityManager initialized")

    def generate_file_hash(self, file_path: Path) -> str:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def check_sync_status(self, agent_id: str, base_path: Path) -> dict:
        print(f"DEBUG: check_sync_status({agent_id})")
        current_hashes = self._get_current_hashes(base_path)
        registered = self.session.execute(
            "SELECT key, value FROM agent_metadata WHERE agent_id = %s AND type = 'structure'",
            [uuid.UUID(agent_id)]
        )
        registered_hashes = {row.key: row.value for row in registered}
        discrepancies = []
        for path, curr_hash in current_hashes.items():
            reg_hash = registered_hashes.get(path)
            if not reg_hash:
                discrepancies.append(f"New file: {path}")
            elif reg_hash != curr_hash:
                discrepancies.append(f"Modified: {path}")
        for reg_path in registered_hashes:
            if reg_path not in current_hashes:
                discrepancies.append(f"Missing: {reg_path}")
        return {
            "is_valid": len(discrepancies) == 0,
            "last_sync": self.get_last_sync(agent_id),
            "discrepancies": discrepancies,
            "total_files": len(current_hashes)
        }

    def _get_current_hashes(self, base_path: Path) -> dict:
        print("DEBUG: _get_current_hashes()")
        hashes = {}
        
        # Process all files from REQUIRED_FILES
        for category in REQUIRED_FILES.values():
            for file_path in category:
                full_path = base_path / file_path
                if full_path.exists():
                    rel_path = str(full_path.relative_to(base_path))
                    hashes[rel_path] = self.generate_file_hash(full_path)
        return hashes

    def get_last_sync(self, agent_id: str) -> datetime:
        result = self.session.execute(
            "SELECT last_updated FROM agent_metadata WHERE agent_id = %s AND type = 'sync' AND key = 'last_sync'",
            [uuid.UUID(agent_id)]
        ).one()
        return result.last_updated if result else None

    def update_sync_status(self, agent_id: str, base_path: Path):
        print(f"DEBUG: update_sync_status({agent_id})")
        current_hashes = self._get_current_hashes(base_path)
        timestamp = datetime.now()

        try:
            print("DEBUG: Clearing existing records")
            self.session.execute(
                "DELETE FROM agent_metadata WHERE agent_id = %s AND type = 'structure'",
                (uuid.UUID(agent_id),)
            )

            print("DEBUG: Preparing batch insert")
            insert_query = """
                INSERT INTO agent_metadata 
                (agent_id, type, key, last_updated, value)
                VALUES (?, ?, ?, ?, ?)
            """
            prepared = self.session.prepare(insert_query)
            batch = BatchStatement(consistency_level=ConsistencyLevel.QUORUM)

            for path, hash_val in current_hashes.items():
                params = (
                    uuid.UUID(agent_id),
                    'structure',
                    path,
                    timestamp,
                    hash_val
                )
                print(f"DEBUG: Adding batch params: {params}")
                batch.add(prepared, params)

            print("DEBUG: Executing batch")
            self.session.execute(batch)

            print("DEBUG: Updating sync timestamp")
            self.session.execute(
                """
                INSERT INTO agent_metadata 
                (agent_id, type, key, last_updated, value)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    uuid.UUID(agent_id),
                    'sync',
                    'last_sync',
                    timestamp,
                    'true'
                )
            )

        except Exception as e:
            print(f"CRITICAL ERROR: {str(e)}")
            raise

