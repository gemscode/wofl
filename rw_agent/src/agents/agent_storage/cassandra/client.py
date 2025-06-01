from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy
from typing import Dict, List, Any, Optional
import uuid
import logging

logger = logging.getLogger(__name__)

class CassandraClient:
    def __init__(self, hosts: List[str], keyspace: str, username: str = None, password: str = None):
        self.keyspace = keyspace
        self.session = None
        self._connect(hosts, username, password)
    
    def _connect(self, hosts: List[str], username: str = None, password: str = None):
        """Establish Cassandra connection"""
        try:
            auth_provider = None
            if username and password:
                auth_provider = PlainTextAuthProvider(username=username, password=password)
            
            cluster = Cluster(
                hosts,
                auth_provider=auth_provider,
                load_balancing_policy=DCAwareRoundRobinPolicy()
            )
            
            self.session = cluster.connect(self.keyspace)
            logger.info(f"Connected to Cassandra keyspace: {self.keyspace}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Cassandra: {e}")
            raise
    
    def create_record(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new record"""
        try:
            # Generate UUID if not provided
            if 'id' not in data:
                data['id'] = uuid.uuid4()
            
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['%s'] * len(data))
            
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            self.session.execute(query, list(data.values()))
            
            return {"success": True, "id": str(data['id'])}
            
        except Exception as e:
            logger.error(f"Failed to create record: {e}")
            return {"success": False, "error": str(e)}
    
    def read_record(self, table: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Read records with optional filters"""
        try:
            query = f"SELECT * FROM {table}"
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(f"{key} = %s")
                    params.append(value)
                query += f" WHERE {' AND '.join(conditions)}"
            
            result = self.session.execute(query, params)
            records = []
            
            for row in result:
                record = {}
                for column in row._fields:
                    record[column] = getattr(row, column)
                records.append(record)
            
            return {"success": True, "data": records}
            
        except Exception as e:
            logger.error(f"Failed to read records: {e}")
            return {"success": False, "error": str(e)}
    
    def update_record(self, table: str, data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Update records matching filters"""
        try:
            set_clause = ', '.join([f"{key} = %s" for key in data.keys()])
            where_clause = ' AND '.join([f"{key} = %s" for key in filters.keys()])
            
            query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
            params = list(data.values()) + list(filters.values())
            
            self.session.execute(query, params)
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Failed to update record: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_record(self, table: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Delete records matching filters"""
        try:
            where_clause = ' AND '.join([f"{key} = %s" for key in filters.keys()])
            query = f"DELETE FROM {table} WHERE {where_clause}"
            
            self.session.execute(query, list(filters.values()))
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Failed to delete record: {e}")
            return {"success": False, "error": str(e)}

