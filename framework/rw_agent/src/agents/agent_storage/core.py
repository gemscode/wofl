from typing import Dict, Any, Optional
from .cassandra.client import CassandraClient
from .elasticsearch.client import ElasticsearchClient
import logging

logger = logging.getLogger(__name__)

class StorageAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cassandra_client: Optional[CassandraClient] = None
        self.elasticsearch_client: Optional[ElasticsearchClient] = None
        self.initialize()
    
    def initialize(self):
        """Initialize storage clients based on configuration"""
        try:
            # Initialize Cassandra if configured
            if self.config.get('cassandra', {}).get('enabled', False):
                self.cassandra_client = CassandraClient(
                    hosts=self.config['cassandra']['hosts'],
                    keyspace=self.config['cassandra']['keyspace'],
                    username=self.config['cassandra'].get('username'),
                    password=self.config['cassandra'].get('password')
                )
                logger.info("Cassandra client initialized")
            
            # Initialize Elasticsearch if configured
            if self.config.get('elasticsearch', {}).get('enabled', False):
                self.elasticsearch_client = ElasticsearchClient(
                    hosts=self.config['elasticsearch']['hosts'],
                    auth=self.config['elasticsearch'].get('auth')
                )
                logger.info("Elasticsearch client initialized")
                
        except Exception as e:
            logger.error(f"Storage agent initialization failed: {e}")
            raise
    
    def validate_config(self) -> bool:
        """Validate storage configuration"""
        required_fields = ['cassandra', 'elasticsearch']
        return all(field in self.config for field in required_fields)
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute storage operations"""
        operation = task.get('operation')
        target = task.get('target', 'cassandra')
        
        if target == 'cassandra' and self.cassandra_client:
            return self._execute_cassandra_operation(operation, task)
        elif target == 'elasticsearch' and self.elasticsearch_client:
            return self._execute_elasticsearch_operation(operation, task)
        else:
            raise ValueError(f"Invalid target: {target}")
    
    def _execute_cassandra_operation(self, operation: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Cassandra operations"""
        if operation == 'create':
            return self.cassandra_client.create_record(
                table=task['table'],
                data=task['data']
            )
        elif operation == 'read':
            return self.cassandra_client.read_record(
                table=task['table'],
                filters=task.get('filters', {})
            )
        elif operation == 'update':
            return self.cassandra_client.update_record(
                table=task['table'],
                data=task['data'],
                filters=task['filters']
            )
        elif operation == 'delete':
            return self.cassandra_client.delete_record(
                table=task['table'],
                filters=task['filters']
            )
        else:
            raise ValueError(f"Unknown Cassandra operation: {operation}")
    
    def _execute_elasticsearch_operation(self, operation: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Elasticsearch operations"""
        if operation == 'index':
            return self.elasticsearch_client.index_document(
                index=task['index'],
                document=task['document'],
                doc_id=task.get('doc_id')
            )
        elif operation == 'search':
            return self.elasticsearch_client.search(
                index=task['index'],
                query=task['query']
            )
        elif operation == 'delete':
            return self.elasticsearch_client.delete_document(
                index=task['index'],
                doc_id=task['doc_id']
            )
        else:
            raise ValueError(f"Unknown Elasticsearch operation: {operation}")

