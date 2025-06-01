from elasticsearch import Elasticsearch
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ElasticsearchClient:
    def __init__(self, hosts: List[str], auth: Optional[Dict[str, str]] = None):
        self.client = None
        self._connect(hosts, auth)
    
    def _connect(self, hosts: List[str], auth: Optional[Dict[str, str]] = None):
        """Establish Elasticsearch connection"""
        try:
            config = {"hosts": hosts}
            
            if auth:
                if auth.get('type') == 'basic':
                    config['basic_auth'] = (auth['username'], auth['password'])
                elif auth.get('type') == 'api_key':
                    config['api_key'] = auth['api_key']
            
            self.client = Elasticsearch(**config)
            
            # Test connection
            if self.client.ping():
                logger.info("Connected to Elasticsearch cluster")
            else:
                raise ConnectionError("Failed to ping Elasticsearch")
                
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
    
    def index_document(self, index: str, document: Dict[str, Any], doc_id: str = None) -> Dict[str, Any]:
        """Index a document"""
        try:
            result = self.client.index(
                index=index,
                body=document,
                id=doc_id
            )
            return {"success": True, "id": result['_id'], "result": result['result']}
            
        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            return {"success": False, "error": str(e)}
    
    def search(self, index: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Search documents"""
        try:
            result = self.client.search(
                index=index,
                body={"query": query}
            )
            
            hits = []
            for hit in result['hits']['hits']:
                hits.append({
                    "id": hit['_id'],
                    "score": hit['_score'],
                    "source": hit['_source']
                })
            
            return {
                "success": True,
                "total": result['hits']['total']['value'],
                "hits": hits
            }
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_document(self, index: str, doc_id: str) -> Dict[str, Any]:
        """Delete a document"""
        try:
            result = self.client.delete(index=index, id=doc_id)
            return {"success": True, "result": result['result']}
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return {"success": False, "error": str(e)}
    
    def create_index(self, index: str, mapping: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create an index with optional mapping"""
        try:
            body = {}
            if mapping:
                body['mappings'] = mapping
            
            result = self.client.indices.create(index=index, body=body)
            return {"success": True, "acknowledged": result['acknowledged']}
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return {"success": False, "error": str(e)}

