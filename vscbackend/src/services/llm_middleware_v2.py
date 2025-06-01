# services/llm_middleware_v2.py
import os
import uuid
import logging
import re
from elasticsearch import Elasticsearch
from datetime import datetime
from typing import Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
import dotenv

logger = logging.getLogger(__name__)

class LLMMiddlewareV2:
    def __init__(self):
        self._init_llm()
        self.es = Elasticsearch(os.getenv("ELASTICSEARCH_URL"))
        self.index_name = "chat_history_v2"
        self._ensure_es_index()

    def _init_llm(self):
        """Initialize Claude 3.5 Sonnet with focused instructions"""
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0.1,
            max_tokens=4000,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.prompt_template = ChatPromptTemplate.from_template("""
[System]
You are a Python code maintenance expert. Follow these rules:
1. MODIFY THE EXISTING CODE below to implement the new requirement
2. PRESERVE the original function name and parameters
3. RETURN ONLY THE MODIFIED CODE without explanations
4. USE 4-space indentation consistently

[Existing Code]
{context}

[New Requirement]
{input}
""")

    def _ensure_es_index(self):
        """Create Elasticsearch index with proper mapping"""
        try:
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(
                    index=self.index_name,
                    body={
                        "mappings": {
                            "properties": {
                                "session_id": {"type": "keyword"},
                                "role": {"type": "keyword"},
                                "content": {"type": "text"},
                                "timestamp": {"type": "date"}
                            }
                        }
                    }
                )
                logger.info("Created Elasticsearch index: %s", self.index_name)
        except Exception as e:
            logger.error("Index creation failed: %s", str(e))
            raise

    def _get_conversation_context(self, session_id: str) -> str:
        """Retrieve latest valid code from the session"""
        try:
            response = self.es.search(
                index=self.index_name,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"session_id.keyword": session_id}},
                                {"term": {"role": "assistant"}}
                            ]
                        }
                    },
                    "sort": [{"timestamp": "desc"}],
                    "size": 1
                }
            )
            if response["hits"]["total"]["value"] > 0:
                return response["hits"]["hits"][0]["_source"]["content"]
            return ""
        except Exception as e:
            logger.error("Context retrieval error: %s", str(e))
            return ""

    def _store_message(self, session_id: str, role: str, content: str):
        """Store message with validation"""
        try:
            self.es.index(
                index=self.index_name,
                document={
                    "session_id": session_id,
                    "role": role,
                    "content": content.strip(),
                    "timestamp": datetime.utcnow()
                }
            )
        except Exception as e:
            logger.error("Message storage failed: %s", str(e))

    def generate_response(self, prompt: str, session_id: str = None) -> Tuple[str, str]:
        """Generate context-aware code response"""
        session_id = session_id or str(uuid.uuid4())
        try:
            context = self._get_conversation_context(session_id)
            logger.info("Using context:\n%s", context)
            
            result = self.llm.invoke(
                self.prompt_template.format(context=context, input=prompt)
            )
            
            code = self._sanitize_code(result.content)
            self._validate_code(code, context)
            
            self._store_message(session_id, "user", prompt)
            self._store_message(session_id, "assistant", code)
            
            return code, session_id

        except Exception as e:
            logger.error("Generation failed: %s", str(e))
            return "# Error: Unable to generate valid code", session_id

    def _sanitize_code(self, code: str) -> str:
        """Extract code while preserving indentation"""
        # Match Python code blocks with indentation
        code_match = re.search(
            r'``````',
            code,
            re.DOTALL
        )
        if code_match:
            return code_match.group(1).strip()

        # Remove any remaining markdown
        clean_code = re.sub(r'``````', '', code, flags=re.DOTALL)

        # Preserve existing indentation
        return '\n'.join([
            line.rstrip() for line in clean_code.split('\n')
            if line.strip() and not line.strip().startswith(('Here is', 'def test', 'def count'))
        ]).strip()

    def _validate_code(self, code: str, original_context: str):
        """Validate code preserves core functionality"""
        # Check if main function exists in both context and new code
        context_func = re.search(r'def\s+(\w+)\s*\(', original_context)
        new_func = re.search(r'def\s+(\w+)\s*\(', code)
        
        if context_func and (not new_func or new_func.group(1) != context_func.group(1)):
            raise ValueError("Core function changed or removed")
            
        # Basic syntax check
        try:
            compile(code, "<string>", "exec")
        except Exception as e:
            raise ValueError(f"Invalid Python syntax: {str(e)}")

if __name__ == "__main__":
    dotenv.load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Missing ANTHROPIC_API_KEY in environment")
    
    middleware = LLMMiddlewareV2()
    
    test_queries = [
        "Write a recursive Fibonacci function",
        "Add error handling for negative inputs",
        "Optimize with memoization",
        "Add type hints and docstring"
    ]
    
    current_session = None
    for query in test_queries:
        code, session = middleware.generate_response(query, current_session)
        current_session = session
        print(f"\nQuery: {query}")
        print(f"Session ID: {session}")
        print(f"Generated Code:\n{code}\n{'='*40}")

