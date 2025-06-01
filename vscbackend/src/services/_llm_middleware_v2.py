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
from langchain_core.runnables import RunnablePassthrough
import dotenv

logger = logging.getLogger(__name__)

class LLMMiddlewareV2:
    def __init__(self):
        self._init_llm()
        self.es = Elasticsearch(os.getenv("ELASTICSEARCH_URL"))
        self.index_name = "chat_history_v2"
        self._ensure_es_index()

    def _init_llm(self):
        """Initialize Claude 3 with indentation-aware settings"""
        self.llm = ChatAnthropic(
            model="claude-3-opus-20240229",
            temperature=0.0,
            max_tokens=4000,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        self.prompt_template = ChatPromptTemplate.from_template("""
[System]
You are a Python specialist. Follow these rules:
1. Maintain 4-space indentation
2. No markdown formatting
3. Only return code
4. Preserve existing functionality

[Existing Code]
{context}

[New Requirement]
{input}

Return ONLY the modified Python code.
""")
        
        self.chain = (
            RunnablePassthrough.assign(
                context=lambda x: self._get_conversation_context(x["session_id"]),
                input=lambda x: x["input"]
            )
            | self.prompt_template
            | self.llm
        )

    def _ensure_es_index(self):
        """Create Elasticsearch index if missing"""
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
        """Retrieve valid code context from Elasticsearch"""
        try:
            response = self.es.search(
                index=self.index_name,
                body={
                    "query": {"term": {"session_id.keyword": session_id}},
                    "sort": [{"timestamp": "desc"}],
                    "size": 1
                }
            )
            if response["hits"]["total"]["value"] > 0:
                latest_code = response["hits"]["hits"][0]["_source"]["content"]
                if self._validate_python_syntax(latest_code):
                    return latest_code
            return ""
        except Exception as e:
            logger.error("Context retrieval error: %s", str(e))
            return ""

    def _validate_python_syntax(self, code: str) -> bool:
        """Validate Python syntax with indentation check"""
        try:
            compile(code, "<string>", "exec")
            return True
        except IndentationError as e:
            logger.warning("Indentation error in stored code: %s", str(e))
            return False
        except SyntaxError as e:
            logger.warning("Syntax error in stored code: %s", str(e))
            return False

    def _store_message(self, session_id: str, role: str, content: str):
        """Store message with indentation preservation"""
        try:
            self.es.index(
                index=self.index_name,
                document={
                    "session_id": session_id,
                    "role": role,
                    "content": self._sanitize_code(content),
                    "timestamp": datetime.utcnow()
                }
            )
        except Exception as e:
            logger.error("Message storage failed: %s", str(e))

    def generate_response(self, prompt: str, session_id: str = None) -> Tuple[str, str]:
        """Generate and store properly indented code"""
        session_id = session_id or str(uuid.uuid4())
        try:
            result = self.chain.invoke({"input": prompt, "session_id": session_id})
            code = self._sanitize_code(result.content)
            self._store_message(session_id, "user", prompt)
            self._store_message(session_id, "assistant", code)
            return code, session_id
        except Exception as e:
            logger.error("Generation failed: %s", str(e))
            return "# Error: Unable to generate code", session_id

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
        "Write a Python function to calculate Fibonacci numbers",
        "Now add memoization to optimize it",
        "Add type hints and docstring"
    ]
    
    current_session = None
    for query in test_queries:
        code, session = middleware.generate_response(query, current_session)
        current_session = session
        print(f"\nQuery: {query}")
        print(f"Session ID: {session}")
        print(f"Generated Code:\n{code}\n{'='*40}")

