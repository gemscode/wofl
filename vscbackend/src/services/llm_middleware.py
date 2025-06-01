# services/llm_middleware.py
from abc import ABC, abstractmethod
import os
import uuid
import logging
from elasticsearch import Elasticsearch, BadRequestError
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ElasticsearchChatMessageHistory
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logger = logging.getLogger(__name__)

class LLMMiddleware:
    def __init__(self):
        self.client = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0.0,
            max_tokens=4000,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.es = Elasticsearch(os.getenv("ELASTICSEARCH_URL"))
        self.index_name = "chat_history"
        self._initialize_elasticsearch()

    def _initialize_elasticsearch(self):
        """Create Elasticsearch index with LangChain-compatible mapping"""
        try:
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(
                    index=self.index_name,
                    body={
                        "mappings": {
                            "properties": {
                                "session_id": {"type": "keyword"},
                                "type": {"type": "keyword"},  # 'human' or 'ai'
                                "content": {"type": "text"},
                                "created_at": {"type": "date"},  # Required by LangChain
                                "timestamp": {"type": "date"}    # Your existing field
                            }
                        }
                    }
                )
                logger.info("Created new Elasticsearch index with correct mapping")
        except Exception as e:
            logger.error(f"Elasticsearch initialization failed: {str(e)}")
            raise

    def _get_message_history(self, thread_id):
        """Get LangChain message history with proper error handling"""
        try:
            return ElasticsearchChatMessageHistory(
                index=self.index_name,
                session_id=thread_id,
                es_connection=self.es
            )
        except Exception as e:
            logger.error(f"Message history error: {str(e)}")
            raise

    def generate_response(self, prompt: str, model: str = "claude-3-5-sonnet-20240620", **kwargs) -> tuple:
        try:
            # Thread management
            thread_id = kwargs.get("thread_id")
            new_thread = kwargs.get("new_thread", False)
            
            if new_thread or not thread_id:
                thread_id = str(uuid.uuid4())
                logger.info(f"Starting new thread: {thread_id}")

            # Initialize message history
            message_history = self._get_message_history(thread_id)
            
            # Clear history if new thread
            if new_thread:
                message_history.clear()
                logger.debug(f"Cleared history for new thread: {thread_id}")

            # Build context from previous messages
            context = "\n".join(
                f"{msg.type.capitalize()}: {msg.content}"
                for msg in message_history.messages[-3:]  # Last 3 exchanges
            )

            # Create augmented prompt
            augmented_prompt = f"Context:\n{context}\n\nNew Query: {prompt}" if context else prompt
            
            # Generate response
            response = self._call_anthropic(augmented_prompt)
            
            # Store interaction
            message_history.add_user_message(prompt)
            message_history.add_ai_message(response)
            
            return response, thread_id

        except Exception as e:
            logger.error(f"Request failed: {str(e)}", exc_info=True)
            return "# Error processing request", "error"

    def _call_anthropic(self, prompt: str) -> str:
        """Execute Anthropic API call with strict code-only output"""
        try:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """
                You are a Python code generator. Return ONLY valid Python code.
                DO NOT include:
                - Markdown formatting (no ``````)
                - Comments or explanations
                - Example usage
                If unclear, return exactly:
                # I do not understand the request please provide information to generate python code
                """),
                ("human", "{input}")
            ])
            
            chain = prompt_template | self.client
            result = chain.invoke({"input": prompt})
            
            response = result.content.strip()
            
            # Remove any accidental markdown
            response = response.replace('``````', '').strip()
            return response
        except Exception as e:
            logger.error(f"Anthropic API failure: {str(e)}")
            return "# Error generating response"

