# services/llm_middleware.py
from abc import ABC, abstractmethod
from groq import Groq
import os
import uuid
import logging
from elasticsearch import Elasticsearch, BadRequestError
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ElasticsearchChatMessageHistory

# Configure logging
logger = logging.getLogger(__name__)

class LLMMiddleware:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
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

    def generate_response(self, prompt: str, model: str = "llama3-70b-8192", **kwargs) -> tuple:
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
            response = self._call_groq(augmented_prompt, model)
            
            # Store interaction
            message_history.add_user_message(prompt)
            message_history.add_ai_message(response)
            
            return response, thread_id

        except Exception as e:
            logger.error(f"Request failed: {str(e)}", exc_info=True)
            return "# Error processing request", "error"

    def _call_groq(self, prompt: str, model: str) -> str:
        """Execute Groq API call with strict code-only output"""
        try:
            completion = self.client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": (
                        "You are a Python code generator. Return ONLY valid Python code.\n"
                        "DO NOT include:\n"
                        "- Markdown formatting (no ``````)\n"
                        "- Comments or explanations\n"
                        "- Example usage\n"
                        "If unclear, return exactly:\n"
                        "# I do not understand the request please provide information to generate python code"
                    )
                }, {
                    "role": "user",
                    "content": prompt
                }],
                model=model,
                temperature=0.0  # Fully deterministic
            )
            response = completion.choices[0].message.content.strip()
            
            # Remove any accidental markdown
            response = response.replace('``````', '').strip()
            return response
        except Exception as e:
            logger.error(f"Groq API failure: {str(e)}")
            return "# Error generating response"

