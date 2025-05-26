from abc import ABC, abstractmethod
from groq import Groq
import os

class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, model: str) -> str:
        pass

class GroqService(LLMProvider):
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def generate_response(self, prompt: str, model: str = "llama3-70b-8192") -> str:
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# Example for future OpenAI implementation
# class OpenAIService(LLMProvider): ...

