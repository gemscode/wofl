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
            system_prompt = (
                "You are a Python code generator. "
                "Return ONLY valid Python code that solves the user's request. "
                "DO NOT include markdown formatting (no triple backticks), comments, or any explanation. "
                "If you cannot generate code, return exactly:\n"
                "# I do not understand the request please provide information to generate python code"
            )

            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=0.1
            )
            response = completion.choices[0].message.content.strip()

            print(response)
            # Fallback: If the response is empty or not Python code, return the fallback message
            if not response or not any(keyword in response for keyword in ['def ', 'import ', 'print(', 'class ']):
                return "# I do not understand the request please provide information to generate python code"

            return response

        except Exception:
            return "# I do not understand the request please provide information to generate python code"

# Example for future OpenAI implementation
# class OpenAIService(LLMProvider): ...

