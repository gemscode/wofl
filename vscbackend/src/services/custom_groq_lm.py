from dspy import BaseLM
from groq import Groq
from typing import Tuple

class GroqDSPyLM(BaseLM):
    def __init__(self, model, api_key=None):
        super().__init__(model=model)
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024)
        )
        return completion.choices[0].message.content

    def loglikelihood(self, context: str, continuation: str) -> Tuple[float, bool]:
        # Return dummy value and success flag
        print("loglikelihood() was called.")
        return 0.0, True

    def decode(self, output):
        # Groq returns strings already — just return as-is
        print("decode() called")  # debug
        return output

    def logprobs(self, prompt: str):
        print("logprobs() called (stubbed)")
        return [] 


