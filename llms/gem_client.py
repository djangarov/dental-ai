import os
from google import genai
from google.genai import types

from llms import ClientInterface


MODEL = 'gemini-2.5-flash'

class GemClient(ClientInterface):
    def __init__(self, model: str = MODEL, api_key: str = None) -> None:
        self.model = model
        api_key = api_key or os.getenv('GEMINI_API_KEY')

        if not api_key:
            raise ValueError('GEMINI_API_KEY not found.')

        self.client = genai.Client(api_key=api_key)

    def chat(self, message: str) -> None:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=message,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )

            print(response.text)
        except Exception as e:
            print(f"An error occurred during gemini chat: {e}")
            raise

    def stream_chat(self, message: str) -> None:
        try:
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=[message]
            )

            self._handle_stream(stream)
        except Exception as e:
            print(f"An error occurred during gemini stream chat: {e}")
            raise

    def _handle_stream(self, stream: types.GenerateContentResponse) -> None:
        for chunk in stream:
            print(chunk.text, end='', flush=True)

