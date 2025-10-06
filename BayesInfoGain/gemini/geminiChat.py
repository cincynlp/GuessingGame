from google import genai
from google.genai import types
import time
from env import GEMINI_API_KEYS

class ChatBot:
    def __init__(self, system="", model="gpt-4o", provider="openai", generation_config = {
                    "temperature": 0.5,
                    "top_p": 0.5,
                    "system_instruction": "You are named the guesser."
                }, API_KEY=GEMINI_API_KEYS):
        if provider != "gemini":
            raise ValueError(f"Unknown provider: {provider}")
        
        self.provider = provider
        self.model = model
        self.system = system
        self.messages = []
        self.generation_config = generation_config
        self.client = genai.Client(api_key=API_KEY)

    def __call__(self, message):
        time.sleep(3)
        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(**self.generation_config),
            contents=message
        )
        return response.text.replace("\n", "")

class MultiChatBot:
    def __init__(self, system="", model="gpt-4o", provider="openai", generation_config = {
                    "temperature": 0.5,
                    "top_p": 0.5,
                    "system_instruction": "You are named the guesser."
                }, API_KEY=GEMINI_API_KEYS, sleep=1):
        if provider != "gemini":
            raise ValueError(f"Unknown provider: {provider}")
        
        self.system = system
        self.model = model
        self.provider = provider
        self.messages = []
        self.generation_config = generation_config
        self.chat = None
        self.client = genai.Client(api_key=API_KEY)
        self.ssleep = sleep
        self.chat = self.client.chats.create(model=model, config=types.GenerateContentConfig(**self.generation_config))

    def __call__(self, message):
        time.sleep(self.ssleep)
        response = self.chat.send_message(message)
        return response.text.replace("\n", "")

    def invalidQuestion(self, promptC):
        prompt = "Oracle said: This question is invalid as it violates what questions can be asked " + promptC
        return self.__call__(prompt)

    def top5(self):
        prompt = "Based on the previous questions what 5 physical objects are most likely? Why do you think these objects are most likely? Return a list of the objects, after the list put a tab then an explanation on why you think they are most likely."
        return self.__call__(prompt)