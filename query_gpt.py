# import genai
import time
import genai
import openai
import google.generativeai as genai
# Set default keys (can override later)
openai.api_key = ""
genai.configure(api_key="")

class ChatBot:
    def __init__(self, system="", model="gpt-4o", provider="openai"):
        self.provider = provider
        self.model = model
        self.system = system
        self.messages = []
        if self.provider == "openai" and system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        if self.provider == "openai":
            client = openai.OpenAI(api_key=openai.api_key)
            completion = client.chat.completions.create(
                model=self.model,
                messages=self.messages + [{"role": "user", "content": message}],
                top_p=0.5,
                timeout=10
            )
            return completion.choices[0].message.content.replace("\n", "")

        elif self.provider == "gemini":
            model = genai.GenerativeModel(
                model_name=self.model,
                generation_config={
                    # "temperature": 0.5,
                    "top_p": 0.5
                },
                system_instruction=self.system if self.system else None
            )
            response = model.generate_content(message)
            return response.text.replace("\n", "")

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

class MultiChatBot:
    def __init__(self, system="", model="gpt-4o", provider="openai"):
        self.system = system
        self.model = model
        self.provider = provider
        self.messages = []
        if system:
            self.messages.append({"role": "system", "content": system})
        self.chat = None
        if provider == "gemini":
            self.chat = genai.GenerativeModel(model).start_chat(history=[])
            self.response = self.chat.send_message(system)
            

    def __call__(self, message):
        if self.provider == "openai":
            self.messages += [{"role": "user", "content": message}]
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                top_p=0.5,
                timeout=10
            )
            result = response.choices[0].message.content.replace("\n", "")
            self.messages += [{"role": "assistant", "content": result}]
            return result
        elif self.provider == "gemini":
            time.sleep(3)
            response = self.chat.send_message(message)
            return response.text.replace("\n", "")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def invalidQuestion(self, promptC):
        prompt = "Oracle said: This question is invalid as it violates what questions can be asked " + promptC
        return self.__call__(prompt)

    def top5(self):
        prompt = "Based on the previous questions what 5 physical objects are most likely? Why do you think these objects are most likely? Return a list of the objects, after the list put a tab then an explanation on why you think they are most likely."
        return self.__call__(prompt)