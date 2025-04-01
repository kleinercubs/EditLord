from .base import BaseLLM

import vertexai
from vertexai.generative_models import GenerativeModel, Content, Part, GenerationConfig

class GeminiPro(BaseLLM):
    def __init__(self, args):
        super(GeminiPro, self).__init__()
        self.args = args
        self.model = self.build_model(args)
        self.llm_round_per_sample = 0
        self.history = []

    def build_model(self, args):
        vertexai.init(project=args.project_id, location="us-central1")
        model = GenerativeModel(
            model_name="gemini-1.0-pro-002",
            generation_config=GenerationConfig(
                temperature = 0.0
            )
        )
        return model

    def start_chat(self):
        self.history = []
        self.llm_round_per_sample = 0
        self.chat = self.model.start_chat()

    def send_message(self, env, message: str) -> str:
        """
        Send a message to the model and return the response
        """
        self.llm_round_per_sample += 1
        try:
            response = self.chat.send_message(message)
            self.history.append([
                {
                    'role': 'user',
                    'message': message
                },
                {
                    'role': 'assistant',
                    'message': response.text
                }
            ])
            
            if env is not None:
                env.update_conversations({
                    'Question': message,
                    'Answer': response.text
                })
            return [response.text]
        except Exception as e:
            error = str(e)
            return [error + "|||||||||||||'code': 'context_length_exceeded'" ]