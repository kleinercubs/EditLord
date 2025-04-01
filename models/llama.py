from .base import BaseLLM

import os
import openai
class Llama(BaseLLM):
    def __init__(self, args):
        super(Llama, self).__init__()
        self.args = args
        self.model = self.build_model(args)
        self.llm_round_per_sample = 0
        self.history = ""

    def build_model(self, args):
        model = openai.OpenAI(
            api_key='token-abc123',
            base_url="http://0.0.0.0:1234/v1",
        )
        return model

    def start_chat(self):
        self.history = ""
        self.llm_round_per_sample = 0

    def send_message(self, env, message: str, num_samples: int=1, temperature: float=0.7, max_tokens:int =10000)-> str:
        """
        Send a message to the model and return the response
        """
        self.llm_round_per_sample += 1
        self.history += message
        try:
            # print(self.history + [{'role': 'user', 'content': message}])
            response = self.model.completions.create(
                model="ds-1b",
                prompt=self.history,
                max_tokens=max_tokens,
                temperature=temperature,
                n=num_samples
            )
            # print(response)
            response = [choice.text for choice in response.choices]
            # print(response)
            # print('==============================')
            self.history += '\n' + response[0] + '\n'
            
            # import json
            # print(json.dumps(self.history, indent=2))
            if env is not None:
                env.update_conversations({
                    'Question': message,
                    'Answer': response[0]
                })
            return response
        except Exception as e:
            error = str(e)
            return [error + "|||||||||||||'code': 'context_length_exceeded'" ]