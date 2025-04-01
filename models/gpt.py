from .base import BaseLLM

import os
import openai
import tiktoken

class GPT(BaseLLM):
    def __init__(self, args):
        super(GPT, self).__init__()
        self.args = args
        self.model = self.build_model(args)
        self.llm_round_per_sample = 0
        self.history = []
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self.max_token_length = 16385

    def build_model(self, args):
        from openai import OpenAI
        model = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )
        return model

    def start_chat(self):
        self.history = []
        self.llm_round_per_sample = 0
        self.max_token_length = 16385

    def send_message(self, env, message: str, num_of_samples: int=1, temperature: float=0.0, top_p: float=0.7) -> str:
        """
        Send a message to the model and return the response
        """
        self.llm_round_per_sample += 1
        self.max_token_length -= len(self.tokenizer.encode(message)) + 10
        response = self.model.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.history + [{'role': 'user', 'content': message}],
            temperature=temperature,
            top_p=top_p,
            n=num_of_samples,
            max_tokens=4096,
        )
        # print(response)
        response = [choice.message.content for choice in response.choices]
        # response = [message]
        # print(response)
        # print('==============================')
        self.history.extend([
            {
                'role': 'user',
                'content': message
            },
            {
                'role': 'assistant',
                'content': response[0]
            }
        ])
        # print(response)
        # import json
        # print(json.dumps(self.history, indent=2))
        if env is not None:
            env.update_conversations({
                'Question': message,
                'Answer': response
            })
        return response
        # except Exception as e:
        #     error = str(e)
        #     return [error + "|||||||||||||'code': 'context_length_exceeded'" ]
