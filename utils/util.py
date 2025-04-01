from pathlib import Path
import re
import glob


def increment_path(path, exist_ok=False, sep='', mkdir=True):
    """
    Copied from yolov5.
    Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    :param path: file or directory path to increment
    :param exist_ok: existing project/name ok, do not increment
    :param sep: separator for directory name
    :param mkdir: create directory
    :return: incremented path
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir_ = path if path.suffix == '' else path.parent  # directory
    if not dir_.exists() and mkdir:
        dir_.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def build_model(args):
    if args.model == "gpt3" or args.model == "gpt4":
        from openai import OpenAI
        # openai.api_key = os.environ["OPENAI_API_KEY"]
        # clientCall = openai.ChatCompletion.create
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ["OPENAI_API_KEY"]
        )
        clientCall = client.chat.completions.create
    else:
        if args.online:
            from openai import OpenAI
            # TOGETHER_API_KEY="3e62864884b295dd308a6c5d02c2e0564dca526527e9f15228be68687bbe6c65"
            # client = OpenAI(api_key=TOGETHER_API_KEY,
            #     base_url='https://api.together.xyz',
            # )
            openai_api_key = "EMPTY"
            openai_api_base = "http://0.0.0.0:8000/v1"
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            clientCall = client.chat.completions.create
            # from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # quantization_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.float16
            # )

            # model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, cache_dir='.cache')
            # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='.cache')

            # clientCall = pipeline("text-generation", model=model, tokenizer=tokenizer, cache_dir='.cache')

        else:
            from llama import Llama
            clientCall = Llama.build(
                ckpt_dir=f'/proj/rcs-ssd/lwc/{model_name}',
                tokenizer_path=f'/proj/rcs-ssd/lwc/{model_name}/tokenizer.model',
                max_seq_len=10000,
                max_batch_size=1,
            )

def llm(input, llm_round_per_sample, env, top_k=5):
    llm_round_per_sample += 1
    return_error = 'empty'
    rounds = 1 if args.online else 1
    for _ in range(rounds):
        try:
            if model_name.startswith('codellama'):# == 'codellama/CodeLlama-7b-Instruct-hf':
                if args.online:
                    response = clientCall(
                        messages=input,
                        model=model_name,
                        max_tokens=10000,
                        temperature=0.0,
                        top_p=0.95
                    )
                    edited_text = [choice.message.content for choice in response.choices]
                    # response = clientCall(input, max_new_tokens=10000)
                    # print('[response]::::', response)
                    # edited_text = [response[0]['generated_text'][-1]['content']]
                    # tokenized_chat = tokenizer.apply_chat_template(input, tokenize=True, add_generation_prompt=True, return_tensors="pt")
                    # edited_text = model.generate(tokenized_chat, max_new_tokens=10000) 
                else:
                    response = clientCall.chat_completion(
                        [input],
                        temperature=0.0,
                        top_p=0.95
                    )
                    edited_text = [choice['generation']['content'] for choice in response]
            else: # gpt3/gpt4
                response = clientCall(
                    model=model_name,
                    messages=input,
                    temperature=0.0,
                    n=top_k,
                )
                edited_text = [choice.message.content for choice in response.choices]
            env.update_conversations({
                'Question': input,
                'Answer': edited_text[0]
            })
            return edited_text, True, llm_round_per_sample
        except Exception as e:
            e = str(e)
            if "'code': 'context_length_exceeded'" in e:
                return ['error - context_length_exceeded'], False, llm_round_per_sample
            print(e)
            time.sleep(65)
            return_error = e
            pass
        
        # except openai.error.InvalidRequestError:
        #     openai_invalid_request_error_count += 1
        #     return [], False
        # except openai.error.RateLimitError:
        #     time.sleep(65)
        #     openai_api_connection_error_count += 1
        #     pass
        # except:
        #     openai_unexpected_error_count += 1
        #     print("Unexpected error:", sys.exc_info()[0])
        #     return [], False
    return [return_error], False, llm_round_per_sample