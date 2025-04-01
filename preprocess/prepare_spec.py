import json
import os
import openai
from utils import increment_path, parse_args, build_model
from tqdm import tqdm
from copy import deepcopy
from models import GeminiPro, GPT
from utils import parse_args
from collections import defaultdict
import random
import prompt as PROMPTS
import pandas as pd

args = parse_args()
if args.model == 'gemini':
    model = GeminiPro(args)
elif args.model == 'gpt':
    model = GPT(args)

env = None

response_list = []
upd_data = []

hierarchy = {}

def build_hierarchy():
    return json.dumps(hierarchy)

def get_in_context_examples():
    in_context_examples = ''
    for example in PROMPTS.PROBLEM_SPEC_EXAMPLES:
        updated_and_applied = {
            'problem_description': example['problem_description'],
            'input': example['input'],
            'output': example['output']
        }
        in_context_examples += f"""[CODE]
{example['code']}
[/CODE]
[OUTPUT]
{updated_and_applied}
[/OUTPUT]
"""
    return in_context_examples

def extract_json(response):
    if ('[OUTPUT]' in response): 
        response = response.split("[OUTPUT]")
    if ('[/OUTPUT]' in response):
        response = response[1].split("[/OUTPUT]")[0]
    if ('```json' in response):
        response = response.split("```json")[1]
        if ('```' in response):
            response = response.split("```")[0]
    if ('```' in response):
        response = response.split("```")[1]
        if ('```' in response):
            response = response.split("```")[0]
    return response

def extract_given_key_from(response, key):
    response = extract_json(response)
    response = json.loads(response.strip())
    response = response[key]
    return response

analytics = defaultdict(int)
strategy_provided = {}
IN_CONTEXT_EXAMPLES = get_in_context_examples()
finetuning_data = []
all_data = pd.read_json('data/{args.task}/data.jsonl', lines=True, orient='records').to_dict(orient='records')

bar = tqdm(enumerate(all_data), total=len(all_data))
semantics_aligned = 0
no_undefined = 0
if args.model == 'gpt':
    processed_data = set()
    for line_idx, line in bar:
        d = json.loads(line)
        processed_data.add(json.dumps({'src_code': d['src_code'], 'tgt_code': d['tgt_code']}))
    data = []

    bar = tqdm(all_data, total=len(all_data))
    for d in bar:
        if json.dumps({'src_code': d['src_code'], 'tgt_code': d['tgt_code']}) in processed_data:
            continue
        data.append(json.dumps({'src_code': d['src_code'], 'tgt_code': d['tgt_code']}))
        
    data = list(set(data))
    print(len(data))
    bar = tqdm(enumerate(data), total=len(data))
    
    for line_idx, line in bar:
        try:
            d = json.loads(line)
            src_code = d['src_code']
            tgt_code = d['tgt_code']

            ENSURE_FORMAT = 'Ensure that the output is provided in the exact following format: {"problem_description": "Your problem description here.", "input": "Your input description here.", "output": "Your output description here."}'
            summarization = f"""Analyze the following program that attempt to solve a problem. Please provide a natural language description that briefly describes the problem description this provided code tries to solve. Clearly describe the inputs and outputs based on the code structure. For inputs, specify the parameters, their types, and their intended purpose in order. For the output, specify its type and what it represents in order.

{ENSURE_FORMAT}

{IN_CONTEXT_EXAMPLES}
[CODE]
{tgt_code}
[/CODE]
[OUTPUT]
"""
            model.start_chat()
            resp = model.send_message(env, summarization)
            old_hierarchy = deepcopy(strategy_provided)
            try:
                input = extract_given_key_from(resp[0], key='input')
            except:
                input = ''
            try:
                output = extract_given_key_from(resp[0], key='output')
            except:
                output = ''
            try:
                problem_description = extract_given_key_from(resp[0], key='problem_description')
            except:
                problem_description = ''
            d.update({'raw_response': resp[0]})
            d.update({'input_spec': input})
            d.update({'output_spec': output})
            d.update({'problem_description': problem_description})
            finetuning_data.append(d)
            semantics_aligned += 1
            bar.set_description(f"Semantics Aligned: {semantics_aligned} | No Undefined: {no_undefined} | Total: {line_idx+1}")
            json.dump(finetuning_data, open(args.output_file, "w"), indent=2)
        except Exception as e:
            print('# line no: ', line_idx)
            pass
finetuning_data = pd.DataFrame(finetuning_data)
finetuning_data.to_json(args.output_file.replace('.json', '.jsonl'), orient='records', lines=True)