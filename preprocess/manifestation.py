import json
from tqdm import tqdm
import openai
import os
from utils import parse_args
from models import GeminiPro, GPT
import pandas as pd

args = parse_args()
if args.model == 'gemini':
    model = GeminiPro(args)
elif args.model == 'gpt':
    model = GPT(args)

query_prompt = """You will be provided with a code along with a list of security-degrading symptons. Can you iterate through this list of symptons one by one and determine whether that symton is presented in the provided code? Ensure that each sympton is indeed found in the code. Output each sympton on separate lines. Do not include any further explanation.

Here is an example of the output: 
[ROOT CAUSES] 
... 
[/ROOT CAUSES]

Here is the list of security-degrading rules: {performance_list}

Here is the code you need to analyze: {code}
"""

global_strategy = json.load(open('data/vulnerability/meta-rule.jsonl'))
slow_rules = set()
fast_rules = set()
global_rules = set()
print(len(global_strategy))
for strategy in global_strategy:
    # extract rules from strategy 'switch from ... to ...'
    try:
        structured_strategy = strategy.split('switch from')[1].split('to')
        if len(structured_strategy) != 2:
            continue
    except:
        continue
    slow_rule = structured_strategy[0].strip()
    fast_rule = structured_strategy[1].strip()
    global_rules.add((slow_rule, fast_rule))
    slow_rules.add(slow_rule)
    fast_rules.add(fast_rule)
rules = {
    'src': list(slow_rules),
    'tgt': list(fast_rules)
}
print('slow_rules: ', len(slow_rules))
print('fast_rules: ', len(fast_rules))
data = pd.read_json(f'data/vulnerability/{args.split}.jsonl', lines=True, orient='records')
print(len(data))

sample_log = pd.DataFrame(columns=['sample_id', 'src_code', 'src_rules', 'src_raw_response', 'tgt_code', 'tgt_rules', 'tgt_raw_response', 'error_message', 'applied_rules'])
no_rules_num = 0
bar = tqdm(data.iterrows(), total=len(data))
for step, d in bar:
    log_info = {
        'step': step,
        'strategy': strategy,
    }
    for code_type in ['src', 'tgt']:
        try:
            model.start_chat()
            resp = model.send_message(None, query_prompt.format(performance_list = '\n'.join(rules[code_type]), code=d[f'{code_type}_code']))[0]
            selected_rules = resp.split('[ROOT CAUSES]')[1].split('[/ROOT CAUSES]')[0]
            selected_rules = selected_rules.split('\n')
            selected_rules = [_.strip() for _ in selected_rules]
            if '' in selected_rules:
                selected_rules.remove('')
            log_info.update({
                f'{code_type}_code': d[f'{code_type}_code'],
                f'{code_type}_raw_response': resp,
                f'{code_type}_rules': selected_rules
            })
        except Exception as e:
            log_info.update({'error': e})
            print(e)
    slow_rules = log_info['src_rules'] if 'src_rules' in log_info else []
    fast_rules = log_info['tgt_rules'] if 'tgt_rules' in log_info else []
    applied_rules = set()
    for slow_rule in slow_rules:
        for fast_rule in fast_rules:
            slow_rule = slow_rule.strip()
            fast_rule = fast_rule.strip()
            if (slow_rule, fast_rule) in global_rules:
                applied_rules.add((slow_rule, fast_rule))
    no_rules_num += len(applied_rules) == 0
    bar.set_description(f"No Rules: {no_rules_num}")
    applied_rules = [f'switch from {slow_rule} to {fast_rule}' for slow_rule, fast_rule in applied_rules]
    applied_rules = '\n'.join(applied_rules)
    log_info.update({'applied_rules': applied_rules})
    sample_log = pd.concat([sample_log, pd.DataFrame([log_info])], ignore_index=True)
    sample_log.to_csv(f'data/vulnerability/{args.split}_manifested.csv', index=False, header=True)
    
