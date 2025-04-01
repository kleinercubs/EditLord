import json
import ast
import pandas as pd
import random
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='test')
args = parser.parse_args()

data_spec = f"data/vulnerability/{args.split}_correctness_specs.jsonl"

data_strategy = f"data/vulnerability/{args.split}_manifested.csv"
data_strategy = pd.read_csv(data_strategy)

data_spec = pd.read_json(data_spec, orient='records', lines=True).to_dict('records')

src_to_spec = {(_['src_code'], _['tgt_code']): _ for _ in data_spec}

data = []
rule_num = []
for index, row in data_strategy.iterrows():
    if (row['src_code'], row['tgt_code']) not in src_to_spec:
        continue
    spec = src_to_spec[(row['src_code'], row['tgt_code'])]
    raw_spec = json.loads(spec['raw_response'])
    if row['applied_rules'] is np.nan:
        continue
    if row['applied_rules'].startswith('{('):
        rule_pairs_set = ast.literal_eval(row['applied_rules'])
        rules = [f'switch from {src} to {tgt}' for (src, tgt) in rule_pairs_set]
        row['applied_rules'] = '\n'.join(rules)
    if len(row['applied_rules'].split('\n')) > 9:
        rules = row['applied_rules'].split('\n')
        rules = random.sample(rules, 9)
        row['applied_rules'] = '\n'.join(rules)
    rule_num.append(len(row['applied_rules'].split('\n')))
    assert type(row['applied_rules']) == str
    assert '{(' not in row['applied_rules']

    d = spec
    d.update({
        'spec_raw': raw_spec,
        'input': raw_spec['input'],
        'output': raw_spec['output'],
        'problem_description': raw_spec['problem_description'],
        'applied_strategies': row['applied_rules']
    })
    data.append(d)

print(len(data))
data = pd.DataFrame(data)
import numpy as np
print(np.mean(rule_num), np.median(rule_num))
data.to_json(f"data/vulnerability/{args.split}.jsonl", orient='records', lines=True)
