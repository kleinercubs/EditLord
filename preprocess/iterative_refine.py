import json
from tqdm import tqdm
from utils import parse_args
from models import GeminiPro, GPT
import pandas as pd

args = parse_args()
if args.model == 'gemini':
    model = GeminiPro(args)
elif args.model == 'gpt':
    model = GPT(args)

generic_rule_examples = json.load(open("data/{args.task}/generic_rule_examples.json"))
specific_rule_examples = json.load(open("data/{args.task}/specific_rule_examples.json"))
applied_strategies = json.load(open("data/{args.task}/editing_rules.json"))
if args.task == 'performance':
    old_property = 'slow code'
    new_property = 'fast code'
elif args.task == 'security':
    old_property = 'vulnerable code'
    new_property = 'secure code'
elif args.task == 'readability':
    old_property = 'unreadable code'
    new_property = 'readable code'
print(len(applied_strategies))

def retrieve_actions(resp, delimiter=', '):
    actions = []
    for action_type in ['UPDATE', 'MERGE', 'DELETE', 'REFORMAT']:
        if f'[{action_type}]' in resp:
            action = resp.split(f'[{action_type}]')[1].split(f'[/{action_type}]')[0]
            action = action.split(delimiter)
            action = [_.strip() for _ in action]
            actions.append((action_type, action))
    return actions

def perform_action(global_strategy, actions):
    for action_type, action in actions:
        action = set(action)
        if len(action) == 0:
            continue
        if action_type == 'UPDATE':
            for _ in action:
                # print(f'add {_}')
                if (len(_) == 0):
                    continue
                global_strategy.add(_)
        elif action_type == 'MERGE':
            for _ in action:
                # print(f'merge {_}')
                if (len(_) == 0):
                    continue
                global_strategy.add(_)
        elif action_type == 'DELETE':
            for _ in action:
                if _ in global_strategy:
                    # print(f'remove {_}')
                    global_strategy.remove(_)
        elif action_type == 'REFORMAT':
            new_global_strategy = set()
            for _ in action:
                # print(f'add {_}')
                if (len(_) == 0):
                    continue
                new_global_strategy.add(_)
            global_strategy = new_global_strategy
    return global_strategy
def grow_mechainism(model, current_global_strategy, strategy):
    model.start_chat()
    global_strategy = '\n'.join(current_global_strategy)
    resp = model.send_message(None, growing_prompt.format(task=args.task, old_property=old_property, new_property=new_property, meta_rule_set=global_strategy, editing_rule=strategy))[0]
    actions = retrieve_actions(resp)
    current_global_strategy = perform_action(current_global_strategy, actions)
    return current_global_strategy, resp

def reformat_mechanism(model, current_global_strategy):
    model.start_chat()
    global_strategy = '\n'.join(current_global_strategy)
    resp = model.send_message(None, reformat_prompt.format(meta_rule_set=global_strategy))[0]
    actions = retrieve_actions(resp, delimiter='\n')
    current_global_strategy = perform_action(current_global_strategy, actions)
    return current_global_strategy, resp

generic_or_specific_prompt = """Please analyze the provided editing rule (in order to improve {task}) and determine whether it is broadly applicable across different code snippets (generic) or tailored to a specific code snippet (specific). An editing rule like ”{generic_rule_examples}” should be considered as a generic rule. While a rule like ”{specific_rule_examples}” should be considered as a specific rule.
Provide your response in the following format:
The rule is [generic/specific] because ...
So, what do you think about the rule ”{editing_rule}”? Is it generic or specific?
"""

growing_prompt = """Please analyze the provided editing rule (in order to improve {task}) and compare it with the existing editing rules in the meta-rule set. If it’s similar to any existing editing rule, please suggest how it should be integrated into the existing meta-rule set. Specify the one and only one appropriate action from the options below:
[ADD]: If none of the existing editing rules in the meta-rule set is similar to the current one, provide the refined and updated editing rule to be added to the set.
[MERGE]: If the current editing rule is similar to an existing editing rule, indicate which existing meta-rule is similar to the current editing rule so that they can be merged and how they should be merged.
If [ADD] is selected, please provide the refined and updated editing rule to be added to the set directly without any other information. If [MERGE] is selected, please provide exactly the existing meta-rule that is similar to the current editing rule with an updated editing rule.
Please notice that whether you select [ADD] or [MERGE], the editing rule you add or merge into the meta-rule set must adhere to the format “switch from ... to ...”. Ensure that you only provide editing rules that transition from a {old_property} to {new_property}.
Here are several examples of the output:
[Example Output 1]
[ADD] only the editing rule to be added here [/ADD]
[/Example Output 1]
[Example Output 2]
[MERGE] only the editing rule to be merged and the updated rule, split by semicolon [/MERGE]
[/Example Output 2]
Meta-Rule Set:
{meta_rule_set}
Editing Rule Requested for Analysis:
{editing_rule}
"""

reformat_prompt = """Please read and analyze the global stategy set (in order to improve security), and restructure it to be more concise and easier to understand. The restructured set should include only unique strategies, presented clearly and concisely. Output each strategy in the format "switch from ... to ..." or "use ..." on separate lines, ensuring all original strategies are covered. Do not include any further explanations. Ensure the strategies remain are generally applicable and understandable to developers. Ensure each strategies are atomic and independent of each other. Ensure that you only provide strategies that transition from a vulnerable code to secure code.

Here is an example of the output:
[DELETE]
switch from ... to ...
switch from ... to ...
...
use ...
...
[/DELETE]
Here is the global meta-rule set:
{meta_rule_set}
"""


generic_num, specific_num = 0, 0
global_strategy = set()
current_global_strategy = set(global_strategy)
iterated_strategies = pd.DataFrame(columns=['current_strategy', 'action', 'global_strategy', 'raw_response'])

remaining_global_strategy = applied_strategies
steps_log = pd.DataFrame(columns=['step', 'strategy', 'shrinked_resp', 'grow_resp', 'current_global_strategy'])
bar = tqdm(enumerate(remaining_global_strategy))
for step, strategy in bar:
    log_info = {
        'step': step,
        'strategy': strategy,
    }
    try:
        model.start_chat()
        resp = model.send_message(None, generic_or_specific_prompt.format(task=args.task, generic_rule_examples=generic_rule_examples, specific_rule_examples=specific_rule_examples, editing_rule=strategy))[0]
        generic_num += 1 if 'The strategy is generic' in resp else 0
        specific_num += 1 if 'The strategy is specific' in resp else 0
        log_info.update({
            'generic_or_specific': 'generic' if 'The strategy is generic' in resp else 'specific',
            'generic_or_specific_reason': resp
        })

        if 'The strategy is generic' not in resp:
            log_info.update({'global_strategy': global_strategy})
            iterated_strategies = pd.concat([iterated_strategies, pd.DataFrame([log_info])], ignore_index=True)
            continue
        current_global_strategy, grow_resp = grow_mechainism(model, current_global_strategy, strategy)

        if step % 100 == 0:
            current_global_strategy, reformat_resp = reformat_mechanism(model, current_global_strategy)
            log_info.update({'reformat_resp': reformat_resp})
        log_info.update({'grow_resp': grow_resp})
        log_info.update({'current_global_strategy': current_global_strategy})
    except Exception as e:
        print('step: ', step)
        print('strategy: ', strategy)
        print('current_global_strategy: ', current_global_strategy)
        print('e: ', e)
        pass
    steps_log = pd.concat([steps_log, pd.DataFrame([log_info])], ignore_index=True)
    steps_log.to_csv(f'data/{args.task}/meta-rule-log.csv', index=False, header=True)
    json.dump(list(current_global_strategy), open(f'data/{args.task}/meta-rule.jsonl', 'w'), indent=2)
    bar.set_description(f"Global Strategy: {len(current_global_strategy)}")
    
current_global_strategy, reformat_resp = reformat_mechanism(model, current_global_strategy)
json.dump(list(current_global_strategy), open(f'data/{args.task}/meta-rule.jsonl', 'w'), indent=2)