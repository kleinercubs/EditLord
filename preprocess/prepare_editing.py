import json
import os
from utils import parse_args
from tqdm import tqdm
from models import GeminiPro, GPT
from utils import parse_args
from collections import defaultdict
import prompts.editingRulePrompt as editingRulePrompt

args = parse_args()
if args.model == 'gemini':
    model = GeminiPro(args)
elif args.model == 'gpt':
    model = GPT(args)

env = None

response_list = []
upd_data = []

analytics = defaultdict(int)

finetune_dataset_path = f'{args.output_file}_finetuning_data.json'
global_hierarchy_path = f'{args.output_file}_global_editing_hierarchy.json'
if os.path.exists(finetune_dataset_path):
    finetuning_data = json.load(open(finetune_dataset_path))
    global_hierarchy = json.load(open(global_hierarchy_path))
else:
    finetuning_data = []
    global_hierarchy = []
processed_length = len(finetuning_data)

with open(args.data) as f:
    if args.model.startswith('gpt'):
        unprocessed_data = f.readlines()[processed_length:]
        for line_idx, line in tqdm(enumerate(unprocessed_data), total=len(unprocessed_data)):
            d = json.loads(line)
            try:
                src_code = d['src_code']
                tgt_code = d['tgt_code']
                summarization = editingRulePrompt.editing_rule_through_diff[args.task].format(src_code=src_code, tgt_code=tgt_code, global_hierarchy=global_hierarchy, in_context_examples=editingRulePrompt.in_context_examples[args.task])
                model.start_chat()
                resp = model.send_message(env, summarization)
                d.update({'raw_response': resp[0]})
                try:
                    reasoning = resp[0].split('[REASONING]')[1].split('[/REASONING]')[0]
                except:
                    reasoning = ""
                try:
                    applied_strategies = resp[0].split('[APPLIED_STRATEGIES]')[1].split('[/APPLIED_STRATEGIES]')[0]
                except:
                    applied_strategies = ""
                d.update({'reasoning': reasoning})
                d.update({'applied_strategies': applied_strategies})
                d.update({'prompt': summarization})
                finetuning_data.append(d)
                json.dump(finetuning_data, open(finetune_dataset_path, "w"), indent=2)
            except Exception as e:
                print('# line no: ', line_idx, e)
                d.update({'error': str(e)})
                finetuning_data.append(d)
                pass
