import os
import json
from loguru import logger
import traceback
from argparse import ArgumentParser
import sys
from tqdm import tqdm
import concurrent.futures
from collections import OrderedDict
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import os
import json
import concurrent.futures
from prompts import promptManager
from sven.utils import set_seed
from sven.constant import CWES_DICT

logger.add(sys.stdout, colorize=False, format="{time} {level} {message}")
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=8)
    parser.add_argument("--max_num_seqs", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.82)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max_total_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--base_url", type=str)
    parser.add_argument('--num_gen', type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='CWEval/benchmark')
    parser.add_argument('--output_dir', type=str, default='output/sec_eval')
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--remove_comment', action='store_true')
    parser.add_argument('--eval_type', type=str, choices=['trained', 'trained_subset', 'prompts', 'gen_1', 'gen_2', 'edit'], default='edit')
    parser.add_argument('--seed', type=int, default=1)

    return parser.parse_args()

import re

def remove_comments(c_code: str) -> str:
    # Step 1: Temporarily replace all URLs (or anything in quotes) with a placeholder
    # This preserves URLs and other quoted strings during comment removal
    c_code = re.sub(r'https?://[^\s"]+', '<URL>', c_code)  # Match and replace URLs
    c_code = re.sub(r'"[^"]*"', '<QUOTE>', c_code)  # Match and replace quoted strings

    # Step 2: Remove single-line comments (//)
    c_code = re.sub(r'//.*$', '', c_code, flags=re.MULTILINE)
    
    # Step 3: Remove multi-line comments (/* */)
    c_code = re.sub(r'/\*.*?\*/', '', c_code, flags=re.DOTALL)

    # Step 4: Restore URLs and quoted strings
    c_code = c_code.replace('<URL>', 'https://www.example.com')  # Placeholder restoration can be dynamic
    c_code = c_code.replace('<QUOTE>', '"Some quoted string"')  # Adjust quoted string restoration dynamically if needed

    return c_code

def prepare_input(args, controls, output_dir, data_dir, scenario):
    control = 'orig'
    s_out_dir = os.path.join(output_dir, scenario)
    os.makedirs(s_out_dir)
    s_in_dir = os.path.join(data_dir, scenario)

    input_src_list = []
    for file_name in os.listdir(s_in_dir):
        if 'unsafe' not in file_name:
            continue
        input_content = open(os.path.join(s_in_dir, file_name)).read()
        input_code_lines = input_content.splitlines()
        input_code = ''
        for line in input_code_lines:
            if line.strip().startswith('#include'):
                continue
            if line.strip() == '// BEGIN PROMPT':
                continue
            if line.strip() == '// BEGIN SOLUTION':
                continue
            if line.strip() == '// BEGIN ENTRYPOINT':
                break
            if line.strip() == '// BEGIN ENTRYPONT':
                break
            if '// BEGIN ENTRY' in line:
                break
            input_code += line + '\n'
        if args.remove_comment:
            input_code = remove_comments(input_code)
        set_seed(args)
        data_point = {'src_code': input_code}
        input_src = promptManager.get_user_prompt(
            data_point = data_point,
            method=args.method,
            task="vulnerability"
        )
        input_src_list.append({
            'output_dir': output_dir,
            'data_dir': data_dir,
            'scenario': scenario,
            'control': control,
            'input_src': input_src,
            'file_name': file_name,
            'lang': file_name.split('.')[-1]
        })
    return input_src_list



def run_eval_pipeline(args: ArgumentParser) -> int:

    try:
        controls = ['orig']
        input_list = []
        benchmark_info = {
            'core/c', 
            'core/cpp', 
            'lang/c', 
        }
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)#os.environ["OPENAI_API_KEY"])
        print(client.models.list())
        for benchmark in benchmark_info:
            input_list += prepare_input(args, controls, args.output_dir, args.data_dir, benchmark)
        input_list = input_list
        def process_request(index, row):
            prompt = row['input_src']
            messages = [
                {"role": "user", "content": prompt},
            ]
            messages = [
                {"role": "user", "content": prompt},
            ]
            completion = client.chat.completions.create(
                model=args.model_path,
                messages=messages,
                n=args.num_gen,
                temperature=args.temperature,
            )
            best_of_sequences = [choice.message.content for choice in completion.choices]
            all_programs = best_of_sequences
                
            return index, all_programs, messages
        
        result_df = []
        
        if not os.path.exists(args.output_file):
            # Use ThreadPoolExecutor to process in parallel
            with tqdm(total=len(input_list)) as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                    future_to_index = {executor.submit(process_request, i, row): i for i, row in enumerate(input_list)}
                    for future in concurrent.futures.as_completed(future_to_index):
                        index, all_programs, prompt = future.result()
                        result_df.append({
                            'prompt': prompt,
                            'generated_answers': all_programs,
                            'src_code': input_list[index]
                        })
                        pbar.update(1)
                        
            result_df = pd.DataFrame(result_df)

            # save test_df to output_file
            result_df.to_json(args.output_file, orient="records", lines=True)
        result_df = pd.read_json(args.output_file, orient="records", lines=True).to_dict(orient="records")

        result_df = {d['src_code']['file_name']: d for d in result_df}
        gen_results = [result_df[input_dict['file_name']]['generated_answers'] for input_dict in input_list] #[gen['generated_answers'] for gen in result_df]

        result_dict = OrderedDict()
        for gen_result, input_dict in zip(gen_results, input_list):
            # go through the #num_gen of generated results
            output_srcs, dup_srcs, non_parsed_srcs = [], [], []
            for i, output in enumerate(gen_result):
                raw_output_path = os.path.join(input_dict['output_dir'], input_dict['scenario'], f'{input_dict["file_name"]}_{i}.txt')
                output_dir = os.path.join(input_dict['output_dir'], input_dict['scenario'], 'orig_output')
                os.makedirs(output_dir, exist_ok=True)
                non_parsed_dir = os.path.join(input_dict['output_dir'], input_dict['scenario'], 'orig_non_parsed')
                os.makedirs(non_parsed_dir, exist_ok=True)
                dup_dir = os.path.join(input_dict['output_dir'], input_dict['scenario'], 'orig_dup')
                os.makedirs(dup_dir, exist_ok=True)
                output_path = os.path.join(input_dict['output_dir'], input_dict['scenario'], 'orig_output', f'{input_dict["file_name"]}_{i}.{input_dict["lang"]}')
                non_parsed_path = os.path.join(input_dict['output_dir'], input_dict['scenario'], 'orig_non_parsed', f'{input_dict["file_name"]}_{i}.{input_dict["lang"]}')
                dup_path = os.path.join(input_dict['output_dir'], input_dict['scenario'], 'orig_dup', f'{input_dict["file_name"]}_{i}.{input_dict["lang"]}')
                with open(raw_output_path, 'w', encoding='utf-8') as f:
                    if output is None:
                        output = 'None'
                    f.write(output)
                if ('[/SECURE CODE]' not in output):
                    non_parsed_srcs.append(output)
                    with open(non_parsed_path, 'w', encoding='utf-8') as f:
                        f.write(output)
                    code = output
                else:
                    code = output
                    if '[SECURE CODE]' in output:
                        code = output.split('[SECURE CODE]')[1]
                    code = code.split('[/SECURE CODE]')[0]
                    if code in output_srcs:
                        dup_srcs.append(code)
                        with open(dup_path, 'w', encoding='utf-8') as f:
                            f.write(code)
                    else:
                        output_srcs.append(code)
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(code)

                gen_file_path = os.path.join(
                    args.output_dir,
                    'cweval',
                    f'generated_{i}',
                    input_dict['scenario'],
                    input_dict['file_name'].replace('_unsafe', '_raw'),
                )
                os.makedirs(os.path.dirname(gen_file_path), exist_ok=True)
                with open(gen_file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                    
                result_dict[raw_output_path] = {
                    'output': output,
                    'input': input_dict['input_src']
                }
        
        with open(os.path.join(args.output_dir, 'result.jsonl'), 'w', encoding='utf-8') as f:
            for k, v in result_dict.items():
                f.write(json.dumps({'path': k, 'code': v['output'], 'prompt': v['input']})+'\n')
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        return -1
    


def main():
    args = parse_args()    
    args.output_dir = os.path.join(args.output_dir, args.output_name, args.eval_type)
    ret = run_eval_pipeline(args)
    sys.exit(ret)


if __name__ == "__main__":
    main()
