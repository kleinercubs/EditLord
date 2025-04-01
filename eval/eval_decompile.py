"""
Code used for sampling programs based on the text-generation-inference API at https://github.com/huggingface/text-generation-inference

"""

from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import fire
import os
import json
import concurrent.futures
from prompts import promptManager
from utils import decompile_pass_rate
import re

def extract_code_from_response(response):
    edited_code = None
    edited_code = re.search(r'```c\n(.*)```', response.replace('```c++', '```cpp').replace('```cpp', '```c'), re.DOTALL)
    if edited_code is None:
        edited_code = re.search(r'```\n(.*)```', response, re.DOTALL)
        if edited_code is None:
            edited_code = response
        else:
            edited_code = edited_code.group(1)
    else:
        edited_code = edited_code.group(1)
    if '[Equivalent]' in edited_code:
        edited_code = edited_code.split('[Equivalent]')[1]
        if '[/Equivalent]' in edited_code:
            edited_code = edited_code.split('[/Equivalent]')[0]
    return edited_code

def parse_code(code):
    try:
        parsed_code = code.split('[/ORIGINAL SOURCE CODE]')[0].strip()
    except:
        parsed_code = code
    code = parsed_code
    try:
        parsed_code = code.split('[ORIGINAL SOURCE CODE]')[1].strip()
    except:
        parsed_code = code
    code = parsed_code
    try:
        parsed_code = extract_code_from_response(code)
    except:
        parsed_code = code
    return parsed_code

def extract_first_program(query, text):
    try:
        code = text.split('[ORIGINAL SOURCE CODE]')[1].split('[/ORIGINAL SOURCE CODE]')[0].strip()
    except:
        # print(text)
        code = text

    return {
        'query': query,
        'code': code,
        'raw_data': text
    }


def main(
    test_file=None,
    output_file=None,
    num_samples=8,
    temperature=0.7,
    base_url=None,
    api_key='token-abc123',
    num_threads=20,
    fine_tuned_model: str = "",
    method: str = "",
    task: str = "decompile",
):
    client = OpenAI(api_key=api_key, base_url=base_url)

    print(test_file)
    test_df = json.load(open(test_file))
    test_df = pd.DataFrame(test_df)
    # rename input_asm_prompt to src_code
    test_df.rename(columns={"input_asm_prompt": "src_code"}, inplace=True)

    # create results dataframe with src_code column
    results_df = pd.DataFrame(columns=["src_code"])
    results_df["src_code"] = test_df["src_code"]
    # create empty column for completions
    results_df["generated_answers"] = results_df.apply(lambda x: [], axis=1)
    results_df["prompt"] = results_df.apply(lambda x: [], axis=1)

    def process_request(index, row):
        data_point = {
            'src_code': row['src_code'],
        }
        prompt = promptManager.get_user_prompt(data_point, task=task, method=method)
        messages = [
            {"role": "user", "content": prompt},
        ]
        completion = client.chat.completions.create(
            model=fine_tuned_model,
            messages=messages,
            n=num_samples,
            temperature=temperature,
        )
        best_of_sequences = [choice.message.content for choice in completion.choices]
        
        all_programs = [
            extract_first_program(prompt, best_of_sequences[i])
            for i in range(len(best_of_sequences))
        ]
            
        return index, all_programs, messages
    
    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Use ThreadPoolExecutor to process in parallel
        with tqdm(total=len(test_df)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_index = {executor.submit(process_request, i, row): i for i, row in test_df.iterrows()}
                for future in concurrent.futures.as_completed(future_to_index):
                    index, all_programs, prompt = future.result()
                    results_df.at[index, "prompt"] = prompt
                    results_df.at[index, "generated_answers"] = all_programs
                    pbar.update(1)
                    pbar.set_description(f"Processing {index}")
                    
        test_df["generated_answers"] = results_df["generated_answers"]
        test_df["messages"] = results_df["prompt"]

        # save test_df to output_file
        test_df.to_json(output_file, orient="records", lines=True)
    test_df = pd.read_json(output_file, orient="records", lines=True).to_dict(orient="records")

    testsets = []
    gen_results_repeat = []
    for d in test_df:
        raw_output = [generated_answer['raw_data'] for generated_answer in d['generated_answers']]
        output = [parse_code(generated_answer['code']) for generated_answer in d['generated_answers']]
        gen_results_repeat.append(output)
        d.update({
            'input_asm_prompt': d['src_code'],
            'output': raw_output,
            'raw_output': raw_output
        })
        testsets.append(d)
        print(d)
    gen_results_repeat = [gen_results_repeat]
    avg_stats, output_info = decompile_pass_rate(testsets, gen_results_repeat)

    output_info.to_csv(output_file.replace('.json', '-output.csv'), index=False)
    json.dump(avg_stats, open(output_file.replace('.json', '-stat.json'), "w"), indent=4)
    
if __name__ == "__main__":
    fire.Fire(main)