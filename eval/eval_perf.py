from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import fire
import os
import concurrent.futures
from prompts import promptManager

def extract_first_program(query, text):
    try:
        code = text.split('[FAST CODE]')[1].split('[/FAST CODE]')[0].strip()
    except:
        code = text

    return {
        'query': query,
        'code': code,
        'raw_data': text
    }

def main(
    test_file=None,
    output_file=None,
    do_sample=None,
    num_samples=8,
    temperature=0.7,
    api_key='token-abc123',
    num_threads=20,
    fine_tuned_model: str = "",
    method: str = "",
    task: str = "",
):
    client = OpenAI(api_key=api_key)
    test_df = pd.read_json(test_file, lines=True, orient="records")

    # create results dataframe with src_code column
    results_df = pd.DataFrame(columns=["src_code"])
    results_df["src_code"] = test_df["src_code"]
    # create empty column for completions
    results_df["generated_answers"] = results_df.apply(lambda x: [], axis=1)

    def process_request(index, row):

        # prompt = src_code
        data_point = {
            'src_code': row['src_code'],
            'problem_description': row['problem_description'] if 'problem_description' in row else '',
            'input': row['input_spec'] if 'input_spec' in row else '',
            'output': row['output_spec'] if 'output_spec' in row else '',
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
    
    # Use ThreadPoolExecutor to process in parallel
    with tqdm(total=len(test_df)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_index = {executor.submit(process_request, i, row): i for i, row in test_df.iterrows()}
            for future in concurrent.futures.as_completed(future_to_index):
                index, all_programs, prompt = future.result()
                results_df.at[index, "prompt"] = prompt
                results_df.at[index, "generated_answers"] = all_programs
                pbar.update(1)
                
    test_df["generated_answers"] = results_df["generated_answers"]
    test_df["messages"] = results_df["prompt"]

    # save test_df to output_file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    test_df.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    fire.Fire(main)