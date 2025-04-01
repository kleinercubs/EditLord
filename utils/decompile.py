import subprocess
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm
import multiprocessing
import tempfile
import torch
from collections import defaultdict

def edit_distance(str1, str2):
    len1 = len(str1)
    len2 = len(str2)

    # Create a matrix to store the distances
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Initialize the first row and column
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Calculate the distances
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[len1][len2]


def compute_ES(target, prediction):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines_tmp = []
    for line in target_lines:
        if '#include' in line:
            continue
        target_lines_tmp.append(line)
    target_lines = target_lines_tmp
    target_str = '\n'.join(target_lines)
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_str = '\n'.join(prediction_lines)
    # tokenizer = AutoTokenizer.from_pretrained('/proj/rcs-hdd/aj3051/SecureAI4Code/saved_models/decompile/base-spec/ds-1b')
    tokenizer = AutoTokenizer.from_pretrained('saved_models/decompile/ds-1b-base-spec-short')
    target_str = tokenizer.decode(tokenizer.encode(target_str))
    prediction_str = tokenizer.decode(tokenizer.encode(prediction_str))
    ES_score=1 - (edit_distance(target_str, prediction_str) / max(len(target_str), len(prediction_str)))

    import editdistance
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines_tmp = []
    for line in target_lines:
        if '#include' in line:
            continue
        target_lines_tmp.append(line)
    target_lines = target_lines_tmp
    target_str = '\n'.join(target_lines)
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_str = '\n'.join(prediction_lines)
    native_es_score=1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))

    # model = AutoModel.from_pretrained("codesage/codesage-base", trust_remote_code=True, cache_dir = '/proj/rcs-hdd/aj3051/hf_transformers').cuda()
    # tokenizer = AutoTokenizer.from_pretrained("codesage/codesage-base", add_eos_token=True, trust_remote_code=True, cache_dir = '/proj/rcs-hdd/aj3051/hf_transformers')
    model = AutoModel.from_pretrained("codesage/codesage-base", trust_remote_code=True, cache_dir = '.cache').cuda()
    tokenizer = AutoTokenizer.from_pretrained("codesage/codesage-base", add_eos_token=True, trust_remote_code=True, cache_dir = '.cache')
    inputs1 = tokenizer.encode(" ".join(target_lines), return_tensors="pt", padding="max_length", max_length=2048, truncation=True).cuda()
    inputs2 = tokenizer.encode(" ".join(prediction_lines), return_tensors="pt", padding="max_length", max_length=2048, truncation=True).cuda()
    embedding1 = model(inputs1)[0]
    embedding2 = model(inputs2)[0]
    similarity = torch.nn.CosineSimilarity(dim=1)(embedding1.mean(dim=1), embedding2.mean(dim=1)).cpu()

    return_dict = {
        "es_score": ES_score,
        "native_es_score": native_es_score,
        "codesage_similarity": similarity.item(),
    }
    return return_dict

def evaluate_func(params):
    c_func, c_test, c_func_decompile = (
        params["c_func"],
        params["c_test"],
        params["c_func_decompile"],
    )

    timeout = 10
    flag_func_compile = 0
    flag_compile = 0
    flag_fix_compile = 0
    flag_run = 0
    flag_fix_run = 0
    c_include = ""
    for line in c_func.split("\n"):
        if "#include" in line:
            c_include += line + "\n"
            c_func = c_func.replace(line, "")
    for line in c_test.split("\n"):
        if "#include" in line:
            c_include += line + "\n"
            c_test = c_test.replace(line, "")
    func_name = None
    try:
        c_func_decompile_only = ""
        for line in c_func_decompile.split("\n"):
            if "#include" not in line:
                c_func_decompile_only += line + "\n"

        c_combine = c_include + "\n" + c_func_decompile + "\n" + c_test
        func_name = c_func_decompile_only.split("(")[0].split()[-1]
        func_name = func_name.replace("*", "")
        func_name = func_name.strip()
    except Exception as e:
        func_name = None
    if func_name is not None and len(func_name) > 0:
        c_test = c_test.replace("func0", func_name)
    c_combine_fix = c_include + "\n" + c_func_decompile + "\n" + c_test
    c_onlyfunc = c_include + "\n" + c_func_decompile

    with tempfile.TemporaryDirectory() as temp_dir:
        pid = os.getpid()
        c_file = os.path.join(temp_dir, f"combine_{pid}.c")
        executable = os.path.join(temp_dir, f"combine_{pid}")
        c_file_onlyfunc = os.path.join(temp_dir, f"onlyfunc_{pid}.c")
        executable_onlyfunc = os.path.join(temp_dir, f"onlyfunc_{pid}")
        c_file_fix = os.path.join(temp_dir, f"combine_fix_{pid}.c")
        executable_fix = os.path.join(temp_dir, f"combine_fix_{pid}")

        with open(c_file, "w") as f:
            f.write(c_combine.encode('ascii', 'ignore').decode('ascii'))
        with open(c_file_fix, "w") as f:
            f.write(c_combine_fix.encode('ascii', 'ignore').decode('ascii'))
        with open(c_file_onlyfunc, "w") as f:
            f.write(c_onlyfunc.encode('ascii', 'ignore').decode('ascii'))

        # Compile the C program to an assembly
        compile_command = [
            "gcc",
            "-S",
            c_file_onlyfunc,
            "-o",
            executable_onlyfunc,
            "-lm",
        ]
        try:
            subprocess.run(compile_command, check=True, timeout=timeout)
            flag_func_compile = 1
        except Exception as e:
            print(e)
            return func_name, flag_func_compile, flag_compile, flag_fix_compile, flag_run, flag_fix_run, c_combine, c_combine_fix, c_onlyfunc
            # pass
        
        # Compile the C program to an executable
        compile_command = ["gcc", c_file, "-o", executable, "-lm"]
        try:
            subprocess.run(compile_command, check=True, timeout=timeout)
            flag_compile = 1
        except Exception as e:
            print(e)
            return func_name, flag_func_compile, flag_compile, flag_fix_compile, flag_run, flag_fix_run, c_combine, c_combine_fix, c_onlyfunc
            # pass

        # Run the compiled executable
        run_command = [executable]
        try:
            process = subprocess.run(
                run_command, capture_output=True, text=True, timeout=timeout, check=True
            )
            flag_run = 1
        except Exception as e:
            print(e)
            # if "process" in locals() and process:
            #     process.kill()
            #     process.wait()
            return func_name, flag_func_compile, flag_compile, flag_fix_compile, flag_run, flag_fix_run, c_combine, c_combine_fix, c_onlyfunc
            # pass
        
        # Compile the C program to an executable
        compile_command = ["gcc", c_file_fix, "-o", executable_fix, "-lm"]
        try:
            subprocess.run(compile_command, check=True, timeout=timeout)
            flag_fix_compile = 1
        except Exception as e:
            print(e)
            return func_name, flag_func_compile, flag_compile, flag_fix_compile, flag_run, flag_fix_run, c_combine, c_combine_fix, c_onlyfunc
            # pass

        # Run the compiled executable
        run_command = [executable_fix]
        try:
            process = subprocess.run(
                run_command, capture_output=True, text=True, timeout=timeout, check=True
            )
            flag_fix_run = 1
        except Exception as e:
            print(e)
            # if "process" in locals() and process:
            #     process.kill()
            #     process.wait()
            return func_name, flag_func_compile, flag_compile, flag_fix_compile, flag_run, flag_fix_run, c_combine, c_combine_fix, c_onlyfunc
            # pass

    return func_name, flag_func_compile, flag_compile, flag_fix_compile, flag_run, flag_fix_run, c_combine, c_combine_fix, c_onlyfunc


def decompile_pass_rate(testsets, gen_results_repeat, opts:str = {'O0', 'O1', 'O2', 'O3'}, num_workers:int = 16):
    all_stats = []

    for gen_index, gen_results in enumerate(gen_results_repeat):
        print(f"Loop {gen_index}: {len(gen_results)}, {len(testsets)}")
        with multiprocessing.Pool(num_workers) as pool:
            tasks = [
                {
                    "c_func": testset["c_func"],
                    "c_test": testset["c_test"],
                    "c_func_decompile": output[0],
                }
                for testset, output in zip(testsets, gen_results)
            ]

            eval_results = list(tqdm(pool.imap(evaluate_func, tasks), total=len(tasks), desc="Evaluating"))

        pool.terminate()
        pool.join()

        # stats = {opt: {"func_compile": 0, "compile": 0, "fix_compile": 0, "run": 0, "fix_run": 0, "total": 0, "es_score": 0, "native_es_score": 0, "codesage_similarity": 0, "readability": defaultdict(float)} for opt in opts}
        stats = {opt: defaultdict(float) for opt in opts}
        output_info = []
        for idx, (testset, output, flag) in enumerate(
            tqdm(
                zip(testsets, gen_results, eval_results),
                total=len(testsets),
                desc="Evaluating",
            )
        ):
            c_func_decompile = output[0]
            c_func = testset["c_func"]
            c_test = testset["c_test"]

            func_name, flag_func_compile, flag_compile, flag_fix_compile, flag_run, flag_fix_run, c_combine, c_combine_fix, c_onlyfunc = flag
            opt = testset["type"]
            if flag_run:
                readability = compute_ES(c_func, c_func_decompile)
            else:
                readability = compute_ES(c_func, testset["input_asm_prompt"])

            stats[opt]["total"] += 1
            if flag_func_compile:
                stats[opt]["func_compile"] += 1
            if flag_compile:
                stats[opt]["compile"] += 1
            if flag_fix_compile:
                stats[opt]["fix_compile"] += 1
            if flag_run:
                stats[opt]["run"] += 1
            if flag_fix_run:
                stats[opt]["fix_run"] += 1
            
            for readability_stat in readability:
                # stats[opt]["readability"][readability_stat] += readability[readability_stat]
                stats[opt][readability_stat] += readability[readability_stat]
            
            testset.update({
                "compile": flag_compile,
                "run": flag_run,
                "func_name": func_name, 
                "func_compile": flag_func_compile,
                "fix_compile": flag_fix_compile,
                "fix_run": flag_fix_run,
                "readability": readability,
                "c_combine": c_combine,
                "c_combine_fix": c_combine_fix,
                "c_onlyfunc": c_onlyfunc,
            })
            output_info.append(testset)

        all_stats.append(stats)
        import pandas as pd
        output_info = pd.DataFrame(output_info)

    # average
    # avg_stats = {opt: {"compile": 0, "run": 0, "func_compile": 0, "fix_compile": 0, "fix_run": 0, "readability": defaultdict(float), "total": 0} for opt in opts}
    # avg_stats["all"] = {"compile": 0, "run": 0, "func_compile": 0, "fix_compile": 0, "fix_run": 0, "readability": defaultdict(float), "total": 0}
    avg_stats = {opt: defaultdict(float) for opt in opts}
    avg_stats["all"] = defaultdict(float)
    for stats in all_stats:
        for opt in opts:
            avg_stats[opt]["compile"] += stats[opt]["compile"]
            avg_stats[opt]["run"] += stats[opt]["run"]
            avg_stats[opt]["func_compile"] += stats[opt]["func_compile"]
            avg_stats[opt]["fix_compile"] += stats[opt]["fix_compile"]
            avg_stats[opt]["fix_run"] += stats[opt]["fix_run"]
            avg_stats[opt]["es_score"] += stats[opt]["es_score"]
            avg_stats[opt]["native_es_score"] += stats[opt]["native_es_score"]
            avg_stats[opt]["codesage_similarity"] += stats[opt]["codesage_similarity"]
            avg_stats[opt]["total"] += stats[opt]["total"]

            # for readability_stat in stats[opt]["readability"]:
            #     stats[opt]["readability"][readability_stat] += readability[readability_stat]
            avg_stats["all"]["compile"] += stats[opt]["compile"]
            avg_stats["all"]["run"] += stats[opt]["run"]
            avg_stats["all"]["func_compile"] += stats[opt]["func_compile"]
            avg_stats["all"]["fix_compile"] += stats[opt]["fix_compile"]
            avg_stats["all"]["fix_run"] += stats[opt]["fix_run"]
            avg_stats["all"]["es_score"] += stats[opt]["es_score"]
            avg_stats["all"]["native_es_score"] += stats[opt]["native_es_score"]
            avg_stats["all"]["codesage_similarity"] += stats[opt]["codesage_similarity"]
            avg_stats["all"]["total"] += stats[opt]["total"]
            # for readability_stat in stats[opt]["readability"]:
            #     avg_stats["all"]["readability"][readability_stat] += readability[readability_stat]

    for opt in opts:
        avg_stats[opt]["compile"] /= len(gen_results_repeat)
        avg_stats[opt]["run"] /= len(gen_results_repeat)
        avg_stats[opt]["func_compile"] /= len(gen_results_repeat)
        avg_stats[opt]["fix_compile"] /= len(gen_results_repeat)
        avg_stats[opt]["fix_run"] /= len(gen_results_repeat)
        avg_stats[opt]["total"] /= len(gen_results_repeat)
        avg_stats[opt]["es_score"] /= len(gen_results_repeat)
        avg_stats[opt]["native_es_score"] /= len(gen_results_repeat)
        avg_stats[opt]["codesage_similarity"] /= len(gen_results_repeat)
        # for readability_stat in avg_stats[opt]["readability"]:
        #     avg_stats[opt]["readability"][readability_stat] /= len(gen_results_repeat)
    avg_stats["all"]["compile"] /= len(gen_results_repeat)
    avg_stats["all"]["run"] /= len(gen_results_repeat)
    avg_stats["all"]["func_compile"] /= len(gen_results_repeat)
    avg_stats["all"]["fix_compile"] /= len(gen_results_repeat)
    avg_stats["all"]["fix_run"] /= len(gen_results_repeat)
    avg_stats["all"]["total"] /= len(gen_results_repeat)
    avg_stats["all"]["es_score"] /= len(gen_results_repeat)
    avg_stats["all"]["native_es_score"] /= len(gen_results_repeat)
    avg_stats["all"]["codesage_similarity"] /= len(gen_results_repeat)
    # for readability_stat in avg_stats["all"]["readability"]:
    #     avg_stats["all"]["readability"][readability_stat] /= len(gen_results_repeat)

    for opt, data in avg_stats.items():
        compile_rate = data["compile"] / data["total"] if data["total"] > 0 else 0
        run_rate = data["run"] / data["total"] if data["total"] > 0 else 0
        func_compile_rate = data["func_compile"] / data["total"] if data["total"] > 0 else 0
        fix_compile_rate = data["fix_compile"] / data["total"] if data["total"] > 0 else 0
        fix_run_rate = data["fix_run"] / data["total"] if data["total"] > 0 else 0
    

        avg_stats[opt]["func_compile_rate"] = func_compile_rate
        avg_stats[opt]["fix_compile_rate"] = fix_compile_rate
        avg_stats[opt]["fix_run_rate"] = fix_run_rate
        avg_stats[opt]["compile_rate"] = compile_rate
        avg_stats[opt]["run_rate"] = run_rate

        avg_stats[opt]["es_score_rate"] = data["es_score"] / data["total"] if data["total"] > 0 else 0
        avg_stats[opt]["native_es_score_rate"] = data["native_es_score"] / data["total"] if data["total"] > 0 else 0
        avg_stats[opt]["codesage_similarity_rate"] = data["codesage_similarity"] / data["total"] if data["total"] > 0 else 0
        # for readability_stat in avg_stats[opt]["readability"]:
        #     avg_stats[opt][f"readability_{readability_stat}_rate"] = data["readability"][readability_stat] / data["total"] if data["total"] > 0 else 0
        print(
            f"Optimization {opt}: Compile Rate: {compile_rate:.4f}, Run Rate: {run_rate:.4f}"
        )

    # sort avg_stats
    avg_stats = dict(sorted(avg_stats.items(), key=lambda x: x[0], reverse=True))
    return avg_stats, output_info