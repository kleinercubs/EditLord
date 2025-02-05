"""
Finetune the model on the codellama dataset

Code adapted from the alpaca-lora repository at https://github.com/tloen/alpaca-lora/blob/main/finetune.py
"""

import os
import sys
import random
import numpy as np

import fire
import torch
import transformers
from prompts import promptManager

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from transformers import AutoModelForCausalLM, AutoTokenizer


import json

def train(
    # model/data params
    base_model: str = "codellama/CodeLlama-7b-hf", 
    data_path: str = "data/code_data",
    output_dir: str = "./code_opt/13b-test",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 2048,
    val_set_size: int = 2000,
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = True,
    group_by_length: bool = False,  
    # wandb params
    wandb_project: str = "code-llama",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "code_opt",  # The prompt template to use, will default to code_opt.
    use_wandb: bool = True, # if True, will use wandb if wandb_project is set
    # training data and prompt template
    train_name: str = "train.jsonl",
    val_name: str = "val.jsonl",
    test_name: str = "test.jsonl",
    with_speedup_desc: bool = False,  # if True, we use templates/code_opt_w_speedup_desc.json
    with_speedup_bin: bool = False,   # if True, we use templates/code_opt_w_speedup_bin.json
    task: str = "performance",
    method: str = "baseline",
    src_code_length: int = 0,
    tgt_code_length: int = 0,
):
    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    seed_everything(42)
    if with_speedup_desc and with_speedup_bin:
        raise ValueError("Both with_speedup_desc and with_speedup_bin can not be TRUE!!!")
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training code_opt-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"Train File: {os.path.join(data_path, train_name)}\n"
            f"Val File: {os.path.join(data_path, val_name)}\n"
            f"Test File: {os.path.join(data_path, test_name)}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if use_wandb:
        # Check if parameter passed or if set within environ
        use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
        )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # make sure to have latest version of transformers library and flash attention installed
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        cache_dir = '.cache'
    )

    print(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir = '.cache')

    tokenizer.pad_token_id = (
        0  # unk. 
    )
    tokenizer.padding_side = "left"  # Allow batched inference


    def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
        """Collects the state dict and dump to disk."""
        # state_dict = trainer.model.state_dict()
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )
        model=trainer.model  
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state_dict = model.state_dict()
        if trainer.args.should_save:
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point, task, method, src_code_length, tgt_code_length):
        full_prompt = promptManager.get_full_prompt(data_point, task, method, src_code_length, tgt_code_length)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = promptManager.get_user_prompt(data_point, task, method, src_code_length)
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    ## Loading data

    try:
        data = json.load(open(train_name))
    except:
        with open(train_name, "r") as f:
            data = []
            for line in f:
                data.append(json.loads(line))

    try:
        _val_data = json.load(open(val_name))
    except:
        with open(val_name, "r") as f:
            _val_data = []
            for line in f:
                _val_data.append(json.loads(line))

    train_data, val_data = [], []

    token_length = []
    for d in data:
        train_data.append(generate_and_tokenize_prompt(d, task, method, src_code_length, tgt_code_length))
    
    for d in _val_data:
        # d.update({'global_strategies': HIERARCHY})
        val_data.append(generate_and_tokenize_prompt(d, task, method, src_code_length, tgt_code_length))
    
    print(f"Max token length: {max(token_length)}")

    # model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=100 if val_set_size > 0 else None,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=10,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else "none",
            fsdp=["full_shard", "auto_wrap"],
            gradient_checkpointing=True,
            resume_from_checkpoint=f"{output_dir}" if resume_from_checkpoint else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    safe_save_model_for_hf_trainer(trainer, output_dir)
    # save tokenizer
    tokenizer.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)