
from prompts import performancePrompt
from prompts import decompilePrompt
from prompts import securePrompt

def get_full_prompt(data_point, task, method, src_code_length=-1, tgt_code_length=-1):
    if task == "performance":
        prompt = get_user_prompt(data_point, task, method, src_code_length=src_code_length) + get_assistant_prompt(data_point, task, method, tgt_code_length=tgt_code_length)
    elif task == "decompile":
        prompt = get_user_prompt(data_point, task, method, src_code_length=src_code_length) + get_assistant_prompt(data_point, task, method, tgt_code_length=tgt_code_length)
    elif task == "secure":
        prompt = get_user_prompt(data_point, task, method, src_code_length=src_code_length) + get_assistant_prompt(data_point, task, method, tgt_code_length=tgt_code_length)
    else:
        raise ValueError(f"Task {task} not supported")
    return prompt

def get_user_prompt(data_point, task, method, src_code_length=-1):
    if task == "performance":
        prompt = performancePrompt.user_prompt[method].format(
            src_code=data_point['src_code'][:src_code_length],
        )
    elif task == "decompile":
        prompt = decompilePrompt.user_prompt[method].format(
            src_code=data_point['src_code'][:src_code_length]
        )
    elif task == "secure":
        prompt = securePrompt.user_prompt[method].format(
            src_code=data_point['src_code'][:src_code_length],
        )
    else:
        raise ValueError(f"Task {task} not supported")
    return prompt


def get_assistant_prompt(data_point, task, method, tgt_code_length=-1):
    if method == "base-spec":
        use_edit_rule = False
        use_correctness_spec = False
    if method == "correct-spec":
        use_edit_rule = False
        use_correctness_spec = True
    if method == "edit-spec":
        use_edit_rule = True
        use_correctness_spec = False
    if method == "correct-edit-spec":
        use_edit_rule = True
        use_correctness_spec = True
    if (data_point["edit_rules"] is None) or (data_point["edit_rules"] == "None") or (data_point["edit_rules"].strip() == ""):
        use_edit_rule = False
    if (data_point["functional_specification"] is None) or (data_point["functional_specification"] == "None") or (data_point["functional_specification"].strip() == ""):
        use_correctness_spec = False
    if task == "performance":
        templatePrompt = performancePrompt.assistant_prompt
    elif task == "decompile":
        templatePrompt = decompilePrompt.assistant_prompt
    elif task == "secure":
        templatePrompt = securePrompt.assistant_prompt
    else:
        raise ValueError(f"Task {task} not supported")
    if (not use_edit_rule) and (not use_correctness_spec):
        prompt = templatePrompt.assistant_prompt['base-spec'].format(
            tgt_code=data_point['tgt_code'][:tgt_code_length],
        )
    elif (not use_edit_rule) and use_correctness_spec:
        prompt = templatePrompt.assistant_prompt['correct-spec'].format(
            functional_specification=data_point['functional_specification'],
            input=data_point['input_specification'],
            output=data_point['output_specification'],
            tgt_code=data_point['tgt_code'][:tgt_code_length],
        )
    elif use_edit_rule and (not use_correctness_spec):
        prompt = templatePrompt.assistant_prompt['edit-spec'].format(
            edit_rules=data_point['edit_rules'],
            tgt_code=data_point['tgt_code'][:tgt_code_length],
        )
    elif use_edit_rule and use_correctness_spec:
        prompt = templatePrompt.assistant_prompt['correct-edit-spec'].format(
            edit_rules=data_point['edit_rules'],
            functional_specification=data_point['functional_specification'],
            input=data_point['input_specification'],
            output=data_point['output_specification'],
            tgt_code=data_point['tgt_code'][:tgt_code_length],
        )
    else:
        raise ValueError(f"Method {method} not supported in Task {task}")
    return prompt