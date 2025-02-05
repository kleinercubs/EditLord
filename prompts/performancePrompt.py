full_prompt = {
    "base-spec": """This is the slow code:
[SLOW CODE]
{src_code}
[/SLOW CODE]
The corresponding fast code is:
[FAST CODE]
{tgt_code}
[/FAST CODE]
""",
    "correct-spec": """This is the slow code:
[SLOW CODE]
{src_code}
[/SLOW CODE]
The given code describes the following problem: {functional_specification}
The input specification is: {input_specification}
The output specification is: {output_specification}
The corresponding fast code is:
[FAST CODE]
{tgt_code}
[/FAST CODE]
""",
    "edit-spec": """This is the slow code:
[SLOW CODE]
{src_code}
[/SLOW CODE]
Following editing rules should be applied: {editing_rules}
The corresponding fast code is:
[FAST CODE]
{tgt_code}
[/FAST CODE]
""",
    "correct-edit-spec":"""This is the slow code:
[SLOW CODE]
{src_code}
[/SLOW CODE]
The given code describes the following problem: {functional_specification}
The input specification is: {input_specification}
The output specification is: {output_specification}
Following editing rules should be applied: {editing_rules}
The corresponding fast code is:
[FAST CODE]
{tgt_code}
[/FAST CODE]
"""
}

user_prompt = {
    "base-spec": """This is the slow code:
[SLOW CODE]
{src_code}
[/SLOW CODE]
The corresponding fast code is:
[FAST CODE]
""",
    "correct-spec": """This is the slow code:
[SLOW CODE]
{src_code}
[/SLOW CODE]
The given code describes the following problem: """,
    "edit-spec": """This is the slow code:
[SLOW CODE]
{src_code}
[/SLOW CODE]
Following editing rules should be applied: """,
    "correct-edit-spec":"""This is the slow code:
[SLOW CODE]
{src_code}
[/SLOW CODE]
The given code describes the following problem: """
}

assistant_prompt = {
    "base-spec": """{tgt_code}
[/FAST CODE]
""",
    "correct-spec": """{functional_specification}
The input specification is: {input_specification}
The output specification is: {output_specification}
The corresponding fast code is:
[FAST CODE]
{tgt_code}
[/FAST CODE]
""",
    "edit-spec": """{editing_rules}
The corresponding fast code is:
[FAST CODE]
{tgt_code}
[/FAST CODE]
""",
    "correct-edit-spec":"""{functional_specification}
The input specification is: {input_specification}
The output specification is: {output_specification}
Following editing rules should be applied: {editing_rules}
The corresponding fast code is:
[FAST CODE]
{tgt_code}
[/FAST CODE]
"""
}