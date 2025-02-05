full_prompt = {
    "base-spec": """This is the decompiled code:
[MACHINE DECOMPILED CODE]
{src_code}
[/MACHINE DECOMPILED CODE]
The corresponding original source code is:
[ORIGINAL SOURCE CODE]
{tgt_code}
[/ORIGINAL SOURCE CODE]
""",
    "correct-spec": """This is the decompiled code:
[MACHINE DECOMPILED CODE]
{src_code}
[/MACHINE DECOMPILED CODE]
The given code describes the following problem: {functional_specification}
The input specification is: {input_specification}
The output specification is: {output_specification}
The corresponding original source code is:
[ORIGINAL SOURCE CODE]
{tgt_code}
[/ORIGINAL SOURCE CODE]
""",
    "edit-spec": """This is the decompiled code:
[MACHINE DECOMPILED CODE]
{src_code}
[/MACHINE DECOMPILED CODE]
Following editing rules should be applied: {editing_rules}
The corresponding original source code is:
[ORIGINAL SOURCE CODE]
{tgt_code}
[/ORIGINAL SOURCE CODE]
""",
    "correct-edit-spec":"""This is the decompiled code:
[MACHINE DECOMPILED CODE]
{src_code}
[/MACHINE DECOMPILED CODE]
The given code describes the following problem: {functional_specification}
The input specification is: {input_specification}
The output specification is: {output_specification}
Following editing rules should be applied: {editing_rules}
The corresponding original source code is:
[ORIGINAL SOURCE CODE]
{tgt_code}
[/ORIGINAL SOURCE CODE]
"""
}

user_prompt = {
    "base-spec": """This is the decompiled code:
[MACHINE DECOMPILED CODE]
{src_code}
[/MACHINE DECOMPILED CODE]
The corresponding original source code is:
[ORIGINAL SOURCE CODE]
""",
    "correct-spec": """This is the decompiled code:
[MACHINE DECOMPILED CODE]
{src_code}
[/MACHINE DECOMPILED CODE]
The given code describes the following problem: """,
    "edit-spec": """This is the decompiled code:
[MACHINE DECOMPILED CODE]
{src_code}
[/MACHINE DECOMPILED CODE]
Following editing rules should be applied: """,
    "correct-edit-spec":"""This is the decompiled code:
[MACHINE DECOMPILED CODE]
{src_code}
[/MACHINE DECOMPILED CODE]
The given code describes the following problem: """
}

assistant_prompt = {
    "base-spec": """{tgt_code}
[/ORIGINAL SOURCE CODE]
""",
    "correct-spec": """{functional_specification}
The input specification is: {input_specification}
The output specification is: {output_specification}
The corresponding original source code is:
[ORIGINAL SOURCE CODE]
{tgt_code}
[/ORIGINAL SOURCE CODE]
""",
    "edit-spec": """{editing_rules}
The corresponding original source code is:
[ORIGINAL SOURCE CODE]
{tgt_code}
[/ORIGINAL SOURCE CODE]
""",
    "correct-edit-spec":"""{functional_specification}
The input specification is: {input_specification}
The output specification is: {output_specification}
Following editing rules should be applied: {editing_rules}
The corresponding original source code is:
[ORIGINAL SOURCE CODE]
{tgt_code}
[/ORIGINAL SOURCE CODE]
"""
}