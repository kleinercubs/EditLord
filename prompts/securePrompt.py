full_prompt = {
    "base-spec": """This is the vulnerable code:
[VULNERABLE CODE]
{src_code}
[/VULNERABLE CODE]
The corresponding secure code is:
[SECURE CODE]
{tgt_code}
[/SECURE CODE]
""",
    "correct-spec": """This is the vulnerable code:
[VULNERABLE CODE]
{src_code}
[/VULNERABLE CODE]
The given code describes the following problem: {functional_specification}
The input specification is: {input_specification}
The output specification is: {output_specification}
The corresponding secure code is:
[SECURE CODE]
{tgt_code}
[/SECURE CODE]
""",
    "edit-spec": """This is the vulnerable code:
[VULNERABLE CODE]
{src_code}
[/VULNERABLE CODE]
Following editing rules should be applied: {editing_rules}
The corresponding secure code is:
[SECURE CODE]
{tgt_code}
[/SECURE CODE]
""",
    "correct-edit-spec":"""This is the vulnerable code:
[VULNERABLE CODE]
{src_code}
[/VULNERABLE CODE]
The given code describes the following problem: {functional_specification}
The input specification is: {input_specification}
The output specification is: {output_specification}
Following editing rules should be applied: {editing_rules}
The corresponding secure code is:
[SECURE CODE]
{tgt_code}
[/SECURE CODE]
"""
}

user_prompt = {
    "base-spec": """This is the vulnerable code:
[VULNERABLE CODE]
{src_code}
[/VULNERABLE CODE]
The corresponding secure code is:
[SECURE CODE]
""",
    "correct-spec": """This is the vulnerable code:
[VULNERABLE CODE]
{src_code}
[/VULNERABLE CODE]
The given code describes the following problem: """,
    "edit-spec": """This is the vulnerable code:
[VULNERABLE CODE]
{src_code}
[/VULNERABLE CODE]
Following editing rules should be applied: """,
    "correct-edit-spec":"""This is the vulnerable code:
[VULNERABLE CODE]
{src_code}
[/VULNERABLE CODE]
The given code describes the following problem: """
}

assistant_prompt = {
    "base-spec": """{tgt_code}
[/SECURE CODE]
""",
    "correct-spec": """{functional_specification}
The input specification is: {input_specification}
The output specification is: {output_specification}
The corresponding secure code is:
[SECURE CODE]
{tgt_code}
[/SECURE CODE]
""",
    "edit-spec": """{editing_rules}
The corresponding secure code is:
[SECURE CODE]
{tgt_code}
[/SECURE CODE]
""",
    "correct-edit-spec":"""{functional_specification}
The input specification is: {input_specification}
The output specification is: {output_specification}
Following editing rules should be applied: {editing_rules}
The corresponding secure code is:
[SECURE CODE]
{tgt_code}
[/SECURE CODE]
"""
}