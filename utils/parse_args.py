import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    if 'gemini' in sys.argv:
        parser.add_argument('--project_id', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--data', type=str, default='codenet')
    parser.add_argument('--output_file', type=str, default='output.txt')
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--enable_ir', action='store_true')
    parser.add_argument('--enable_preserver', action='store_true')
    parser.add_argument('--enable_testcase', action='store_true')
    parser.add_argument('--enable_cot', action='store_true')
    parser.add_argument('--enable_combination', action='store_true')
    parser.add_argument('--enable_strategy', action='store_true')
    parser.add_argument('--decompilation_option', default='')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--timeout', type=int, default=20)
    args = parser.parse_args()
    args.ext = 'py' if args.lang == 'python3' else 'cpp'
    args.base_dir = None
    print(args)
    return args