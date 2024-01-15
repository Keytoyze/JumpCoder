import json
import argparse
import os
from jumpcoder.language_spec.language import LanguageSpec
from pathlib import Path
from jumpcoder import JumpCoder
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--port_infilling", type=int, default=12345)
parser.add_argument("--port_generation", type=int, default=12345)
parser.add_argument("--prompt_format", type=str, 
                    choices=['plain', 'wizardcoder'],
                    default='plain')
parser.add_argument("--save_name", type=str)
parser.add_argument("--dataset_path", type=str, default='dataset/humaneval.json')

parser.add_argument("--n_max_lines", type=int, default=64)
parser.add_argument("--similar_threshold", type=float, default=0.85)
parser.add_argument("--threshold_improvement", type=float, default=0.3)
parser.add_argument("--topk_infilling", type=int, default=5)
parser.add_argument("--infill_comment", action='store_true')
parser.add_argument("--no_infill_prompt", action='store_true')
parser.add_argument("--no_ast", action='store_true')
parser.add_argument("--parallel_generate_with_infill", action='store_true')
parser.add_argument("--parallel_evaluate_score", action="store_true")
parser.add_argument("--speculative_infill", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()


def load_json(data_path):
    with open(data_path, 'r', encoding='utf8') as fp:
        load_dict = json.load(fp)
    return load_dict


def compute_pass_1(data, key):
    return sum([1 if x[key][1] == 0 else 0 for x in data]) / len(data)


if __name__ == "__main__":
    json_dict = load_json(args.dataset_path)
    data = []
    if os.path.exists(args.save_name):
        data = json.load(open(args.save_name))
    
    jump_coder = JumpCoder(
        port_infilling=args.port_infilling,
        port_generation=args.port_generation,
        prompt_format=args.prompt_format,
        n_max_lines=args.n_max_lines,
        parallel_generation_with_infilling=args.parallel_generate_with_infill,
        speculative_infill=args.speculative_infill,
        parallel_evaluate_score=args.parallel_evaluate_score,
        infill_comment=args.infill_comment,
        infill_prompt=not args.no_infill_prompt,
        similar_threshold=args.similar_threshold,
        threshold_improvement=args.threshold_improvement,
        topk_infilling=args.topk_infilling,
        use_ast_judging=not args.no_ast,
        verbose=args.verbose
    )

    pbar = tqdm(json_dict)

    for i, task in enumerate(pbar):
        if i >= len(data):
            prompt = task['prompt']
            reference = task['reference']
            jump_coder.set_language(task['language'])
            jump_coder.stop_tokens = task['stop_tokens']
            output_j, output_a, select_j = jump_coder.generate_filtered(prompt)
            language_spec = LanguageSpec.of(task['language'])
            result_j, result_code_j = language_spec.evaluate(output_j, reference)
            result_a, result_code_a = language_spec.evaluate(output_a, reference)
            data.append({
                "task": task['entry_point'],
                "code_jumpcoder_v": output_j.split("\n"),
                "code_autoregression": output_a.split("\n"),
                "result_autoregression": [result_a, result_code_a],
                "result_jumpcoder_v": [result_j, result_code_j],
                "result_jumpcoder_f": [result_j, result_code_j] if select_j else [result_a, result_code_a],
                "result_jumpcoder_o": [result_j, result_code_j] if result_code_j == 0 else [result_a, result_code_a],
            })

            os.makedirs(Path(args.save_name).parent, exist_ok=True)
            with open(args.save_name, 'w') as fp:
                json.dump(data, fp, indent=4)
        
        pbar.set_description(f"A.R: {compute_pass_1(data, 'result_autoregression'):.3f} / J.C (V): {compute_pass_1(data, 'result_jumpcoder_v'):.3f} / J.C (F): {compute_pass_1(data, 'result_jumpcoder_f'):.3f} / J.C (O): {compute_pass_1(data, 'result_jumpcoder_o'):.3f}")