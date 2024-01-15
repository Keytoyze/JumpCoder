from jumpcoder import JumpCoder
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--port_infilling", type=int, default=12345)
parser.add_argument("--port_generation", type=int, default=12346)
args = parser.parse_args()

problem_list = [
    "\ndef minSubArraySum(nums):\n    \"\"\"\n    Given an array of integers nums, find the minimum sum of any non-empty sub-array\n    of nums.\n    Example\n    minSubArraySum([2, 3, 4, 1, 2, 4]) == 1\n    minSubArraySum([-1, -2, -3]) == -6\n    \"\"\"\n"
]


for problem in problem_list:

    print("~~~~~~~~~~~~~~~~~ Using Autoregression ~~~~~~~~~~~~~~~~~")
    JumpCoder(
        port_generation=args.port_generation,
        port_infilling=args.port_infilling,
        language="Python",
        stop_tokens=["<EOT>", "\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n\"\"\""] # HumanEval's stop tokens
    ).generate(problem, enable_jumpcoder=False)


    print("~~~~~~~~~~~~~~~~~ Using JumpCoder ~~~~~~~~~~~~~~~~~")
    JumpCoder(
        port_generation=args.port_generation,
        port_infilling=args.port_infilling,
        language="Python",
        stop_tokens=["<EOT>", "\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n\"\"\""] # HumanEval's stop tokens
    ).generate(problem, enable_jumpcoder=True)