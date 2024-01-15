import os, sys
sys.path.append(os.getcwd())
from .language import LanguageSpec, create_temp_folder
from typing import List, Tuple
import re
import hashlib
from jumpcoder.code_eval.safe_subprocess import run

class CppSpec(LanguageSpec):

    def __init__(self) -> None:
        self.method_pattern = re.compile(r"^\s*(static)?\s*(\w+)\s+(\w+)\s*\([\s\S]*$")
        self.anonymous_method_pattern = re.compile(r"\[.*?\]\s*\(")
        self.compile_cache = {}
    
    def get_name(self) -> str:
        return "cpp"

    def infilling_bad_words(self) -> List[str]: 
        return ["//", "/*"]

    def complete_brackets(self, cpp_code):
        open_brackets = cpp_code.count('{')
        close_brackets = cpp_code.count('}')
        missing_brackets = open_brackets - close_brackets

        # Append the missing closing brackets
        for _ in range(open_brackets - close_brackets):
            cpp_code += "\n}"
            missing_brackets -= 1
        return cpp_code
    
    def get_num_of_undefined_symbols_cachable(self, code_text: str) -> int:
        md5 = hashlib.md5(code_text.strip().encode()).hexdigest()
        if md5 not in self.compile_cache:
            self.compile_cache[md5] = self.get_num_of_undefined_symbols(code_text)
        return self.compile_cache[md5]


    def get_num_of_undefined_symbols(self, code_text: str) -> int:
        try:
            completed_code = self.complete_brackets(code_text)

            with create_temp_folder() as f:
                cpp_file_name = os.path.join(f, "test.cpp")

                with open(cpp_file_name, 'w') as file:
                    file.write(completed_code)

                result = run(['g++', "-c", cpp_file_name, "-std=c++17"])

            if "was not declared in this scope" in result.stderr:
                count = result.stderr.count("was not declared in this scope")
                return count
            else:
                return -1 if result.exit_code != 0 else 0
        except Exception as e:
            import traceback
            traceback.print_exc()
            return -1
    

    def has_undefined_symbols(self, code_text: str) -> bool:
        return self.get_num_of_undefined_symbols_cachable(code_text) > 0

    def is_function(self, code_text: str) -> bool:
        return self.method_pattern.search(code_text.split("\n")[0]) is not None or self.anonymous_method_pattern.search(code_text.split("\n")[0]) is not None
    
    def is_import(self, code_text: str) -> bool:
        return code_text.lstrip().startswith("#include")

    def can_address_undefined_symbols(self, code_text: str, infilling_lines: str, combine_code: str) -> bool:
        previous_num_undefined_symbols = self.get_num_of_undefined_symbols_cachable(code_text)
        if previous_num_undefined_symbols == 0:
            return False
        current_num_undefined_symbols = self.get_num_of_undefined_symbols_cachable(combine_code)
        return current_num_undefined_symbols < previous_num_undefined_symbols and current_num_undefined_symbols >= 0

    def is_in_comment(self, code_lines: List[str], index: int) -> bool:
        in_block_comment = False
        for i, line in enumerate(code_lines):
            if '/*' in line:
                in_block_comment = True
            if i == index:
                return line.lstrip().startswith("//") or in_block_comment
            if '*/' in line:
                in_block_comment = False
        return False


    def is_illegal_infilling(self, infilling_line: str) -> bool:
        return infilling_line.lstrip().startswith(("for", "if", "while", "try", "else"))
    
    def extract_first_function(self, code_text: str) -> Tuple[List[str], bool]:
        open_brackets = 0
        method_body = []
        in_block_comment = False
        code_after_method = False

        lines = code_text.split('\n')
        for i, line in enumerate(lines):
            # Check for the start or end of a block comment
            if '/*' in line:
                in_block_comment = True
            if '*/' in line:
                in_block_comment = False
                method_body.append(line)
                continue

            if not in_block_comment:
                # Check for the start of the method body
                if '{' in line:
                    open_brackets += line.count('{')
                    if open_brackets == 1 and not method_body:
                        # Start capturing the method body
                        method_body.append(line)
                        continue

                if open_brackets > 0:
                    method_body.append(line)

                # Check for the end of the method body
                if '}' in line:
                    open_brackets -= line.count('}')
                    if open_brackets == 0:
                        break

        # Check if there is code after the method
        if i + 1 < len(lines):
            code_after_method = any(line.strip() for line in lines[i+1:])

        return method_body, code_after_method


    
    def evaluate(self, code_text: str, reference: str) -> Tuple[str, float]:
        with create_temp_folder() as out_dir:
            out_code_file = os.path.join(out_dir, "test.cpp")
            basename = ".".join(str(out_code_file).split(".")[:-1])

            code = code_text
            code = code + "\n" + reference
            with open(out_code_file, "w") as file:
                file.write(code)

            build_result = run(["g++", out_code_file, "-lcrypto", "-o", basename, "-std=c++17"])
            if build_result.exit_code != 0:
                return "Compile error: " + build_result.stderr, -3
            try:
                run_result = run([basename])
                if run_result.timeout:
                    return "Timeout", -2
                elif run_result.exit_code != 0:
                    return f"Exception: {run_result.stderr}", -1
                else:
                    return "Pass", 0
            except:
                return "Time out", -2

if __name__ == "__main__":
    print(CppSpec().is_function("int main(int s) {"))
    print(CppSpec().is_function("auto s = []() {"))