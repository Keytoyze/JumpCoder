import os, sys
sys.path.append(os.getcwd())
from .language import LanguageSpec, create_temp_folder
from typing import List, Tuple
import re
from pathlib import Path
import hashlib
from jumpcoder.code_eval.safe_subprocess import run

class JavaSpec(LanguageSpec):

    def __init__(self) -> None:
        self.method_pattern = re.compile(r"^\s*(public|protected|private)?\s*(static|final|abstract)?\s*(\w+)\s+(\w+)\s*\([\s\S]*$")
        self.compile_cache = {}
    
    def get_name(self) -> str:
        return "java"

    def infilling_bad_words(self) -> List[str]: 
        return ["//", "/*"]

    def complete_brackets(self, java_code):
        open_brackets = java_code.count('{')
        close_brackets = java_code.count('}')
        missing_brackets = open_brackets - close_brackets

        # Append the missing closing brackets
        for _ in range(open_brackets - close_brackets):
            if missing_brackets == 2:
                java_code += "throw new RuntimeException();\n"
            java_code += "\n}"
            missing_brackets -= 1

        return java_code
    
    def get_num_of_undefined_symbols_cachable(self, code_text: str) -> int:
        md5 = hashlib.md5(code_text.strip().encode()).hexdigest()
        if md5 not in self.compile_cache:
            self.compile_cache[md5] = self.get_num_of_undefined_symbols(code_text)
        return self.compile_cache[md5]

    def prepare_java_sys_env(self):
        sys_env = os.environ.copy()
        javatuples_path = Path("code_eval/javatuples-1.2.jar")
        sys_env["CLASSPATH"] = f"{javatuples_path}"
        return sys_env

    def get_num_of_undefined_symbols(self, code_text: str) -> int:
        try:
            # Complete the brackets in the Java code
            completed_code = self.complete_brackets(code_text)

            with create_temp_folder() as out_dir:
                java_file_name = os.path.join(out_dir, "Problem.java")

                # Writing the completed Java code to a file
                with open(java_file_name, 'w') as file:
                    file.write(completed_code)

                # Compiling the Java code using subprocess
                result = run(['javac', java_file_name], env=self.prepare_java_sys_env())

            if "cannot find symbol" in result.stderr or "reason: actual and formal argument lists differ in length" in result.stderr:
                count = result.stderr.count("cannot find symbol") + result.stderr.count("reason: actual and formal argument lists differ in length")
                return count
            else:
                # if result.returncode != 0: breakpoint()
                return -1 if result.exit_code != 0 else 0
        except Exception as e:
            import traceback
            traceback.print_exc()
            return -1
    

    def has_undefined_symbols(self, code_text: str) -> bool:
        return self.get_num_of_undefined_symbols_cachable(code_text) > 0

    def is_function(self, code_text: str) -> bool:
        return self.method_pattern.match(code_text.split("\n")[0]) is not None
    
    def is_import(self, code_text: str) -> bool:
        return code_text.lstrip().startswith("import ")

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
            out_code_file = os.path.join(out_dir, "Problem.java")
            sys_env = self.prepare_java_sys_env()

            code = code_text
            code = code + "\n" + reference
            with open(out_code_file, "w") as file:
                file.write(code)
            compile_result = run(["javac", out_code_file], env=sys_env)
            if compile_result.exit_code != 0:
                return "Compile error: " + compile_result.stderr, -3
            try:
                run_result = run(["java", "-ea", "-cp", out_dir, "Problem"])
                if run_result.exit_code != 0:
                    return "Runtime Error: " + run_result.stderr , -1  
                else:
                    return "Pass", 0
            except:
                return "Time out", -2

if __name__ == "__main__":
    print(JavaSpec().is_function("        boolean isVowel(char c) {\n            return c == "))