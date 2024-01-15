from .language import LanguageSpec
import ast
import builtins
from typing import List, Tuple
from jumpcoder.code_eval.execute import check_correctness

class PythonSpec(LanguageSpec):

    def get_name(self) -> str:
        return "python"

    def get_undefined_symbols(self, code_text: str) -> bool:
        try:
            tree = ast.parse(code_text)
        except:
            return set()
        undefined_symbols = set()
        excluded_symbols = set(dir(builtins)).union(['List', 'Tuple', 'Optional', 'Any']) # HumanEval's typing
        class UndefinedSymbolVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    symbol_name = node.id
                    if symbol_name not in excluded_symbols:
                        undefined_symbols.add(symbol_name)
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        excluded_symbols.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                excluded_symbols.add(elt.id)
                self.generic_visit(node)
            def visit_Import(self, node):
                for alias in node.names:
                    excluded_symbols.add(alias.asname or alias.name)
                self.generic_visit(node)
            def visit_ImportFrom(self, node):
                self.visit_Import(node)
            def visit_FunctionDef(self, node):
                try:
                    for arg in node.args.args:
                        excluded_symbols.add(arg.arg)
                    if node.args.vararg:
                        excluded_symbols.add(node.args.vararg.arg)
                    if node.args.kwarg:
                        excluded_symbols.add(node.args.kwarg.arg)
                    excluded_symbols.add(node.name)
                except: pass
                self.generic_visit(node)
            def visit_Lambda(self, node):
                self.visit_FunctionDef(node)
            def visit_For(self, node):
                if isinstance(node.target, ast.Tuple):
                    for elt in node.target.elts:
                        if isinstance(elt, ast.Name):
                            excluded_symbols.add(elt.id)
                elif isinstance(node.target, ast.Name):
                    excluded_symbols.add(node.target.id)
                    self.generic_visit(node)
            def visit_ListComp(self, node):
                if isinstance(node.elt, ast.Name):
                    excluded_symbols.add(node.elt.id)
                for generator in node.generators:
                    self.visit(generator)
                self.visit(node.elt)
                self.generic_visit(node)
            def visit_GeneratorExp(self, node):
                for generator in node.generators:
                    self.visit(generator)
                self.generic_visit(node)
            def visit_comprehension(self, node):
                if isinstance(node.target, ast.Tuple):
                    for elt in node.target.elts:
                        if isinstance(elt, ast.Name):
                            excluded_symbols.add(elt.id)
                elif isinstance(node.target, ast.Name):
                    excluded_symbols.add(node.target.id)
                self.generic_visit(node)
        visitor = UndefinedSymbolVisitor()
        visitor.visit(tree)
        return undefined_symbols
    

    def infilling_bad_words(self) -> List[str]: 
        return []
    
    def has_undefined_symbols(self, code_text: str) -> bool:
        undefined_symbols = self.get_undefined_symbols(code_text)
        return len(undefined_symbols) != 0
    
    def is_function(self, code_text: str) -> bool:
        return code_text.lstrip().startswith("def")
    
    def is_import(self, code_text: str) -> bool:
        return code_text.lstrip().startswith(("from", "import"))

    def can_address_undefined_symbols(self, code_text: str, infilling_lines: str, combine_code: str) -> bool:
        identifiers = []
        if self.is_function(infilling_lines):
            identifiers = [infilling_lines.split("def")[1].split("(")[0].strip()]
        elif self.is_import(infilling_lines):
            if "\n" in infilling_lines:
                infilling_lines = infilling_lines.split("\n")[0]
            identifiers = infilling_lines.strip().split("import")[-1].split(",")
            identifiers = list(map(lambda x: x.strip(), identifiers))
        if len(identifiers) == 0: return False

        undefined_symbol = self.get_undefined_symbols(code_text)

        for identifier in identifiers:
            if identifier in undefined_symbol:
                return True

    def is_in_comment(self, code_lines: List[str], index: int) -> bool:
        if code_lines[index].strip().startswith(('"""', '#')):
            return True
        is_in_comment = False
        for i in range(len(code_lines)):
            if '"""' in code_lines[i]: 
                is_in_comment = not is_in_comment
            
            if index == i and is_in_comment:
                return True
        return False


    def is_illegal_infilling(self, infilling_line: str) -> bool:
        return infilling_line.lstrip().startswith(("for", "if", "while", "try", "#!/"))
    
    def extract_first_function(self, code_text: str) -> Tuple[List[str], bool]:
        init_indent_count = -1
        lines = []
        has_other_external_code = False
        for cur_line in code_text.split("\n"):
            if len(cur_line.strip()) == 0: continue
            cur_indent = len(cur_line) - len(cur_line.lstrip())
            if init_indent_count == -1:
                init_indent_count = cur_indent
            elif init_indent_count == cur_indent:
                has_other_external_code = True
                break
            lines.append(cur_line)
        return lines, has_other_external_code
    
    def evaluate(self, code_text: str, reference: str) -> Tuple[str, float]:
        code = code_text
        for _ in range(10):
            try:
                exec(code, {})
                break
            except:
                code = "\n".join(code.split("\n")[:-1])
        try:
            exec(code, {})
        except:
            return "Compile error", -3
        code = code + "\n" + reference
        result = check_correctness(code, 3, 0, 0)
        if result['passed']:
            return "Pass", 0
        else:
            return result['result'], -1