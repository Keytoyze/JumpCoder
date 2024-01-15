from abc import abstractmethod
from typing import List, Tuple
from contextlib import contextmanager
import os
import shutil
import random

class LanguageSpec:

    @staticmethod
    def of(language_name: str) -> "LanguageSpec":
        from .python import PythonSpec
        from .java import JavaSpec
        from .cpp import CppSpec

        specs = [
            PythonSpec(), 
            JavaSpec(), 
            CppSpec()
        ]
        language_name = language_name.lower()
        for spec in specs:
            if spec.get_name().lower() == language_name:
                return spec
        raise ValueError(f"Unknown language: {language_name}")
    
    @abstractmethod
    def get_name(self) -> str: pass

    @abstractmethod
    def infilling_bad_words(self) -> List[str]: pass

    @abstractmethod
    def has_undefined_symbols(self, code_text: str) -> bool: pass

    @abstractmethod
    def is_function(self, code_text: str) -> bool: pass

    @abstractmethod
    def is_import(self, code_text: str) -> bool: pass

    @abstractmethod
    def can_address_undefined_symbols(self, code_text: str, infilling_code: str, combine_code: str) -> bool: pass

    @abstractmethod
    def is_in_comment(self, code_lines: List[str], index: int) -> bool: pass

    @abstractmethod
    def is_illegal_infilling(self, infilling_line: str) -> bool: pass

    @abstractmethod
    def extract_first_function(self, code_text: str) -> Tuple[List[str], bool]: pass

    @abstractmethod
    def evaluate(self, code_text: str, reference: str) -> Tuple[str, float]: pass

@contextmanager
def create_temp_folder():
    temp_dir = os.path.join("code_eval", "cache", f"temp_{random.randint(1, 10000000)}")
    os.makedirs(temp_dir)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)