import queue
from typing import List, Tuple, Optional
from difflib import SequenceMatcher
import socket
import sys
from .objects import LineScore, InfillingRecord
from .language_spec.language import LanguageSpec
from . import utils
import threading
from dataclasses import dataclass, field
from collections import OrderedDict


def connect_socket(address, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((address, int(port)))
        return s
    except:
        print("===========================================================================")
        print("Connect failed. Please remember to run: python jumpcoder/run_llm_server.py")
        print("===========================================================================")
        raise


@dataclass
class JumpCoder:

    # generation param
    port_infilling: int
    port_generation: int
    language: str = 'Python'
    address_infilling: str = '127.0.0.1'
    address_generation: str = '127.0.0.1'
    prompt_format: Optional[str] = None  # None or wizardcoder
    n_max_lines: int = 64
    stop_tokens: List[str] = field(default_factory=lambda: [])
    verbose: bool = True

    # efficiency param
    parallel_generation_with_infilling: bool = True
    speculative_infill: bool = True
    parallel_evaluate_score: bool = False

    # infill position options
    infill_comment: bool = False
    infill_prompt: bool = True

    # hyperparameters
    similar_threshold: float = 0.85
    threshold_improvement: float = 0.3  # \tau
    topk_infilling: int = 5  # k

    # ablation studies
    use_ast_judging: bool = True

    # private
    __socket_infilling: socket.socket = field(init=False)
    __socket_generation: socket.socket = field(init=False)
    __language_spec: LanguageSpec = field(init=False)
    INFINITE_SCORE = 1000000

    def __post_init__(self) -> None:
        self.__socket_infilling = connect_socket(
            self.address_infilling, self.port_infilling)
        if self.port_generation != self.port_infilling or self.address_infilling != self.address_generation:
            self.__socket_generation = connect_socket(
                self.address_generation, self.port_generation)
        else:
            self.__socket_generation = self.__socket_infillingx
        self.set_language(self.language)
    
    def set_language(self, language: str):
        self.language = language
        self.__language_spec = LanguageSpec.of(language)

    def generate_next_line(self, prompt: str, message_queue: Optional[queue.Queue] = None) -> Tuple[str, bool]:
        response_data = utils.request_api(self.__socket_generation, {
            "request_type": "generate",
            "text": prompt,
            "multi_lines": False,
            "language":  self.__language_spec.get_name(),
            "stop_words": self.stop_tokens
        })
        next_lines = response_data['next_lines']
        is_terminal = response_data['is_terminal']
        if message_queue is not None:
            message_queue.put((next_lines, is_terminal))
        else:
            return next_lines, is_terminal

    def parallel_infill(self, text_list: List[str], allow_function_multi_lines: bool, speculative_infilling: bool) -> List[str]:
        if len(text_list) == 0:
            return []
        response_data = utils.request_api(self.__socket_infilling, {
            "request_type": "parallel_infill",
            "text_list": text_list,
            "stop_words": self.stop_tokens,
            "language": self.__language_spec.get_name(),
            "bad_words":  self.__language_spec.infilling_bad_words(),
            "speculative_infilling": speculative_infilling,
            "allow_function_multi_lines": allow_function_multi_lines
        })
        return response_data['list_of_next_lines']

    def evaluate_score(self, code_list: List[str]) -> List[List[LineScore]]:
        response = utils.request_api(self.__socket_generation, {
            "request_type": "evaluate_score",
            "parallel": self.parallel_evaluate_score,
            "text": code_list
        })
        line_scores_dumped = response["line_scores"]
        list_line_scores = response["line_scores"]
        new_list = []
        for line_scores_dumped in list_line_scores:
            new_list.append([LineScore(x[0], x[1])
                            for x in line_scores_dumped])
        return new_list

    def evaluate_ppl(self, code: str) -> float:
        response = utils.request_api(self.__socket_generation, {
            "request_type": "evaluate_ppl",
            "text": code
        })
        ppl = response["code_ppl"]
        return ppl

    def extract_code_from_lines(self, lines: List[str]) -> str:
        if self.prompt_format == "wizardcoder":
            code_text = utils.process_wizard_code("".join(lines))
        else:
            code_text = "".join(lines)
        return code_text

    def get_valid_infilling_positions(
        self,
        lines: List[str],
        generation_scores: List[LineScore],
        disable_infilling_lines: Optional[List[str]] = None
    ) -> List[int]:
        """
        Filter valid positions based on current lines before running infilling
        Return the valid line numbers in 0, 1, ..., N-1
        """
        if len(lines) != len(generation_scores):
            return []  # happens when the last generated line is TOO LOOOONG
        valid_position = []

        for i in range(len(lines)):

            if len(generation_scores[i].score) == 0:
                continue

            # only valid in ```python
            if self.prompt_format == "wizardcoder":
                for j in range(len(lines)):
                    if "```python" in lines[j]:
                        break
                if i <= j:
                    continue
            # filter based on disable_infilling_lines
            if disable_infilling_lines is not None and lines[i] in disable_infilling_lines:
                continue

            if not self.infill_comment:
                if self.__language_spec.is_in_comment(lines, i):
                    continue
            valid_position.append((i, generation_scores[i].score[0]))

        # select only the top-k positions
        valid_position = utils.find_top_k_by_y(
            valid_position, self.topk_infilling)
        return valid_position

    def do_hybrid_generation(
        self,
        lines: List[str],
        generation_scores: List[LineScore],
        disable_infilling_lines: List[str]
    ):
        """
        Return: infills_at_each_position (Dict[int, str]), generation_line (str), is_terminal (bool)
        """
        if self.parallel_generation_with_infilling:
            message_queue = queue.Queue()
            generation_thread = threading.Thread(
                target=self.generate_next_line,
                args=("".join(lines), message_queue)
            )
            generation_thread.start()

        # Infill
        valid_infilling_positions = self.get_valid_infilling_positions(
            lines, generation_scores, disable_infilling_lines)
        infills_at_each_position = OrderedDict()  # position -> next_lines
        parallel_input = []
        for i in valid_infilling_positions:
            cur_lines = lines.copy()
            cur_lines[i] = "<FILL_ME>" + cur_lines[i]
            parallel_input.append(self.extract_code_from_lines(cur_lines))
        infill_list = self.parallel_infill(
            parallel_input,
            self.__language_spec.has_undefined_symbols(
                self.extract_code_from_lines(lines)), 
            self.speculative_infill
        )
        for i, next_lines in zip(valid_infilling_positions, infill_list):
            infills_at_each_position[i] = next_lines

        if self.parallel_generation_with_infilling:
            generation_thread.join()
            generation_line, is_terminal = message_queue.get()
        else:
            generation_line, is_terminal = self.generate_next_line(
                "".join(lines))

        return infills_at_each_position, generation_line, is_terminal

    def do_combination(
        self,
        lines: List[str],
        infill_lines: str,
        index: int
    ) -> Optional[Tuple[List[str], int]]:
        """
        Insert the infill in current lines.
        Return None if the infill are invalid. 
        Otherwise, return the results and the number of infilled lines.
        """
        # disable empty lines
        if len(infill_lines) == 0:
            return None

        infill = infill_lines.split("\n")[0] + "\n"

        # disable repeat
        if infill in lines:
            return None
        if index > 0 and SequenceMatcher(None, infill, lines[index - 1]).ratio() > self.similar_threshold:
            return None
        if SequenceMatcher(None, infill, lines[index]).ratio() > self.similar_threshold:
            return None

        # disable language-spec illegal text
        if self.__language_spec.is_illegal_infilling(infill_lines):
            return None

        combined_lines = lines.copy()
        n_fill_lines = 0

        if self.__language_spec.is_function(infill_lines):
            infill_lines, _ = self.__language_spec.extract_first_function(
                infill_lines)
            for infill in infill_lines:
                infill += "\n"
                combined_lines.insert(index + n_fill_lines, infill)
                n_fill_lines += 1
        else:
            combined_lines.insert(index + n_fill_lines, infill)
            n_fill_lines += 1

        return combined_lines, n_fill_lines

    def do_judging(
        self,
        index: int,
        num_infilling_lines: int,
        infilling_lines: str,
        previous_lines: List[str],
        previous_scores: List[LineScore],
        cur_lines: List[str],
        cur_scores: List[LineScore],
        prompt_lines: List[str],
    ) -> InfillingRecord:
        n_lines_improve = 0
        total_lines_check = 0
        score_improvement = []

        for j in range(index + num_infilling_lines, len(cur_lines)):
            if len(cur_scores[j].score) != 0:
                total_lines_check += 1

        if self.use_ast_judging and (self.__language_spec.is_import(infilling_lines) or self.__language_spec.is_function(infilling_lines)):
            # Use AST Parser: judge if the next line address undefined identifiers
            if self.__language_spec.can_address_undefined_symbols(
                self.extract_code_from_lines(previous_lines),
                infilling_lines,
                self.extract_code_from_lines(cur_lines)
            ):
                n_lines_improve = -1
                score_improvement.append(self.INFINITE_SCORE)

        else:
            # Use Generation Model Scoring: compute improvement (delta) after the infilling code
            for j in range(index + num_infilling_lines, len(cur_lines)):
                previous_score_j = previous_scores[j - num_infilling_lines]
                utils.debug_assert(0 <= j < len(cur_scores))
                cur_score_j = cur_scores[j]
                if len(previous_score_j) != len(cur_score_j):
                    continue
                utils.debug_assert(len(previous_score_j) == len(cur_score_j))
                if len(cur_score_j.score) == 0:
                    continue
                current_total_score_j = sum(
                    cur_score_j.score) / len(cur_score_j.score)
                previous_total_score_j = sum(
                    previous_score_j.score) / len(previous_score_j.score)
                delta = current_total_score_j - previous_total_score_j

                if delta >= self.threshold_improvement:
                    n_lines_improve += 1
                    score_improvement.append(delta)
                else:
                    break

        if n_lines_improve >= min(2, total_lines_check) and n_lines_improve >= total_lines_check * 0.5:
            # PASS Judging by Generation Model Scoring!
            # We need to remove those lines without significant improvement.
            # However, if the removed lines are in the prompt_lines, disable this infill to avoid hurt prompt.
            for check_i in range(n_lines_improve + index + 1, len(cur_lines)):
                if cur_lines[check_i].strip() != "" and cur_lines[check_i] in prompt_lines:
                    return None
            return InfillingRecord(
                cur_lines=cur_lines[:n_lines_improve + index + 1],
                generation_scores=cur_scores[:n_lines_improve + index + 1],
                index=index,
                score_improvement=score_improvement
            )
        elif n_lines_improve < 0:
            # PASS Judging by AST Parser!
            return InfillingRecord(
                cur_lines=cur_lines,
                generation_scores=cur_scores,
                index=index,
                score_improvement=score_improvement
            )
        return None
    
    def display_combination(self, lines, index, type):
        if not self.verbose:
            return
        lines_disp = lines.copy()
        lines_disp[index] = f"{lines_disp[index].rstrip()}    # <---------------\n"
        print(f"===================\n{type}\n===================\n{''.join(lines_disp)}", file=sys.stderr)
    

    def generate_filtered(self, prompt: str) -> Tuple[str, str, bool]:
        """
        a.k.a. JumpCoder (Filtered)
        Select the result from JumpCoder (Vanilla) and autoregression.
        If all infills are judged by AST Parser instead of Generation Model Scoring, JumpCoder's output is chosen.
        Otherwise select the code with lower perplexity (PPL).
        
        :param
        @prompt: the problem prompt

        :return: [JumpCoder output, autoregression output, is JumpCoder selected]
        """

        jumpcoder_code, infill_record = self.generate(prompt, enable_jumpcoder=True)
        autoregression_code, _ = self.generate(prompt, enable_jumpcoder=False)

        contain_infill_from_scoring = False
        for record in infill_record:
            if record[2] != [self.INFINITE_SCORE]:
                contain_infill_from_scoring = True
                break
        if not contain_infill_from_scoring:
            return jumpcoder_code, autoregression_code, True
        
        ppl_jumpcoder = self.evaluate_ppl(jumpcoder_code)
        ppl_autoregression = self.evaluate_ppl(autoregression_code)
        return jumpcoder_code, autoregression_code, ppl_jumpcoder < ppl_autoregression


    def generate(self, prompt: str, enable_jumpcoder=True) -> Tuple[str, list]:
        """
        a.k.a. JumpCoder (Vanilla) 

        :param
        @prompt: the problem prompt

        :return: a tuple (generated result, infill records)
        - generated result
        - infill records, in the format List([position, infill, score])
        """

        lines = []
        infill_record = []
        count = 0

        lines.extend(x + "\n" for x in prompt.rstrip().split("\n"))
        if self.prompt_format == 'wizardcoder':
            lines[-1] = lines[-1].replace("\n", "")

        prompt_lines = lines.copy()
        if self.infill_prompt:
            disable_infilling_lines = []
        else:
            disable_infilling_lines = prompt_lines.copy()

        generation_scores = self.evaluate_score(["".join(lines)])[0]

        while True:

            if len(lines) >= self.n_max_lines or count >= self.n_max_lines:
                break
            count += 1

            generation_line = None

            if len(lines) != 0 and enable_jumpcoder:
                records: List[InfillingRecord] = []

                infills_at_each_position, generation_line, is_terminal = self.do_hybrid_generation(
                    lines, generation_scores, disable_infilling_lines
                )

                list_of_combine_code = []
                combination_at_each_position = {}

                # Insert
                for i in infills_at_each_position:
                    infill_lines = infills_at_each_position[i]
                    combined = self.do_combination(lines, infill_lines, i)
                    if combined is not None:
                        list_of_combine_code.append("".join(combined[0]))
                        combination_at_each_position[i] = combined

                # Judging
                list_of_current_scores = self.evaluate_score(
                    list_of_combine_code)

                for i, position in enumerate(combination_at_each_position):

                    cur_scores = list_of_current_scores[i]
                    combined_lines, n_fill_lines = combination_at_each_position[position]
                    infill_lines = infills_at_each_position[position]
                    improvement = self.do_judging(
                        index=position,
                        num_infilling_lines=n_fill_lines,
                        infilling_lines=infill_lines,
                        previous_lines=lines,
                        previous_scores=generation_scores,
                        cur_lines=combined_lines,
                        cur_scores=cur_scores,
                        prompt_lines=prompt_lines
                    )
                    if improvement is not None:
                        records.append(improvement)

                if len(records) > 0:
                    # Find improvement! Select the best one
                    best_record = max(records, key=lambda x: sum(x.score_improvement))
                    lines = best_record.cur_lines
                    generation_scores = best_record.generation_scores
                    infill_record.append((best_record.index, best_record.cur_lines[best_record.index], best_record.score_improvement))
                    self.display_combination(lines, best_record.index, "Online Modification")
                    continue

            # Infilling leads to worse case. Generation!
            if generation_line is None:
                generation_line, is_terminal = self.generate_next_line(
                    "".join(lines))

            if is_terminal:
                break

            if self.prompt_format == 'wizardcoder' and count == 1:
                generation_line = generation_line.split('\r\n')[0] + '\r\n'
                lines[-1] = lines[-1] + generation_line
            else:
                generation_line = generation_line.split('\n')[0] + '\n'
                lines.append(generation_line)

            self.display_combination(lines, -1, "Generation")

            generation_scores = self.evaluate_score(["".join(lines)])[0]

        return self.extract_code_from_lines(lines), infill_record

