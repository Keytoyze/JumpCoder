import socket
import argparse
import torch
from typing import List, Tuple, Dict
import math
import sys
import os
sys.path.append(os.getcwd())
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteria, StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM
from transformers import logging
logging.set_verbosity_error()

from jumpcoder.objects import LineScore
from jumpcoder import utils
from jumpcoder.language_spec.language import LanguageSpec

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=12345)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--bad_words", default='')
args = parser.parse_args()

def get_tokens_as_list(tokenizer_with_prefix_space, word_list):
    tokens_list = []
    for word in word_list:
        if word == "": continue
        if isinstance(word, int):
            tokens_list.append([word])
        else:
            tokenized_word = tokenizer_with_prefix_space(
                [word], add_special_tokens=False).input_ids[0]
            tokens_list.append(tokenized_word)
    return tokens_list


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer, language_spec: LanguageSpec, allow_function: bool):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        self.language_spec = language_spec
        self.allow_function = allow_function

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length:]
        )
        done = []
        for decoded_generation in decoded_generations:

            is_function = self.allow_function and self.language_spec.is_function(decoded_generation)
            eof_strings: list = self.eof_strings.copy()

            if is_function:
                if "\n" in eof_strings:
                    eof_strings.remove("\n")
                if "\ndef" in eof_strings:
                    eof_strings.remove("\ndef")
            
            should_stop = False
            for stop_string in eof_strings:
                if stop_string in decoded_generation:
                    should_stop = True
                    break
            if not should_stop and is_function:
                _, has_other_external_code = self.language_spec.extract_first_function(decoded_generation)
                if has_other_external_code:
                    should_stop = True
            
            done.append(should_stop)
        return all(done)


@dataclass
class ModelEnvironment:

    llm: PreTrainedModel
    new_line_token_id: int
    tokenizer: PreTrainedTokenizer
    tokenizer_with_prefix_space: PreTrainedTokenizer
    infilling_tokenizer: PreTrainedTokenizer
    bad_words_ids: List[torch.Tensor]
    # Saving the previous computed infix. Used for speculative infilling
    cache_prefix_to_infix: Dict[str, List[int]]

    def prompt_token_length(self, prompt) -> int:
        return self.tokenizer.encode(prompt, return_tensors="pt").shape[1]
    
    def infill_generate(self, input_ids, attention_masks, stopping_criteria, bad_word_ids):
        return self.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_new_tokens=256,
            do_sample=False,
            return_dict_in_generate=True,
            bad_words_ids=bad_word_ids,
            stopping_criteria=stopping_criteria).sequences

    def speculative_infill(self, prefix_list, input_ids, eos, attention_mask, stopping_criteria, bad_word_ids):

        # 1. select the infill candidates from cache for those repeated infill positions
        guess_inputs = []
        guess_attention_mask = []
        guess_intial_id = []
        for i, (prefix, input_id) in enumerate(zip(prefix_list, input_ids)):
            if prefix in self.cache_prefix_to_infix:
                infix = self.cache_prefix_to_infix[prefix].copy()
                if self.new_line_token_id in infix:
                    for j in range(infix.index(self.new_line_token_id) + 1, len(infix)):
                        infix[j] = eos
                guess_input = input_id.tolist() + infix
                guess_inputs.append(guess_input)
                guess_attention_mask.append(attention_mask[i])
                guess_intial_id.append(i)
        # If not found, invoke normal generate
        if len(guess_inputs) == 0:
            return self.infill_generate(input_ids, attention_mask, stopping_criteria, bad_word_ids)
        
        # 2. check the selected candidate
        input_length = max([len(x) for x in guess_inputs])
        candidate_length = input_length - input_ids.shape[1]
        for i in range(len(guess_inputs)):
            guess_inputs[i] = guess_inputs[i] + [eos] * (input_length - len(guess_inputs[i]))
        candidate_input_ids = torch.IntTensor(guess_inputs).to(input_ids.device)
        with torch.no_grad():
            outputs = self.llm(candidate_input_ids, attention_mask=torch.concat([
                torch.stack(guess_attention_mask),
                torch.ones((len(guess_attention_mask), candidate_length), dtype=attention_mask.dtype, device=attention_mask.device)
            ], dim=1))
        new_logits = outputs.logits[:, -candidate_length - 1 :]
        selected_tokens = new_logits[:, -candidate_length - 1 :, :].argmax(dim=-1)
        candidate_new_tokens = candidate_input_ids[:, -candidate_length:]
        matched = candidate_new_tokens == selected_tokens[:, :-1]
        matched[candidate_new_tokens == eos] = True
        hitted = torch.all(matched, dim=-1).tolist()

        # 3. find those not hitted infills, and invoke normal generation
        result = {}
        generate_attention_mask = []
        generate_input_ids = []
        generate_initial_ids = []
        for i in range(len(hitted)):
            initial_id = guess_intial_id[i]
            if hitted[i]:
                # addressed by speculative decoding
                result[initial_id] = guess_inputs[i]
            else:
                # not hit: should be addressed by normal decoding
                generate_attention_mask.append(attention_mask[initial_id])
                generate_input_ids.append(input_ids[initial_id])
                generate_initial_ids.append(initial_id)
        # the first generation: also should be addressed by normal decoding
        for i in range(len(input_ids)):
            if i not in guess_intial_id:
                generate_initial_ids.append(i)
                generate_attention_mask.append(attention_mask[i])
                generate_input_ids.append(input_ids[i])
        
        # 4. combine the results
        if len(generate_attention_mask) != 0:
            generation_output = self.infill_generate(torch.stack(generate_input_ids), torch.stack(generate_attention_mask), stopping_criteria, bad_word_ids)
            for i in range(len(generate_input_ids)):
                result[generate_initial_ids[i]] = generation_output[i].tolist()
        result_sequence = []
        for i in range(len(input_ids)):
            result_sequence.append(result[i])
        total_length = max([len(x) for x in result_sequence])
        for i in range(len(input_ids)):
            result_sequence[i] = result_sequence[i] + [eos] * (total_length - len(result_sequence[i]))
        return torch.LongTensor(result_sequence).to(input_ids.device)

    def parallel_infilling(self, text_list: List[str], language: str, stop_words: List[str]=None, allow_function=True, bad_words: List[str]=None, speculative_infilling=True) -> List[str]:
        if stop_words is None:
            stop_words = []
        language_spec = LanguageSpec.of(language)
        tokenizer = self.infilling_tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        prefix_list = [x.split("<FILL_ME>")[0] for x in text_list]

        # process <FILL_ME> to incoder format
        if "incoder" in args.checkpoint.lower():
            new_text_list = utils.process_incoder_fill(text_list)
            text_list = new_text_list
        input_ids, attention_masks = utils.encode_parallel(text_list, tokenizer, self.llm.device, tokenizer.eos_token_id)
        bad_word_ids = get_tokens_as_list(self.tokenizer_with_prefix_space, args.bad_words.split(",") + (bad_words if bad_words else []))
        if len(bad_word_ids) == 0:
            bad_word_ids = None
        if input_ids.shape[1] >= 2048:
            return [""] * len(text_list)
        stop_words.append("\n")

        stopping_criteria=StoppingCriteriaList(
            [EndOfFunctionCriteria(
                input_ids.shape[1],
                stop_words,
                tokenizer,
                language_spec,
                allow_function
            )]
        )

        if not allow_function:
            if speculative_infilling:
                outputs_sequence = self.speculative_infill(prefix_list, input_ids, tokenizer.eos_token_id, attention_masks, stopping_criteria, bad_word_ids)

                for prefix, infix in zip(prefix_list, outputs_sequence[:, input_ids.shape[1]:]):
                    infix = infix.detach().cpu().tolist()
                    self.cache_prefix_to_infix[prefix] = infix
            else:
                outputs_sequence = self.infill_generate(input_ids, attention_masks, stopping_criteria, bad_word_ids)    
        else:
            outputs_sequence = self.infill_generate(input_ids, attention_masks, stopping_criteria, bad_word_ids)

        if "incoder" in args.checkpoint.lower():
            sequences = torch.concat([
                torch.tile(self.new_line_token, (len(outputs_sequence), 1)),
                outputs_sequence[:,:]
            ], dim=1)
            output_sequences = [tokenizer.decode(x) for x in sequences]
            output_sequences = utils.process_incoder(output_sequences,allow_function)  
        else:
            sequences = torch.concat([
                torch.tile(self.new_line_token_id, (len(outputs_sequence), 1)),
                outputs_sequence[:, input_ids.shape[1]:]
            ], dim=1)
            output_sequences = [tokenizer.decode(x, skip_special_tokens=True).lstrip('\n') for x in sequences]
        
        output_sequences = [x.replace("<FILL_ME>", "") for x in output_sequences]

        return output_sequences


    def generate_next_line(self, text, language: str, multi_lines=False, stop_words: List[str]=None) -> Tuple[str, bool]:
        if stop_words is None:
            stop_words = []
        language_spec = LanguageSpec.of(language)
        assert "<FILL_ME>" not in text, "Use parallel_infilling instead!"
        tokenizer = self.tokenizer

        inputs = tokenizer.encode(text, return_tensors="pt").to(self.llm.device)
        if inputs.shape[1] > 2048:
            return text, True
        if not multi_lines:
            stop_words.append("\n")

        outputs = self.llm.generate(
            inputs,
            max_new_tokens=256,
            do_sample=False,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=self.bad_words_ids,
            stopping_criteria=StoppingCriteriaList(
                [EndOfFunctionCriteria(
                    inputs.shape[1],
                    stop_words, 
                    tokenizer,
                    language_spec, 
                    allow_function=False
                )]
            ))
        sequences = torch.concat([
            self.new_line_token_id,
            outputs.sequences[0][inputs.shape[1]:]
        ])
        output_sequences = tokenizer.decode(
            sequences, skip_special_tokens=True)[1:]
        stop_words_no_newline = tuple(x.lstrip('\n') for x in stop_words if x.strip() != "")
        is_terminal = (
            sequences[-1].item() == tokenizer.eos_token_id or
            (output_sequences.startswith(stop_words_no_newline)) or
            len(output_sequences.strip()) == ""
        )            
        return output_sequences, is_terminal


def parallel_evaluate_score(env: ModelEnvironment, text_list: List[str]) -> List[List[LineScore]]:
    previous_side = env.tokenizer.padding_side
    env.tokenizer.padding_side = 'right'
    input_tokens = env.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(env.llm.device)
    with torch.no_grad():
        output = env.llm.forward(**input_tokens)
    
    result_list = []
    for i in range(len(text_list)):
        text_scores = process_output(env, input_tokens, output, i)
        result_list.append(text_scores)
    env.tokenizer.padding_side = previous_side
    return result_list

def process_output(env: ModelEnvironment, input_tokens, output, index: int) -> List[LineScore]:
    out_tokens = input_tokens["input_ids"][index].tolist()
    scores: List[LineScore] = []
    score_line = LineScore()

    for i in range(len(out_tokens)):
        cur_score = output.logits[index, i - 1, out_tokens[i]].item()
        decode_token = env.tokenizer.decode(out_tokens[i])
        if decode_token == env.tokenizer.eos_token:
            break            
        score_line.token.append(decode_token)
        score_line.score.append(cur_score)
        if decode_token == '\n':
            scores.append(score_line.clean())
            score_line = LineScore()

    if len(score_line.token) != 0:
        scores.append(score_line.clean())

    return [x.dump() for x in scores]


def evaluate_ppl_score(env: ModelEnvironment, text: str) -> List[LineScore]:
    input_tokens = env.tokenizer.encode(text, return_tensors="pt").to(env.llm.device)
    with torch.no_grad():
        output = env.llm.forward(input_ids=input_tokens)
        out_tokens = input_tokens[0].tolist()
        scores: List[LineScore] = []
        score_line = LineScore()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # Shift so that tokens < n predict n
        shift_logits = output.logits[..., :-1, :].contiguous()
        shift_labels = input_tokens[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, env.llm.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        nl_ppl = -loss_fct(shift_logits, shift_labels)

        for i in range(len(out_tokens)):
            cur_score = nl_ppl[i - 1].item()
            decode_token = env.tokenizer.decode(out_tokens[i])
            score_line.token.append(decode_token)
            score_line.score.append(cur_score)
            if decode_token == '\n':
                scores.append(score_line.clean())
                score_line = LineScore()
        if len(score_line.token) != 0:
            scores.append(score_line.clean())

    return [x.dump() for x in scores]


def evaluate_ppl(env: ModelEnvironment, text: str) -> float:
    input_tokens = env.tokenizer.encode(text, return_tensors="pt").to(env.llm.device)
    with torch.no_grad():
        output = env.llm.forward(input_tokens, labels=input_tokens)
        loss, logits = output[0:2]
        ppl = math.exp(loss.item())
    return ppl


def main(env: ModelEnvironment):

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((args.host, args.port))
    server_socket.listen(1)
    print(f"Listening on {args.host}:{args.port}")
    print(f"Checkpoint: {args.checkpoint}")

    while True:
        conn, address = server_socket.accept()
        print(f"Connection from {address}")

        while True:
            try:
                data = utils.receive(conn)
            except:
                data = None
            if not data:
                break

            request_type = data['request_type']
            print(f"============================================\n<===== Request: {data}")

            if request_type == "generate":
                text = data['text']
                multi_lines = data['multi_lines']
                stop_words = data['stop_words']
                language = data['language']
                next_lines, is_terminal = env.generate_next_line(text, language, multi_lines, stop_words)
                response_data = {
                    "next_lines": next_lines,
                    "is_terminal": is_terminal
                }
            elif request_type == 'parallel_infill':
                text_list = data['text_list']
                stop_words = data['stop_words']
                language = data['language']
                bad_words = data['bad_words']
                allow_function_multi_lines = data['allow_function_multi_lines']
                speculative_infilling = data['speculative_infilling']
                outputs = env.parallel_infilling(text_list, language, stop_words, allow_function_multi_lines, bad_words, speculative_infilling)
                response_data = {
                    "list_of_next_lines": outputs
                }
            elif request_type == "evaluate_score":
                text = data['text']
                if data['parallel']:
                    line_scores = parallel_evaluate_score(env, text)
                else:
                    line_scores = [parallel_evaluate_score(env, [x])[0] for x in text]
                response_data = {
                    "line_scores": line_scores
                }
            elif request_type == "evaluate_ppl":
                text = data['text']
                code_ppl = evaluate_ppl(env, text)
                response_data = {
                    "code_ppl": code_ppl
                }
            elif request_type == "evaluate_ppl_score":
                text = data['text']
                line_scores = evaluate_ppl_score(env, text)
                response_data = {
                    "ppl_score": line_scores
                }
            else:
                response_data = {
                    "Error": f"Error request type: {request_type}"
                }
            print(f"\n=====> Response: {response_data}\n\n\n\n")
            try:
                utils.send(conn, response_data)
            except:
                break

        conn.close()


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    infilling_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    infilling_tokenizer.pad_token = infilling_tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float16,
        device_map={"": args.device},
    )

    tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(args.checkpoint, add_prefix_space=True)

    if args.bad_words.strip() == "":
        bad_word_ids = None
    else:
        bad_word_ids = get_tokens_as_list(tokenizer_with_prefix_space, args.bad_words.split(","))
    env = ModelEnvironment(
        llm=model,
        tokenizer=tokenizer,
        tokenizer_with_prefix_space=tokenizer_with_prefix_space,
        infilling_tokenizer=infilling_tokenizer,
        new_line_token_id=torch.Tensor([tokenizer.convert_tokens_to_ids("<0x0A>")]).int().to(model.device),
        bad_words_ids=bad_word_ids, 
        cache_prefix_to_infix={})

    main(env)
