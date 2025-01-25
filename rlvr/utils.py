# References:
# https://github.com/codelion/optillm/blob/main/scripts/eval_aime_benchmark.py

import torch
import torch.nn as nn

import re
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from tqdm import tqdm
from math_verify import verify


AIME_SYSTEM_PROMPT = """You are solving AIME (American Invitational Mathematics Examination) problems.

Important: Always end your solution with the final answer in one of these two formats:

1. \\[
   \\boxed{X}.
   \\]

2. $n=\\boxed{X}$

where X is your integer answer between 0 and 999."""

# SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
# <answer> answer here </answer>. User: {prompt} Assistant:"""
# SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and you solve it. You first thinks about the reasoning process in the mind and then provide the user with the answer. You must put reasoning process and answer within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
# <answer> answer here </answer>. User: {prompt}"""
SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
# <answer> answer here </answer>."""


def extract_answer(response: str) -> Optional[int]:
    """
    Extract the numerical answer from a math solution response.
    Handles various formats of boxed answers and falls back to last number if needed.
    """
    if not response:
        return None

    # Clean the response
    response = " ".join(response.split())

    patterns = [
        r"\$n=\\boxed{(\d+)}\$",
        r"\\\[\\boxed{(\d+)}\\\]",
        r"\\\[\\boxed{(\d+)}\.\\\]",
        r"\\boxed{(\d+)}",
        r"\$\\boxed{(\d+)}\$",
        r"boxed{(\d+)}",
        r"\\boxed\s*{\s*(\d+)\s*}",
        r"\bboxed\s*{\s*(\d+)\s*}",
        r"final answer is[^\d]*(\d+)",
        r"answer is[^\d]*(\d+)",
        r"answer:[^\d]*(\d+)",
        r"= ?(\d+)$",
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, response, re.IGNORECASE)
        last_match = None
        for match in matches:
            last_match = match

        if last_match:
            try:
                return int(last_match.group(1))
            except (ValueError, IndexError):
                continue

    numbers = re.findall(r"(\d+)", response)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass

    return None


def check_format(s: str) -> int:
    pattern = r"^<think>.+?</think><answer>.+?</answer>$"
    return int(bool(re.fullmatch(pattern, s, re.DOTALL)))


def check_accuracy(completion: str, answer: str):
    completion = extract_answer(completion)
    is_correct = verify(str(completion), answer)
    if is_correct:
        return 1
    else:
        return 0


@torch.no_grad()
def make_n_attempts(problem: str, model: nn.Module, tokenizer, n: int, max_length: int, temperature: float) -> List[Dict]:
    attempts = []
    remaining_attempts = n

    while remaining_attempts > 0:
        # prompt = AIME_SYSTEM_PROMPT + "\n\n" + problem
        prompt = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT + AIME_SYSTEM_PROMPT}, {"role": "user", "content": problem}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(model.device)
        completion_ids = model.generate(**inputs, do_sample=True, num_return_sequences=1, max_length=max_length, temperature=temperature, pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
        predicted_answer = extract_answer(response)
        attempts.append({"attempt_number": len(attempts) + 1, "response": response, "predicted_answer": predicted_answer})
        remaining_attempts -= 1
    return attempts


def evaluate_pass_at_n(attempts: List[Dict], correct_answer: int) -> Tuple[bool, Optional[int]]:
    for attempt in attempts:
        if verify(attempt["predicted_answer"], correct_answer):
            return True, attempt["attempt_number"]
    return False, None


def evaluate_aime(dataset, model: nn.Module, tokenizer, max_length: int = 1024, temperature: float = 0.7, n_attempts: int = 1) -> Tuple[float, List[str]]:

    results = []
    for _, item in enumerate(tqdm(dataset, desc="Evaluating problems")):
        id = int(item["id"])

        problem_text = item["problem"]
        correct_answer = int(item["answer"])

        # Make n attempts for each problem
        attempts = make_n_attempts(problem_text, model, tokenizer, n_attempts, max_length, temperature)
        is_correct, first_correct = evaluate_pass_at_n(attempts, correct_answer)

        result = {"index": id, "problem": problem_text, "attempts": attempts, "correct_answer": correct_answer, "is_correct": is_correct, "first_correct_attempt": first_correct}
        results.append(result)

    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / total if total > 0 else 0
    completions = [r["attempts"][0]["response"] for r in results if r["attempts"]]
    return accuracy, completions
