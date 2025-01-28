# References:
# https://github.com/codelion/optillm/blob/main/scripts/eval_aime_benchmark.py

import torch
import torch.nn as nn

import re
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from math_verify import verify

# DeepSeek R1 Original (Base)
# SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
# <answer> answer here </answer>. User: {prompt} Assistant:"""

# Instruct
SYSTEM_PROMPT = """You are a helpful assistant. Think step by step before answer. Enclose your reasoning process and answer within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer>only the answer here</answer>."""


def extract_answer(response: str) -> Optional[int]:
    response = response.split("<answer>")[-1].replace("</answer>", "").strip()
    if not response:
        return None
    response = " ".join(response.split())

    patterns = [
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
    return int(is_correct)
