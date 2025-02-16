import re
from typing import Optional
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


# DeepSeek R1 Original (Base)
# SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
# <answer> answer here </answer>. User: {prompt} Assistant:"""

BASE_SYSTEM_PROMPT_FEW_SHOT = """The assistant thinks step by step before answering the following question and responds in the following format:
<think>...</think><answer>...</answer>. 
User: What is the largest single-digit prime number?
Assistant: <think>9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.</think><answer>7</answer>.
User: {prompt} 
Assistant:"""

BASE_SYSTEM_PROMPT_ZERO_SHOT = """The assistant thinks step by step before answering the following question and responds in the following format:
<think>...</think><answer>...</answer>. 
User: {prompt} 
Assistant:"""

# Instruct
INSTRUCT_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
# <answer> answer here </answer>."""


def extract_answer(response: str) -> Optional[int]:
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", response)
    if numbers:
        return numbers[-1]
    else:
        return None


def check_format(s: str) -> int:
    pattern = r"^<think>.+?</think><answer>.+?</answer>$"
    return int(bool(re.fullmatch(pattern, s, re.DOTALL)))


def check_accuracy(completion: str, answer: str):
    completion = extract_answer(completion)
    is_correct = verify(str(completion), answer)
    return int(is_correct)


def accuracy_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    completions = [completion[0]["content"] for completion in completions]
    return [check_accuracy(completion, a) for completion, a in zip(completions, answer)]


def format_reward(completions: list[str], **kwargs) -> list[float]:
    completions = [completion[0]["content"] for completion in completions]
    return [check_format(completion) for completion in completions]


def soft_format_reward(completions: list[str], **kwargs) -> list[float]:
    completions = [completion[0]["content"] for completion in completions]
    return [sum(tag in completion for tag in ["<think>", "<answer>", "</think>", "</answer>"]) / 4 for completion in completions]


def soft_format_integer_reward(completions: list[str], **kwargs) -> list[float]:
    completions = [completion[0]["content"].split("<answer>")[-1].split("</answer>")[0] for completion in completions]
    completions = [completion.split("<answer>")[-1].split("</answer>")[0] for completion in completions]
    return [int(x.strip().isnumeric()) for x in completions]


def extract_answer_boxed(response: str) -> Optional[int]:
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

    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", response)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass

    return None


def check_accuracy_hf(completion, answer, **kwargs):
    # gold_parsed = parse(answer, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
    gold_parsed = answer
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            completion,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        reward = float(verify(answer_parsed, gold_parsed))
    else:
        reward = 1.0
        print("Failed to parse gold solution: ", answer)
    return reward
