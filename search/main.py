import requests
import json
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer
from vllm import LLM, SamplingParams
from mcts import MCTS, Node, Config
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def get_data():
    response = requests.get("https://github.com/GAIR-NLP/AIME-Preview/raw/refs/heads/main/eval/data/aime/test2024.jsonl")
    data = []
    for line in response.iter_lines():
        data.append(json.loads(line))
    return data


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/QwQ-32B")
    parser.add_argument("--prm_model_name", type=str, default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--file_name", type=str, default="mcts")
    return parser.parse_args()


def construct_prompts(data, tokenizer: AutoTokenizer):
    return [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."}, {"role": "user", "content": row["problem"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in data
    ]


def check_accuracy(completion, gold):
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
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    return bool(verify(answer_parsed, gold)), answer_parsed


def evaluate_responses(responses, data, file_name=""):
    scores = []
    parsed_answers = []
    for response, row in zip(responses, data):
        score, parsed = check_accuracy(response, row["answer"])
        scores.append(score)
        parsed_answers.append(parsed)
    # save responses
    with open(f"responses_{file_name}.jsonl", "w") as f:
        for response in responses:
            f.write(json.dumps({"response": response}) + "\n")

    score = sum(scores)
    total = len(scores)
    print(f"Accuracy: {score}/{total} ({score/total:.2%})")


if __name__ == "__main__":
    args = get_args()
    data = get_data()
    model = LLM(args.model_name, tensor_parallel_size=2, gpu_memory_utilization=0.8)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prm = AutoModel.from_pretrained(args.prm_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    prm_tokenizer = AutoTokenizer.from_pretrained(args.prm_model_name)
    prompts = construct_prompts(data, tokenizer)

    responses = []
    for prompt in tqdm(prompts):
        out = model.generate([prompt], SamplingParams(temperature=0.3, max_tokens=32768, seed=0, top_p=0.95))
        generated_response = out[0].outputs[0].text
        generated_responses = [x.outputs[0].text for x in out]
        responses.extend(generated_responses)

    evaluate_responses(responses, data, f"{args.file_name}_baseline")

    with torch.no_grad():
        responses = []
        for prompt in tqdm(prompts):
            mcts = MCTS(
                model=model,
                tokenizer=tokenizer,
                cfg=Config(
                    branching_factor=8,
                    max_length_per_stage=256,
                    max_depth=64,
                    simulate_max_length=32768,
                    num_steps=8,
                ),
                prm=prm,
                prm_tokenizer=prm_tokenizer,
            )
            node = mcts.step(prompt)
            responses.append(node.sequence)

    evaluate_responses(responses, data, f"{args.file_name}_mcts_8")

    with torch.no_grad():
        responses = []
        for prompt in tqdm(prompts):
            mcts = MCTS(
                model=model,
                tokenizer=tokenizer,
                cfg=Config(
                    branching_factor=16,
                    max_length_per_stage=256,
                    max_depth=64,
                    simulate_max_length=32768,
                    num_steps=8,
                ),
                prm=prm,
                prm_tokenizer=prm_tokenizer,
            )
            node = mcts.step(prompt)
            responses.append(node.sequence)

    evaluate_responses(responses, data, f"{args.file_name}_mcts_16")
