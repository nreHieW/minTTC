# References: https://github.com/huggingface/open-r1/tree/main
import os
import re
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, List, Optional, Union

import bitsandbytes as bnb
import torch
from datasets import Dataset, load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from peft import LoraConfig
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import get_parameter_names
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from liger_kernel.transformers import AutoLigerKernelForCausalLM

SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--run_name", type=str, default="Distill-RLVR")
    parser.add_argument("--num_generations", type=int, default=8)
    return parser.parse_args()


def check_accuracy(completion, answer):
    gold_parsed = parse(answer, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
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
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        reward = float(verify(answer_parsed, gold_parsed))
    else:
        # If the gold solution is not parseable, we reward 1 to skip this example
        reward = 1.0
        print("Failed to parse gold solution: ", answer)
    return reward


def accuracy_reward(completions, answer, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    return [check_accuracy(content, sol) for content, sol in zip(contents, answer)]


def format_reward(completions, **kwargs):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def get_optimizer(model, training_args: GRPOConfig):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = training_args.learning_rate
    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )
    return adam_bnb_optim


def main():
    args = get_args()
    cfg = GRPOConfig(
        output_dir=f"outputs/{args.run_name}",
        run_name=args.run_name,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0,
        warmup_ratio=0.07,
        lr_scheduler_type="cosine",
        logging_steps=1,
        # eval_steps=20,
        # eval_strategy="steps",
        per_device_train_batch_size=args.num_generations // 2,
        per_device_eval_batch_size=256,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=4096,
        max_grad_norm=0.01,
        report_to="none",
        log_on_each_node=False,
        # push_to_hub=True,
        num_train_epochs=1,
        eval_on_start=False,
        bf16=True,
        beta=0.001,
    )
    os.environ["WANDB_PROJECT"] = "rlvr"
    train_ds = load_dataset("AI-MO/NuminaMath-TIR")["train"]
    eval_ds = load_dataset("HuggingFaceH4/aime_2024")["train"]

    train_ds = train_ds.map(lambda x: {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": x["problem"]}], "answer": x["solution"]}).select_columns(
        ["prompt", "answer"]
    )
    eval_ds = eval_ds.map(lambda x: {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": x["problem"]}], "answer": x["solution"]}).select_columns(["prompt", "answer"])

    model_name = args.model_name
    model = AutoLigerKernelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    optimizer = get_optimizer(model, cfg)

    lora_rank = 8
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[accuracy_reward, format_reward],
        args=cfg,
        train_dataset=train_ds,
        optimizers=(optimizer, None),
        peft_config=peft_config,
        # eval_dataset=eval_ds,
    )
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
