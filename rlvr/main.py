import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig

from grpo import CustomGRPOTrainer
from utils import SYSTEM_PROMPT, check_accuracy, check_format


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
    return [int(x.strip().isnumeric()) for x in completions]


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/Llama1B")
    parser.add_argument("--run_name", type=str, default="Llama1B")
    parser.add_argument("--num_generations", type=int, default=4)
    return parser.parse_args()


def main():
    args = get_args()
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        learning_rate=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_steps=20,
        eval_strategy="steps",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=6,
        num_generations=8,
        max_completion_length=512,
        max_grad_norm=0.01,
        report_to="none",
        log_on_each_node=False,
        push_to_hub=True,
        num_train_epochs=1,
        eval_on_start=True,
        bf16=True,
    )
    os.environ["WANDB_PROJECT"] = "rlvr"
    ds_dict = load_dataset("nreHieW/Extracted_GSM")
    train_ds = ds_dict["train"]
    train_ds = train_ds.map(lambda x: {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": x["question"]}], "answer": x["answer"]})

    eval_ds = load_dataset("openai/gsm8k", "main")["test"]
    eval_ds = eval_ds.map(lambda x: {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": x["question"]}], "answer": x["answer"].rsplit("####", 1)[1].strip()})

    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = CustomGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        # reward_funcs=[accuracy_reward, format_reward, soft_format_reward, soft_format_integer_reward],
        reward_funcs=[accuracy_reward, format_reward],
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
