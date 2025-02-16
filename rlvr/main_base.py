import os
import re
from argparse import ArgumentParser
import torch
from datasets import load_dataset
from grpo import CustomGRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

SYSTEM_PROMPT = """Assistant solves the user's question step by step before answering. Assistant answers using this response structure:
[detailed step by step reasoning process here]
<answer>
[Answer based on the reasoning above enclosed in <answer> tags]
</answer>

User: {prompt}
Assistant:"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--run_name", type=str, default="Qwen")
    parser.add_argument("--num_generations", type=int, default=12)
    return parser.parse_args()


def extract_answer(response: str) -> str:
    return response.split("<answer>")[-1].split("</answer>")[0].strip()


def accuracy_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    return [2.0 if extract_answer(completion) == a else 0.0 for completion, a in zip(completions, answer)]


def strict_format_reward(completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        length = len(completion)
        if re.search("<answer>.*?</answer>\s*$", completion) and completion.index("<answer>") > length * 0.9:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def is_integer_reward(completions: list[str], **kwargs) -> list[float]:
    return [0.1 if extract_answer(c).isdigit() else 0.0 for c in completions]


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
        eval_steps=20,
        eval_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2048,
        gradient_accumulation_steps=32,
        num_generations=args.num_generations,
        max_completion_length=512,
        max_grad_norm=0.01,
        report_to="wandb",
        log_on_each_node=False,
        # push_to_hub=True,
        num_train_epochs=1,
        eval_on_start=True,
        bf16=True,
        beta=0.001,
        ddp_find_unused_parameters=False,
        # use_vllm=True,
        # vllm_gpu_memory_utilization=0.5,
    )
    os.environ["WANDB_PROJECT"] = "rlvr"
    ds_dict = load_dataset("openai/gsm8k", "main")
    train_ds = ds_dict["train"]
    eval_ds = ds_dict["test"]

    train_ds = train_ds.map(lambda x: {"prompt": SYSTEM_PROMPT.format(prompt=x["question"]), "answer": x["answer"].split("####")[1].strip()})
    eval_ds = eval_ds.map(lambda x: {"prompt": SYSTEM_PROMPT.format(prompt=x["question"]), "answer": x["answer"].split("####")[1].strip()})
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = CustomGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[accuracy_reward, strict_format_reward],
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
