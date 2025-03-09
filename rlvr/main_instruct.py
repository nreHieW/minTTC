import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig

from grpo import CustomGRPOTrainer
from utils import *


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--num_generations", type=int, default=12)
    return parser.parse_args()


def main():
    args = get_args()
    cfg = GRPOConfig(
        output_dir=f"outputs/{args.run_name}",
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
        gradient_accumulation_steps=16,
        num_generations=args.num_generations,
        max_completion_length=1024,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
        push_to_hub=True,
        num_train_epochs=1,
        eval_on_start=True,
        bf16=True,
        # use_vllm=True,
        # vllm_gpu_memory_utilization=0.5,
    )
    os.environ["WANDB_PROJECT"] = "rlvr"
    ds_dict = load_dataset("nreHieW/Extracted_GSM")
    train_ds = ds_dict["train"]
    eval_ds = load_dataset("openai/gsm8k", "main")["test"]

    train_ds = train_ds.map(lambda x: {"prompt": [{"role": "system", "content": INSTRUCT_SYSTEM_PROMPT}, {"role": "user", "content": x["question"]}], "answer": x["answer"]})
    eval_ds = eval_ds.map(lambda x: {"prompt": [{"role": "system", "content": INSTRUCT_SYSTEM_PROMPT}, {"role": "user", "content": x["question"]}], "answer": x["answer"].rsplit("####", 1)[1].strip()})

    # train_ds = train_ds.map(lambda x: {"prompt": BASE_SYSTEM_PROMPT_FEW_SHOT.format(prompt=x["question"]), "answer": x["answer"]})
    # eval_ds = eval_ds.map(lambda x: {"prompt": BASE_SYSTEM_PROMPT_ZERO_SHOT.format(prompt=x["question"]), "answer": x["answer"].rsplit("####", 1)[1].strip()})

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
