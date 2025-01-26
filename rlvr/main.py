import torch
import torch.nn
from torch.utils.data import DataLoader
import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from grpo import GRPOConfig, GRPO
from utils import evaluate_aime
import wandb
import datetime


def main():
    config = GRPOConfig(
        num_generations=4,
        max_completion_length=1024,
        temperature=0.7,
        epochs_per_step=1,
        beta=0.01,
        format_weight=1.0,
        ppo_clip_param=0.2,
        grad_max_norm=1.0,
        run_name="Llama1B",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        batch_size=1,
        log_interval=2,
        eval_interval=5,
        learning_rate=1e-6,
        num_epochs=3,
    )

    output_dir = config.run_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ds_dict = load_dataset("nreHieW/Extracted_NuminaMath_Code_Contests")
    # train_ds = ds_dict["train"].filter(lambda example: example["type"] == "math")
    ds_dict = load_dataset("nreHieW/Extracted_GSM")
    # ds_dict = load_dataset("nreHieW/Extracted_MATH")
    train_ds = ds_dict["train"]

    dataset_aime = load_dataset("AI-MO/aimo-validation-aime")
    aime = dataset_aime["train"].filter(lambda example: "2024" in example["url"])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name).to("cuda", dtype=torch.bfloat16)
    grpo = GRPO(model, tokenizer, config)

    optimizer = torch.optim.AdamW(grpo.model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        for i, batch in enumerate(train_loader):
            grpo.step(prompts=batch["question"], answers=batch["answer"], optimizer=optimizer, log=i % config.log_interval == 0, step=i)

            if (i % config.eval_interval == 0) and (i > 0):
                grpo.evaluate(aime, epoch=epoch, step=i)

    model.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
