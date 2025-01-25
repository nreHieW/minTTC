import torch
import torch.nn
from torch.utils.data import DataLoader
import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from grpo import GRPOConfig, GRPO
from utils import evaluate_aime


def main():
    config = GRPOConfig(
        num_generations=2,
        max_completion_length=1024,
        temperature=0.7,
        epochs_per_step=1,
        beta=0.01,
        format_weight=1.0,
        ppo_clip_param=0.2,
        output_dir="output",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        batch_size=1,
        log_interval=2,
        eval_interval=5,
        learning_rate=5e-5,
        num_epochs=3,
    )

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # ds_dict = load_dataset("nreHieW/Extracted_NuminaMath_Code_Contests")
    # train_ds = ds_dict["train"].filter(lambda example: example["type"] == "math")
    ds_dict = load_dataset("nreHieW/Extracted_GSM")
    # ds_dict = load_dataset("nreHieW/Extracted_MATH")
    train_ds = ds_dict["train"]

    dataset_aime = load_dataset("AI-MO/aimo-validation-aime")
    aime = dataset_aime["train"].filter(lambda example: "2024" in example["url"])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name).to("cuda", dtype=torch.float16)
    grpo = GRPO(model, tokenizer, config)

    optimizer = torch.optim.AdamW(grpo.model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        for i, batch in enumerate(train_loader):

            grpo.step(prompts=batch["question"], answers=batch["answer"], optimizer=optimizer, log=i % config.log_interval == 0, step=i)

            if i % config.eval_interval == 0:
                accuracy, completions = evaluate_aime(aime, grpo.model, tokenizer, max_length=config.max_completion_length, temperature=config.temperature, n_attempts=1)
                print(f"Epoch {epoch}, Batch {i}, AIME accuracy: {accuracy})")

                fname = f"{config.output_dir}/AIME_completions_{epoch}_{i}.json"
                with open(fname, "w") as f:
                    json.dump(completions, f)

    model.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
