from datasets import load_dataset
from grpo import GRPOConfig, GRPOTrainer

if __name__ == "__main__":
    # ds_dict = load_dataset("nreHieW/Extracted_NuminaMath_Code_Contests")
    # # math for now
    # train_ds = ds_dict["train"].filter(lambda example: example["type"] == "math")

    ds_dict = load_dataset("nreHieW/Extracted_MATH")
    train_ds = ds_dict["train"]
    dataset_aime = load_dataset("AI-MO/aimo-validation-aime")
    aime = dataset_aime["train"].filter(lambda example: "2024" in example["url"])

    training_args = GRPOConfig(
        output_dir="Qwen2-0.5B-GRPO",
        learning_rate=1e-5,
        logging_steps=10,
        gradient_accumulation_steps=16,
        max_completion_length=1048,
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        report_to="none",
        num_generations=2,
        do_eval=True,
        bf16=True,
    )
    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        args=training_args,
        train_dataset=train_ds,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
