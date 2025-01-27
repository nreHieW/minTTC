from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from utils import SYSTEM_PROMPT, check_accuracy, check_format


def accuracy_reward(completions, answer: list[str], **kwargs) -> list[float]:
    completions = [completion[0]["content"] for completion in completions]
    return [check_accuracy(completion, a) for completion, a in zip(completions, answer)]


def format_reward(completions, **kwargs) -> list[float]:
    completions = [completion[0]["content"] for completion in completions]
    return [check_format(completion) for completion in completions]


def main():
    ds_dict = load_dataset("nreHieW/Extracted_GSM")
    train_ds = ds_dict["train"]
    train_ds = train_ds.map(lambda x: {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": x["question"]}], "answer": x["answer"]})

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=[accuracy_reward, format_reward],
        args=GRPOConfig(
            output_dir="outputs/Llama1B",
            run_name="Llama1B",
            learning_rate=1e-6,
            logging_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=6,
            num_generations=4,
            max_completion_length=1024,
            report_to="wandb",
            log_on_each_node=False,
            push_to_hub=True,
        ),
        train_dataset=train_ds,
    )
    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
