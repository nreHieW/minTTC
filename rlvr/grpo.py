from typing import Optional, Union, Dict, List
import random
import torch
import torch.distributed as dist
from datetime import datetime
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from utils import extract_answer, check_accuracy
from transformers import AutoModelForCausalLM
from muon import Muon
import os
import math


class MultiOptimizer:
    def __init__(self, optimizers):
        self.param_groups = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)

        self.optimizers = optimizers

    def step(self, closure=None):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def set_lr(self, multiplier):
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                if "initial_lr" not in group:
                    group["initial_lr"] = group["lr"]
                group["lr"] = group["initial_lr"] * multiplier


class Scheduler:
    def __init__(self, args: GRPOConfig, optimizer: MultiOptimizer, num_training_steps: int):
        self.args = args
        self.iter = 0
        self.optimizer = optimizer
        self.num_training_steps = num_training_steps

    def step(self, closure=None):
        self.iter += 1
        multiplier = get_lr_multiplier(self.iter, self.num_training_steps, self.args.warmup_ratio)
        self.optimizer.set_lr(multiplier)

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]["lr"]]


def get_lr_multiplier(step: int, max_steps: int, warmup_ratio: float) -> float:
    warmup_steps = int(max_steps * warmup_ratio)

    if step < warmup_steps:
        # During warmup, return the proportion of steps completed
        return step / warmup_steps

    # After warmup, apply cosine decay from 1.0 to 0.0
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def configure_optimizers(model: AutoModelForCausalLM, cfg: GRPOConfig) -> torch.optim.Optimizer:
    # https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py
    params_dict = {}

    # First, categorize all parameters
    for name, param in model.named_parameters():
        if "lm_head" in name:
            params_dict[param] = "head"
        elif "embed" in name:
            params_dict[param] = "embed"
        elif param.ndim >= 2:
            params_dict[param] = "hidden_matrix"
        else:
            params_dict[param] = "scalar"

    hidden_matrix_params = [p for p, group in params_dict.items() if group == "hidden_matrix"]
    embed_params = [p for p, group in params_dict.items() if group == "embed"]
    scalar_params = [p for p, group in params_dict.items() if group == "scalar"]
    head_params = [p for p, group in params_dict.items() if group == "head"]

    adam_params = [dict(params=head_params, lr=cfg.learning_rate), dict(params=embed_params, lr=cfg.learning_rate), dict(params=scalar_params, lr=cfg.learning_rate)]

    optimizer1 = torch.optim.Adam(adam_params, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    optimizer2 = Muon(hidden_matrix_params, lr=0.0005, momentum=0.95, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

    return MultiOptimizer([optimizer1, optimizer2])
    # return MultiOptimizer([optimizer])
    # return optimizer


class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completions = []

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = configure_optimizers(self.model, self.args)
        self.lr_scheduler = Scheduler(self.args, self.optimizer, num_training_steps)

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        batch_size: int = 2048,
    ) -> Dict[str, float]:
        if dist.get_rank() != 0:
            return {}
        batch_size = self.args.per_device_eval_batch_size or batch_size
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        total_correct = 0
        total_examples = len(eval_dataset)
        model = self._wrap_model(self.model, training=False, dataloader=None)

        model = (
            self.accelerator.prepare(model)
            if self.is_deepspeed_enabled or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
            else self.accelerator.prepare_model(model, evaluation_mode=True)
        )

        model.eval()
        all_completions = []
        all_answers = []
        for i in range(0, total_examples, batch_size):
            batch = eval_dataset[i : min(i + batch_size, total_examples)]

            prompt_texts = [maybe_apply_chat_template({"prompt": row}, self.processing_class)["prompt"] for row in batch["prompt"]]
            prompt_inputs = self.processing_class(prompt_texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            prompt_inputs = {k: v.to(self.accelerator.device) for k, v in prompt_inputs.items()}
            prompt_length = prompt_inputs["input_ids"].size(1)
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                completion_ids = unwrapped_model.generate(
                    **prompt_inputs, num_return_sequences=1, max_new_tokens=self.max_completion_length, do_sample=False, pad_token_id=self.processing_class.pad_token_id, temperature=0.0
                )
                completion_ids = completion_ids[:, prompt_length:]
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            all_completions.extend(completions)
            all_answers.extend(batch["answer"])

            for completion, ans in zip(completions, batch["answer"]):
                # extracted_answer = extract_answer(completion)
                # is_correct = check_accuracy(str(extracted_answer), ans)
                extracted_answer = completion.split("<answer>")[-1].split("</answer>")[0].strip()
                total_correct += int(str(extracted_answer) == str(ans))
                # total_correct += int(verify(str(extracted_answer), ans))
        accuracy = total_correct / total_examples
        self._metrics[f"{metric_key_prefix}_gsm8k_accuracy"] = [accuracy]
        all_completions = [f"{completion}\n\nAnswer:{answer}" for completion, answer in zip(all_completions, all_answers)]
        sampled_completions = random.sample(all_completions, 10)
        self.completions = sampled_completions
        return {f"{metric_key_prefix}_gsm8k_accuracy": accuracy}

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        start_time = datetime.now().strftime("%Y%m%d%H%M%S")
        fname = f"{self.args.output_dir}/metrics_{start_time}.txt"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if self.completions:
            with open(fname, "w") as f:
                f.write(("\n\n" + "=" * 50 + "\n\n").join(self.completions))
            self.completions = []
        super().log(logs)
        self._metrics.clear()
