# References:
# https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
# https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.utils.data
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from dataclasses import dataclass, field
from transformers import TrainingArguments
from trl.models import create_reference_model, unwrap_model_for_generation
from utils import *
import json


@dataclass
class GRPOConfig(TrainingArguments):
    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={"help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` " "argument of the `GRPOTrainer` is provided as a string."},
    )

    # Parameters that control the data preprocessing
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."},
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={"help": "Number of generations to sample."},
    )
    temperature: Optional[float] = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of " "`transformers.TrainingArguments`."},
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "KL coefficient."},
    )
    format_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for the format reward."},
    )


class GRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel, nn.Module] = None,
        args: GRPOConfig = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError("Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing " f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}.")
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                raise ValueError("You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. " "This argument can only be used when the `model` argument is a string.")

        self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Data loading and preprocessing
        if data_collator is None:

            def data_collator(features):  # No data collation is needed in GRPO
                return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=processing_class.pad_token_id,
        )
        self.beta = args.beta
        self.format_weight = args.format_weight

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"kl": [], "reward": [], "reward_std": [], "loss": [], "accuracy_rewards": [], "format_rewards": []}

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # prompts_text = [x["question"] for x in inputs]
        prompts_text = [SYSTEM_PROMPT.format(prompt=x["question"]) for x in inputs]
        # prompts_text = [
        #     self.processing_class.apply_chat_template([{"role": "user", "content": SYSTEM_PROMPT.format(prompt=x["question"])}], tokenize=False, add_generation_prompt=True) for x in inputs
        # ]

        answers = [x["answer"] for x in inputs]
        prompt_inputs = self.processing_class(prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Get the per-token log probabilities for the completions for the model and the reference model
        def get_per_token_logps(model, input_ids):
            logits = model(input_ids).logits  # (B, L, V)
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        per_token_logps = get_per_token_logps(model, prompt_completion_ids)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Compute the rewards
        format_rewards = [check_format(completion) for completion in completions]
        format_rewards = torch.tensor(format_rewards, device=device, dtype=per_token_logps.dtype)  # Shape (B*G,)
        answers = [answer for answer in answers for _ in range(self.num_generations)]
        accuracy_rewards = [check_accuracy(completion, answer) for completion, answer in zip(completions, answers)]
        accuracy_rewards = torch.tensor(accuracy_rewards, device=device, dtype=per_token_logps.dtype)

        rewards = self.format_weight * format_rewards + accuracy_rewards
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        advantages = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)

        ratio = torch.exp(per_token_logps - per_token_logps.detach())
        epsilon = 0.2  # PPO clip parameter
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

        # Compute policy loss with clipping
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        per_token_loss = policy_loss + self.beta * per_token_kl
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["accuracy_rewards"].append(self.accelerator.gather_for_metrics(accuracy_rewards).mean().item())
        self._metrics["format_rewards"].append(self.accelerator.gather_for_metrics(format_rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        self._metrics["loss"].append(self.accelerator.gather_for_metrics(loss).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        super().log(logs)
        self._metrics = {key: [] for key in self._metrics}
        if self.completions is not None:
            fname = f"{self.args.output_dir}/completions_{self.state.global_step}.json"
            with open(fname, "w") as f:
                json.dump(self.completions, f)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        accuracy, completions = evaluate_aime(dataset=eval_dataset, model=self.model, tokenizer=self.processing_class, n_attempts=1)
        self.completions = completions
        return {"aime_accuracy": accuracy}
