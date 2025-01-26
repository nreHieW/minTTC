# References:
# https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
# https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py


from calendar import c
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from dataclasses import dataclass
from utils import *
import json
from copy import deepcopy
import wandb
import datetime


@dataclass
class GRPOConfig:
    num_generations: int = 4
    max_completion_length: int = 1024
    temperature: float = 0.7
    epochs_per_step: int = 4
    beta: float = 0.01
    format_weight: float = 1.0
    ppo_clip_param: float = 0.2
    grad_max_norm: float = 1.0
    run_name: str = "output"

    model_name: str = ""
    batch_size: int = 1
    log_interval: int = 1
    eval_interval: int = 1
    learning_rate: float = 5e-5
    num_epochs: int = 3


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


class GRPO:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        args: GRPOConfig,
    ):
        self.model = model
        self.ref_model = deepcopy(model).to(model.device)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.args = args
        self.epoch_per_step = args.epochs_per_step
        self.beta = args.beta
        self.format_weight = args.format_weight
        self.ppo_clip_param = args.ppo_clip_param

        self.metrics = {"kl": [], "reward": [], "loss": [], "pg_loss": [], "accuracy_rewards": [], "format_rewards": [], "grad_norm": [], "avg_seq_len": []}
        self.device = self.model.device
        curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run = wandb.init(
            project="rlvr",
            name=f"{args.run_name}_{curr_time}",
            config=args.__dict__,
        )

    def _compute_advantages(self, completions: List[str], answers: List[str]):
        # Compute the rewards
        format_rewards = [check_format(completion) for completion in completions]
        format_rewards = torch.tensor(format_rewards)
        answers = [answer for answer in answers for _ in range(self.num_generations)]
        accuracy_rewards = [check_accuracy(completion, answer) for completion, answer in zip(completions, answers)]
        accuracy_rewards = torch.tensor(accuracy_rewards)

        rewards = self.format_weight * format_rewards + accuracy_rewards
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        return advantages, rewards, accuracy_rewards, format_rewards

    def _get_mask(self, completion_ids):
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.tokenizer.eos_token_id
        device = completion_ids.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        return completion_mask

    def step(self, prompts: List[str], answers: List[str], optimizer: torch.optim.Optimizer, log: bool = False, step: int = None) -> None:
        # prompts_text = [SYSTEM_PROMPT.format(prompt=x) for x in prompts]
        prompts_text = [self.tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
        prompt_inputs = self.tokenizer(prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(self.model.device)

        with torch.no_grad():
            prompt_completion_ids = self.model.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_length = prompt_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        per_token_logps = get_per_token_logps(self.model, prompt_completion_ids)  # only thing with grad
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        old_per_token_logps = per_token_logps.detach()

        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        advantages, rewards, accuracy_rewards, format_rewards = self._compute_advantages(completions, answers)
        advantages = advantages.unsqueeze(1).to(self.model.device, dtype=self.model.dtype)
        completion_mask = self._get_mask(completion_ids)

        for idx in range(self.epoch_per_step):
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

            ratio = torch.exp(per_token_logps - old_per_token_logps)
            clipped_ratio = torch.clamp(ratio, 1 - self.ppo_clip_param, 1 + self.ppo_clip_param)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

            per_token_loss = policy_loss + self.beta * per_token_kl
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_max_norm)
            optimizer.step()

            if idx < self.epoch_per_step - 1:
                per_token_logps = get_per_token_logps(self.model, prompt_completion_ids)
                per_token_logps = per_token_logps[:, prompt_length - 1 :]

            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self.metrics["kl"].append(mean_kl.item())
            self.metrics["loss"].append(loss.item())
            self.metrics["pg_loss"].append(policy_loss.mean().item())
            self.metrics["grad_norm"].append(grad_norm.item())

        self.metrics["reward"].append(rewards.mean().item())
        self.metrics["accuracy_rewards"].append(accuracy_rewards.float().mean().item())
        self.metrics["format_rewards"].append(format_rewards.float().mean().item())

        completion_lengths = completion_mask.sum(dim=1).float()
        avg_seq_len = completion_lengths.mean().item()
        self.metrics["avg_seq_len"].append(avg_seq_len)

        if log:
            self.log(completions, step)

    def log(self, completions: List[str], step: int):
        fname = f"{self.args.run_name}/completions_{step}.json"
        with open(fname, "w") as f:
            json.dump(completions, f)

        wandb.save(fname)
        metrics = {key: sum(val) / len(val) for key, val in self.metrics.items()}
        print(f"Step {step}: {metrics}")
        wandb.log(metrics)

        self.metrics = {key: [] for key in self.metrics}

    def evaluate(self, aime: torch.utils.data.Dataset, epoch: int, step: int):
        self.model.eval()
        accuracy, completions = evaluate_aime(aime, self.model, self.tokenizer, max_length=self.args.max_completion_length, temperature=self.args.temperature, n_attempts=1)
        print(f"Epoch {epoch}, Batch {step}, AIME accuracy: {accuracy})")
        wandb.log({"AIME_accuracy": accuracy})

        fname = f"{self.args.run_name}/AIME_completions_{epoch}_{step}.json"
        with open(fname, "w") as f:
            json.dump(completions, f)

        wandb.save(fname)
        self.model.train()
