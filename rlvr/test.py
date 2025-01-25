import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from copy import deepcopy
from utils import *


@dataclass
class GRPOConfig:
    group_size: int = 4
    max_seq_length: int = 1024
    format_weight: float = 1.0
    clip_param: float = 0.2
    beta: float = 0.01


class GRPO:
    def __init__(self, policy_model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: GRPOConfig):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.policy_model = policy_model.to(self.device)
        self.ref_model = deepcopy(policy_model).to(self.device)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = config

    def _get_logprobs(self, model: AutoModelForCausalLM, input_ids: torch.Tensor):
        logits = model(input_ids).logits
        logits = logits[:, :-1]
        input_ids = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    def _compute_kl(self, old_logprobs: torch.Tensor, new_logprobs: torch.Tensor, mask: torch.Tensor):
        ratio = old_logprobs - new_logprobs
        ratio = ratio * mask
        return ratio.exp() - ratio - 1

    def compute_loss(self, prompts: List[str], answers: List[str]):
        bs = len(prompts)
        prompts = [SYSTEM_PROMPT.format(prompt=prompt) for prompt in prompts]
        prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(self.device)

        with torch.no_grad():
            old_completion_ids_GB_L = self.policy_model.generate(
                input_ids=prompt_inputs["input_ids"],
                attention_mask=prompt_inputs["attention_mask"],
                max_length=self.config.max_seq_length,
                num_return_sequences=self.config.group_size,
            )

            old_logprobs_GB_L = self._get_logprobs(self.ref_model, old_completion_ids_GB_L)

        mask = (old_completion_ids_GB_L != self.tokenizer.pad_token_id).float()[:, 1:]
        policy_logprobs_GB_L = self._get_logprobs(self.policy_model, old_completion_ids_GB_L)

        old_completion_ids_GB_L = old_completion_ids_GB_L[:, old_completion_ids_GB_L.shape[1] :]

        completions = self.tokenizer.batch_decode(old_completion_ids_GB_L, skip_special_tokens=True)

        # Compute the rewards
        format_rewards = [check_format(completion) for completion in completions]
        format_rewards_GB = torch.tensor(format_rewards, device=self.device, dtype=self.policy_model.dtype)
        answers = [answer for answer in answers for _ in range(self.config.group_size)]
        accuracy_rewards = [check_accuracy(completion, answer) for completion, answer in zip(completions, answers)]
        accuracy_rewards_GB = torch.tensor(accuracy_rewards, device=self.device, dtype=self.policy_model.dtype)

        rewards_GB = self.config.format_weight * format_rewards_GB + accuracy_rewards_GB
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards_GB.view(-1, self.config.group_size).mean(dim=1)
        std_grouped_rewards = rewards_GB.view(-1, self.config.group_size).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.config.group_size, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.config.group_size, dim=0)
        advantages_GB = (rewards_GB - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages_GB_1 = advantages_GB.unsqueeze(1)

        kl_GB_L = self._compute_kl(old_logprobs_GB_L, policy_logprobs_GB_L, mask)
        ratio = (policy_logprobs_GB_L - old_logprobs_GB_L).exp()
        left_term = -(ratio * advantages_GB_1)
        right_term = -torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param) * advantages_GB_1
        policy_loss = torch.max(left_term, right_term)
        policy_loss = policy_loss * mask

        loss = (policy_loss + self.config.beta * kl_GB_L).mean()
        return loss
