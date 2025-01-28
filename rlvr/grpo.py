from typing import Optional, Union, Dict, List

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from utils import extract_answer, check_accuracy


class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        batch_size: int = 2048,
    ) -> Dict[str, float]:
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
        for i in range(0, total_examples, batch_size):
            batch = eval_dataset[i : min(i + batch_size, total_examples)]

            prompt_texts = [maybe_apply_chat_template({"prompt": row}, self.processing_class)["prompt"] for row in batch["prompt"]]
            prompt_inputs = self.processing_class(prompt_texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            prompt_inputs = {k: v.to(self.accelerator.device) for k, v in prompt_inputs.items()}
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                completion_ids = unwrapped_model.generate(
                    **prompt_inputs, num_return_sequences=1, max_length=self.max_completion_length, do_sample=False, pad_token_id=self.processing_class.pad_token_id
                )
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            for completion, ans in zip(completions, batch["answer"]):
                extracted_answer = extract_answer(completion)
                is_correct = check_accuracy(str(extracted_answer), ans)
                total_correct += is_correct
        accuracy = total_correct / total_examples
        self._metrics[f"{metric_key_prefix}_gsm8k_accuracy"] = [accuracy]
        return {f"{metric_key_prefix}_gsm8k_accuracy": accuracy}
