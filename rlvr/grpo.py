from typing import Optional, Union, Dict, List
import random
import torch
import torch.distributed as dist
from datetime import datetime
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.models import unwrap_model_for_generation


class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completions = []

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
