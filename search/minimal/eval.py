import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict
import random
import math

from lm_eval import simple_evaluate
from lm_eval.evaluator_utils import TaskOutput
from lm_eval.api.model import LM


@dataclass
class Config:
    branching_factor: int = 5
    max_length_per_stage: int = 50
    simulate_max_length: int = 100
    num_steps: int = 5
    max_depth: int = 8


@dataclass
class Node:
    # List of tensors, each of shape (1, stage_len)
    token_stages: List[torch.Tensor]
    depth: int
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0

    @property
    def tokens(self) -> torch.Tensor:
        """Concatenate all stages into a single tensor."""
        if not self.token_stages:
            return torch.empty((1, 0), dtype=torch.long)
        return torch.cat(self.token_stages, dim=1)

    def is_terminal(self) -> bool:
        if self.depth == 0:
            return False
        if not self.token_stages:
            return False
        return (self.token_stages[-1][0, -1] == 128009).item()

    @property
    def score(self) -> float:
        c = math.sqrt(2)
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.total_value / self.visit_count
        exploration = c * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        return exploitation + exploration

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


def score_node(node: Node, prm: AutoModel, prm_tokenizer: AutoTokenizer, tokenizer: AutoTokenizer) -> float:
    # first 2 stages are the prompt and the first stage
    steps = [tokenizer.decode(node.token_stages[0][0]) + tokenizer.decode(node.token_stages[1][0])]

    # Add remaining stages
    for stage_tokens in node.token_stages[2:]:
        decoded = tokenizer.decode(stage_tokens[0])
        steps.append(decoded)

    conversation_str = "<extra_0>".join([x.strip() for x in steps])
    conversation_str += "<extra_0>"
    input_ids = prm_tokenizer.encode(
        conversation_str,
        return_tensors="pt",
    ).to(prm.device)

    outputs = prm(input_ids=input_ids)
    step_sep_id = prm_tokenizer.encode("<extra_0>")[0]
    token_masks = input_ids == step_sep_id

    probabilities = F.softmax(outputs[0], dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    assert probabilities.size(0) == 1
    sample = probabilities[0]
    positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
    scores = positive_probs.cpu().tolist()
    # get last score per section 3.2.4 of https://arxiv.org/pdf/2501.07301?
    # return scores[-1]
    # get min score
    return min(scores)


class MCTS:
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer, cfg: Config, prm: AutoModel, prm_tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.prm = prm
        self.prm_tokenizer = prm_tokenizer
        self.cfg = cfg
        self.stop_tokens = set()
        for p in [".", ",", "\n", "\n\n", ":\n\n", ". ", ", "]:
            self.stop_tokens.add(tokenizer.encode(p, add_special_tokens=False)[0])
        self.stop_tokens.add(tokenizer.eos_token_id)
        self.leaf_nodes = []
        self.root = None

    def split_stage(self, stage: torch.Tensor) -> List[torch.Tensor]:
        stages = []
        old_idx = 0
        for i, token in enumerate(stage[0]):
            if token.item() in self.stop_tokens:
                stages.append(stage[:, old_idx : i + 1])
                old_idx = i + 1

        if old_idx < stage.size(1):
            stages.append(stage[:, old_idx:])
        return stages

    def _expand(self, node: Node) -> None:
        if node.is_terminal() or node.depth >= self.cfg.max_depth:
            return

        return_seq_BL = self.model.generate(
            node.tokens,
            attention_mask=torch.ones_like(node.tokens),
            max_new_tokens=self.cfg.max_length_per_stage,
            do_sample=True,
            num_return_sequences=self.cfg.branching_factor,
        )
        for seq_L in return_seq_BL:
            new_tokens = seq_L[node.tokens.size(1) :].unsqueeze(0)
            new_token_stages = node.token_stages.copy()
            new_token_stages.extend(self.split_stage(new_tokens))
            new_node = Node(token_stages=new_token_stages, parent=node, depth=node.depth + 1)
            node.children.append(new_node)

    def _select(self, node: Node) -> Node:
        while node.children:
            if not all(child.visit_count > 0 for child in node.children):
                # If there are unvisited children, visit them first
                unvisited = [child for child in node.children if child.visit_count == 0]
                return random.choice(unvisited)
                # scored_unvisited = [(child, score_node(child, self.prm, self.prm_tokenizer, self.tokenizer))
                #                   for child in unvisited]
                # return max(scored_unvisited, key=lambda x: x[1])[0]

            node = max(node.children, key=lambda child: child.score)

            if node.is_terminal() or node.depth >= self.cfg.max_depth:
                return node

        return node

    def _simulate(self, node: Node) -> float:
        if node.is_terminal() or node.depth >= self.cfg.max_depth:
            return node.value
        return_seq_1L = self.model.generate(
            node.tokens,
            attention_mask=torch.ones_like(node.tokens),
            max_new_tokens=self.cfg.simulate_max_length,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        if return_seq_1L[0, -1] == self.tokenizer.eos_token_id:
            return_seq_1L = return_seq_1L[:, :-1]

        new_tokens = return_seq_1L[:, node.tokens.size(1) :]

        new_token_stages = node.token_stages.copy()
        new_token_stages.extend(self.split_stage(new_tokens))

        new_leaf_node = Node(
            token_stages=new_token_stages,
            parent=node,
            depth=node.depth + 1,
        )

        score = score_node(new_leaf_node, self.prm, self.prm_tokenizer, self.tokenizer)
        new_leaf_node.total_value = score
        new_leaf_node.visit_count = 1
        self.leaf_nodes.append(new_leaf_node)
        return score

    def _backpropagate(self, node: Node, reward: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.total_value += reward
            node = node.parent

    def step(self, initial_tokens: torch.Tensor) -> Node:
        root = Node(token_stages=[initial_tokens.to(self.model.device)], depth=0)
        self.root = root

        for _ in range(self.cfg.num_steps):
            leaf = self._select(root)
            self._expand(leaf)
            reward = self._simulate(leaf)
            self._backpropagate(leaf, reward)
        return max(self.leaf_nodes, key=lambda node: node.value)


device = "cuda:1"


class CustomLLaMAModel(LM):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        self.prm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-PRM-7B", trust_remote_code=True)
        self.prm = AutoModel.from_pretrained(
            "Qwen/Qwen2.5-Math-PRM-7B",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        for param in self.prm.parameters():
            param.requires_grad = False

        self.cfg = Config(branching_factor=16, max_length_per_stage=100, simulate_max_length=1004, num_steps=16, max_depth=8)

    @torch.no_grad()
    def _model_generate(self, context_tokens: torch.Tensor):
        mcts = MCTS(self.model, self.tokenizer, self.cfg, self.prm, self.prm_tokenizer)
        node = mcts.step(context_tokens)
        return node.tokens
        # return self.model.generate(context_tokens, attention_mask=torch.ones_like(context_tokens), do_sample=True, num_return_sequences=1, max_length=8192)

    def generate_until(self, requests) -> List[str]:
        res = []
        for request in tqdm(requests):
            context = request.args[0]
            until = request.args[1]
            context_tokens = self.tokenizer.encode(context, add_special_tokens=False, return_tensors="pt").to(self.model.device)
            generated_tokens = self._model_generate(context_tokens)
            # print(generated_tokens.shape)
            # decoded = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
            for t in generated_tokens:
                decoded = self.tokenizer.decode(t.tolist())
                for stop_seq in until:
                    if stop_seq in decoded:
                        stop_index = decoded.index(stop_seq)
                        decoded = decoded[:stop_index]
                        break
                res.append(decoded)
        return res

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    @property
    def tokenizer_name(self) -> str:
        return "meta-llama/Llama-3.2-1B-Instruct"

    def chat_template(self, chat_template: Union[bool, str] = False) -> str:
        return ""

    def apply_chat_template(self, chat_history: List[Dict[str, str]], **kwargs) -> str:
        return self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_id()

    @property
    def max_length(self) -> int:
        return self.model_params.max_seq_len

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return 12

    @property
    def device(self) -> str:
        return "cuda"


model = CustomLLaMAModel()
with torch.no_grad():
    results = simple_evaluate(
        model=model,
        tasks=["gsm8k_cot_llama"],
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        num_fewshot=8,
        device="cuda",
        batch_size=64,  # not used
    )

with open(f"results.json", "w") as f:
    json.dump(results, f)

print(results["results"])
