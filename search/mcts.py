import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from vllm import LLM, SamplingParams

STOP_TOKEN = ""


@dataclass
class Config:
    branching_factor: int = 5
    max_length_per_stage: int = 50
    simulate_max_length: int = 100
    num_steps: int = 5
    max_depth: int = 8


@dataclass
class Node:
    stages: List[str]
    depth: int
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0

    @property
    def sequence(self) -> str:
        """Concatenate all stages into a single tensor."""
        if not self.stages:
            return ""
        return "".join(self.stages)

    def is_terminal(self) -> bool:
        if self.depth == 0:
            return False
        if not self.stages:
            return False
        return self.stages[-1] == STOP_TOKEN

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


def score_node(node: Node, prm: AutoModel, prm_tokenizer: AutoTokenizer) -> float:
    # first 2 stages are the prompt and the first stage
    steps = [node.stages[0] + node.stages[1]]

    # Add remaining stages
    for indiv_stage in node.stages[2:]:
        steps.append(indiv_stage)

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
    return min(scores)


class MCTS:
    def __init__(self, model: LLM, tokenizer: AutoTokenizer, cfg: Config, prm: AutoModel, prm_tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.prm = prm
        self.prm_tokenizer = prm_tokenizer
        self.cfg = cfg
        self.stop_tokens = [".", ",", "\n", "\n\n", ":\n\n", ". ", ", ", tokenizer.eos_token]
        self.leaf_nodes = []
        self.root = None

    def split_stage(self, stage: str) -> List[str]:
        stages = []
        old_idx = 0
        for i, word in enumerate(stage):
            if word in self.stop_tokens:
                stages.append(stage[old_idx : i + 1])
                old_idx = i + 1
        if old_idx < len(stage):
            stages.append(stage[old_idx:])
        return stages

    def _expand(self, node: Node) -> None:
        if node.is_terminal() or node.depth >= self.cfg.max_depth:
            return

        outputs = self.model.generate(
            node.sequence,
            use_tqdm=False,
            sampling_params=SamplingParams(
                n=self.cfg.branching_factor,
                max_tokens=self.cfg.max_length_per_stage,
                include_stop_str_in_output=True,
            ),
        )
        for output in outputs:
            generated_text = output.outputs[0].text
            new_token_stages = node.stages.copy()
            new_token_stages.extend(self.split_stage(generated_text))
            new_node = Node(stages=new_token_stages, parent=node, depth=node.depth + 1)
            node.children.append(new_node)

    def _select(self, node: Node) -> Node:
        while node.children:
            if not all(child.visit_count > 0 for child in node.children):
                unvisited = [child for child in node.children if child.visit_count == 0]
                return random.choice(unvisited)
                # scored_unvisited = [(child, score_node(child, self.prm, self.prm_tokenizer)) for child in unvisited]
                # return max(scored_unvisited, key=lambda x: x[1])[0]

            node = max(node.children, key=lambda child: child.score)

            if node.is_terminal() or node.depth >= self.cfg.max_depth:
                return node

        return node

    def _simulate(self, node: Node) -> float:
        if node.is_terminal() or node.depth >= self.cfg.max_depth:
            return node.value
        output = self.model.generate(
            node.sequence,
            use_tqdm=False,
            sampling_params=SamplingParams(
                n=1,
                max_tokens=self.cfg.simulate_max_length,
                include_stop_str_in_output=True,
            ),
        )[0]

        output_text = output.outputs[0].text

        if output_text[-1] == self.tokenizer.eos_token:
            output_text = output_text[:-1]

        new_token_stages = node.stages.copy()
        new_token_stages.extend(self.split_stage(output_text))

        new_leaf_node = Node(
            stages=new_token_stages,
            parent=node,
            depth=node.depth + 1,
        )

        score = score_node(new_leaf_node, self.prm, self.prm_tokenizer)
        new_leaf_node.total_value = score
        new_leaf_node.visit_count = 1
        self.leaf_nodes.append(new_leaf_node)
        return score

    def _backpropagate(self, node: Node, reward: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.total_value += reward
            node = node.parent

    def step(self, initial_sequence: str) -> Node:
        root = Node(stages=[initial_sequence], depth=0)
        self.root = root

        for _ in range(self.cfg.num_steps):
            leaf = self._select(root)
            self._expand(leaf)
            reward = self._simulate(leaf)
            self._backpropagate(leaf, reward)
        return max(self.leaf_nodes, key=lambda node: node.value)
