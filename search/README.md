# Monte Carlo Tree Search 
This experiment is [Llama 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) using [Qwen 7B](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B) as a Process Reward Model for search. 
There are only 2 scripts:
 - `main.py` is the core MCTS logic 
 - `eval.py` uses the [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate on GSM8K.

### Results 
The image below are the results on GSM8K. The performance gains is roughly 10%. PRM scores a node by using the minimum score. I found this to work the best which is surprising and different from the results from the [Qwen paper](https://arxiv.org/abs/2501.07301). The number in the bracket is the branching factor. 
- UCT is standard MCTS from games where the PRM scores the completion only at the end of the trajectory. So, $O(Branching Factor)$ number of invocations of the PRM.
- PUCT is from [AlphaZero](https://arxiv.org/abs/1712.01815) and uses the PRM to select children to expand. 

<div align="center">
    <img src="../assets/mcts.png" alt="GSM8K Results" width="600">
    <p style="font-size: small;"><em>GSM8K Results</em></p>
</div>

# AIME 24
## 1.5B
Default:
Accuracy: 7/30 (23.33%)
cfg=Config(
                    branching_factor=8,
                    max_length_per_stage=256,
                    max_depth=64,
                    simulate_max_length=32768,
                    num_steps=8,
                ):
Accuracy: 12/30 (40.00%)
Config(
                    branching_factor=16,
                    max_length_per_stage=256,
                    max_depth=64,
                    simulate_max_length=32768,
                    num_steps=8,
                )
Accuracy: 10/30 (33.33%)


## 34B
Default:
Accuracy: 16/30 (53.33%)
cfg=Config(
                    branching_factor=8,
                    max_length_per_stage=256,
                    max_depth=64,
                    simulate_max_length=32768,
                    num_steps=8,
                ):
Accuracy: 20/30 (66.67%)
Config(
                    branching_factor=16,
                    max_length_per_stage=256,
                    max_depth=64,
                    simulate_max_length=32768,
                    num_steps=8,
                )
Accuracy: 21/30 (70.00%)

Accuracy: 7/30 (23.33%)
Accuracy: 9/30 (30.00%)
Accuracy: 7/30 (23.33%)

# AIME 25
## 1.5B
Accuracy: 7/30 (23.33%)
Accuracy: 9/30 (30.00%)
Accuracy: 7/30 (23.33%)


## 32B
Accuracy: 15/30 (50.00%)
Accuracy: 15/30 (50.00%)
Accuracy: 19/30 (63.33%)