# Dataset Sucker

Train a GFlowNet to sample from the joint distribution `p(x) * p(y|x)` where `p(y|x)` comes from an LLM (GPT-4).

The idea: "resurrect" training data from LLMs by learning what inputs `x` produce high-probability responses `y`.

## Theory

GFlowNets learn to sample proportionally to a reward function. Here:

```
R(x) = p(x) * p(y|x)
```

Where:
- `p(x)` = prior probability of input (GPT-2)
- `p(y|x)` = likelihood of LLM response given input (GPT-4 API logprobs)

Using **Trajectory Balance**, we train the policy such that:

```
Z * P_F(x) = R(x)
```

In log-space:

```
log Z + log P_F(x) = log R(x)
```

The TB loss minimizes `(log Z + log P_F - log R)²`.

## Installation

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

## Usage

### Quick Demo (GPT-2 only, no API needed)

```bash
python demo.py
```

### Full Training with GPT-4

```bash
# Basic run
python suck_gpt4.py --task raw --epochs 500

# With more parallelism and custom settings
python suck_gpt4.py \
    --task raw \
    --batch-size 32 \
    --max-api-workers 32 \
    --lr-z 0.5 \
    --llm-temperature 1.5 \
    --epochs 1000

# Resume from checkpoint
python suck_gpt4.py --resume checkpoint_gfn_raw_epoch150.pt
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `raw` | Task type: `raw`, `assistant`, `creative`, `code` |
| `--model` | `gpt-4o-mini` | OpenAI model for p(y\|x) |
| `--batch-size` | `16` | Batch size for training |
| `--epochs` | `500` | Number of training epochs |
| `--lr` | `3e-4` | Learning rate for policy |
| `--lr-z` | `0.1` | Learning rate for log Z |
| `--temperature` | `1.0` | Sampling temperature for GFlowNet |
| `--llm-temperature` | `1.0` | Temperature for GPT-4 generation |
| `--max-api-workers` | `8` | Parallel API calls per batch |
| `--d-model` | `256` | Transformer hidden dimension |
| `--num-layers` | `4` | Number of transformer layers |
| `--max-len` | `32` | Maximum sequence length |
| `--resume` | `None` | Path to checkpoint to resume from |
| `--save-every` | `25` | Save checkpoint every N epochs |

## Project Structure

```
dataset_sucker/
├── suck_gpt4.py          # Main training script with GPT-4 API
├── demo.py               # Simplified demo with GPT-2 only
├── gflownet_policy.py    # Transformer GFlowNet policy
├── trajectory_balance.py # TB loss implementation
├── reward_oracle.py      # Reward oracles (HuggingFace, OpenAI)
├── evaluation.py         # Metrics and W&B logging
├── train.py              # Generic training script
└── requirements.txt
```

## Metrics

Training logs to Weights & Biases:

- `train/tb_loss` - Trajectory Balance loss (should decrease)
- `train/log_Z` - Learned log partition function
- `train/log_pf_mean` - Mean log probability under GFlowNet
- `train/log_reward_mean` - Mean log reward = log p(x) + log p(y|x)
- `eval/kl_reverse` - KL(GFN || target), should approach 0

## Notes

### API Rate Limits

OpenAI has rate limits depending on your tier:
- Tier 1: 10,000 requests/day
- Tier 2+: No daily limit

With `batch_size=32` and `epochs=500`, you need ~16,000 requests. Consider:
- Using the Batch API for 50% cost reduction
- Spreading training across multiple days
- Upgrading your OpenAI tier

### Why log p(x) >> log p(y|x)?

Typical values:
- `log p(x)` ~ -800 (random tokens under GPT-2)
- `log p(y|x)` ~ -30 (fluent GPT-4 response)

This is expected: garbage input has very low prior probability, while GPT-4 generates fluent text with high per-token probabilities.

### Convergence

- `tb_loss` should decrease over training
- `log_Z` should stabilize around `mean(log_reward - log_pf)`
- Sample quality improves as the GFlowNet learns to generate plausible inputs

## Citation

Based on GFlowNet trajectory balance:

```bibtex
@article{malkin2022trajectory,
  title={Trajectory balance: Improved credit assignment in GFlowNets},
  author={Malkin, Nikolay and Jain, Moksh and Bengio, Emmanuel and Sun, Chen and Bengio, Yoshua},
  journal={NeurIPS},
  year={2022}
}
```
