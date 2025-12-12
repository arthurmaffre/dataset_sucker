"""
Evaluation Metrics for GFlowNet Training

Implements:
- KL divergence estimation
- Sample quality metrics
- Convergence diagnostics
- Logging utilities for Weights & Biases
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import wandb
from scipy import stats


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    # KL divergence metrics
    kl_forward: float = 0.0  # KL(p_target || p_gfn)
    kl_reverse: float = 0.0  # KL(p_gfn || p_target)
    kl_symmetric: float = 0.0  # (KL_forward + KL_reverse) / 2

    # Trajectory balance metrics
    tb_loss: float = 0.0
    tb_error_mean: float = 0.0
    tb_error_std: float = 0.0

    # Reward statistics
    log_reward_mean: float = 0.0
    log_reward_std: float = 0.0
    log_reward_max: float = 0.0
    log_reward_min: float = 0.0

    # Policy statistics
    log_pf_mean: float = 0.0
    log_pf_std: float = 0.0
    entropy_mean: float = 0.0

    # Partition function
    log_Z: float = 0.0
    log_Z_estimate: float = 0.0  # Importance sampling estimate

    # Sample diversity
    unique_ratio: float = 0.0
    self_bleu: float = 0.0

    # Training info
    epoch: int = 0
    step: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "eval/kl_forward": self.kl_forward,
            "eval/kl_reverse": self.kl_reverse,
            "eval/kl_symmetric": self.kl_symmetric,
            "eval/tb_loss": self.tb_loss,
            "eval/tb_error_mean": self.tb_error_mean,
            "eval/tb_error_std": self.tb_error_std,
            "eval/log_reward_mean": self.log_reward_mean,
            "eval/log_reward_std": self.log_reward_std,
            "eval/log_reward_max": self.log_reward_max,
            "eval/log_reward_min": self.log_reward_min,
            "eval/log_pf_mean": self.log_pf_mean,
            "eval/log_pf_std": self.log_pf_std,
            "eval/entropy_mean": self.entropy_mean,
            "eval/log_Z": self.log_Z,
            "eval/log_Z_estimate": self.log_Z_estimate,
            "eval/unique_ratio": self.unique_ratio,
            "eval/self_bleu": self.self_bleu,
            "epoch": self.epoch,
            "step": self.step,
        }


class GFlowNetEvaluator:
    """
    Evaluator for GFlowNet training.

    Computes various metrics to monitor convergence and sample quality.
    """

    def __init__(
        self,
        tokenizer,
        log_to_wandb: bool = True,
        eval_batch_size: int = 32,
        num_samples_for_kl: int = 100,
        num_samples_for_diversity: int = 50,
    ):
        """
        Args:
            tokenizer: Tokenizer for decoding sequences
            log_to_wandb: Whether to log to Weights & Biases
            eval_batch_size: Batch size for evaluation
            num_samples_for_kl: Number of samples for KL estimation
            num_samples_for_diversity: Number of samples for diversity metrics
        """
        self.tokenizer = tokenizer
        self.log_to_wandb = log_to_wandb
        self.eval_batch_size = eval_batch_size
        self.num_samples_for_kl = num_samples_for_kl
        self.num_samples_for_diversity = num_samples_for_diversity

        # History for tracking
        self.metrics_history: List[EvaluationMetrics] = []

    def decode_sequences(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs to strings."""
        texts = []
        for seq in token_ids:
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            texts.append(text.strip())
        return texts

    @torch.no_grad()
    def estimate_kl_divergence(
        self,
        gfn_policy,
        target_oracle,
        num_samples: int = None,
        device: str = "cpu",
    ) -> Tuple[float, float]:
        """
        Estimate KL divergence between GFN policy and target distribution.

        KL(p_target || p_gfn) ≈ E_{x ~ p_target}[log p_target(x) - log p_gfn(x)]

        For GFlowNet:
        - p_gfn(x) ∝ exp(log_pf(x)), so log p_gfn(x) = log_pf(x) - log Z
        - At convergence, log p_gfn(x) = log R(x) - log Z

        Args:
            gfn_policy: GFlowNet policy model
            target_oracle: Reward oracle (gives log R(x) = log p_target(x))
            num_samples: Number of samples for estimation
            device: Torch device

        Returns:
            kl_forward: KL(target || gfn)
            kl_reverse: KL(gfn || target)
        """
        if num_samples is None:
            num_samples = self.num_samples_for_kl

        device = torch.device(device)
        gfn_policy.eval()

        # Sample from GFN
        sequences, log_pf_gfn, lengths = gfn_policy.sample_trajectory(
            batch_size=num_samples,
            temperature=1.0,
        )

        # Decode to text
        texts = self.decode_sequences(sequences)

        # Get rewards (log p_target)
        reward_output = target_oracle.compute_reward(texts)
        log_reward = reward_output.log_reward

        # GFN log probability: log p_gfn(x) = log_pf - log_Z
        log_Z = gfn_policy.ln_Z
        log_p_gfn = log_pf_gfn - log_Z

        # KL(gfn || target) = E_gfn[log p_gfn - log p_target]
        # Note: log p_target = log R - log Z_target, but we use unnormalized log R
        # This gives KL up to an unknown constant
        kl_reverse_per_sample = log_p_gfn - log_reward
        kl_reverse = kl_reverse_per_sample.mean().item()

        # For KL(target || gfn), we'd need samples from target
        # Approximate using importance weighting
        # w(x) = p_target(x) / p_gfn(x) = R(x) / (exp(log_pf) / Z) = R(x) * Z / exp(log_pf)
        log_weights = log_reward - log_pf_gfn + log_Z
        weights = F.softmax(log_weights, dim=0)  # Normalize weights

        kl_forward_per_sample = log_reward - log_p_gfn
        kl_forward = (weights * kl_forward_per_sample).sum().item()

        return kl_forward, kl_reverse

    @torch.no_grad()
    def estimate_log_Z(
        self,
        gfn_policy,
        target_oracle,
        num_samples: int = 100,
    ) -> float:
        """
        Estimate log Z using importance sampling.

        Z = E_{x ~ q}[R(x) / q(x)]
        log Z ≈ log(mean(R(x) / q(x))) for x ~ q

        Where q is the GFN policy.
        """
        sequences, log_pf, _ = gfn_policy.sample_trajectory(batch_size=num_samples)
        texts = self.decode_sequences(sequences)

        reward_output = target_oracle.compute_reward(texts)
        log_reward = reward_output.log_reward

        # Importance weights: R(x) / P_F(x)
        log_weights = log_reward - log_pf

        # Log-sum-exp trick for numerical stability
        max_log_w = log_weights.max()
        log_Z_estimate = max_log_w + torch.log(
            torch.exp(log_weights - max_log_w).mean()
        )

        return log_Z_estimate.item()

    def compute_diversity_metrics(
        self,
        texts: List[str],
    ) -> Tuple[float, float]:
        """
        Compute sample diversity metrics.

        Args:
            texts: List of generated texts

        Returns:
            unique_ratio: Fraction of unique samples
            self_bleu: Average BLEU score between pairs (lower = more diverse)
        """
        # Unique ratio
        unique_texts = set(texts)
        unique_ratio = len(unique_texts) / max(len(texts), 1)

        # Self-BLEU (simplified: fraction of shared n-grams)
        if len(texts) < 2:
            return unique_ratio, 0.0

        # Compute average pairwise similarity
        def get_ngrams(text, n=2):
            words = text.lower().split()
            return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

        similarities = []
        for i in range(min(len(texts), 50)):
            for j in range(i + 1, min(len(texts), 50)):
                ngrams_i = get_ngrams(texts[i])
                ngrams_j = get_ngrams(texts[j])

                if len(ngrams_i) == 0 and len(ngrams_j) == 0:
                    sim = 1.0
                elif len(ngrams_i) == 0 or len(ngrams_j) == 0:
                    sim = 0.0
                else:
                    intersection = len(ngrams_i & ngrams_j)
                    union = len(ngrams_i | ngrams_j)
                    sim = intersection / union if union > 0 else 0.0

                similarities.append(sim)

        self_bleu = np.mean(similarities) if similarities else 0.0

        return unique_ratio, self_bleu

    @torch.no_grad()
    def compute_policy_entropy(
        self,
        gfn_policy,
        sequences: torch.Tensor,
    ) -> float:
        """
        Compute average policy entropy over sequences.

        High entropy = more exploration
        Low entropy = more exploitation
        """
        logits = gfn_policy.forward(sequences[:, :-1])
        probs = F.softmax(logits, dim=-1)

        # Entropy: -sum(p * log p)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        return entropy.item()

    @torch.no_grad()
    def evaluate(
        self,
        gfn_policy,
        target_oracle,
        epoch: int = 0,
        step: int = 0,
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> EvaluationMetrics:
        """
        Run full evaluation.

        Args:
            gfn_policy: GFlowNet policy
            target_oracle: Reward oracle
            epoch: Current training epoch
            step: Current training step
            additional_metrics: Extra metrics to log

        Returns:
            EvaluationMetrics with all computed metrics
        """
        gfn_policy.eval()
        device = next(gfn_policy.parameters()).device

        # Sample from GFN
        sequences, log_pf, lengths = gfn_policy.sample_trajectory(
            batch_size=self.num_samples_for_kl,
        )

        texts = self.decode_sequences(sequences)

        # Get rewards
        reward_output = target_oracle.compute_reward(texts)
        log_reward = reward_output.log_reward.to(device)

        # Trajectory balance error
        log_Z = gfn_policy.ln_Z
        tb_error = log_Z + log_pf - log_reward

        # KL divergence
        log_p_gfn = log_pf - log_Z
        kl_reverse = (log_p_gfn - log_reward).mean().item()

        # Importance-weighted forward KL
        log_weights = log_reward - log_pf + log_Z
        weights = F.softmax(log_weights, dim=0)
        kl_forward = (weights * (log_reward - log_p_gfn)).sum().item()

        # Diversity metrics
        diversity_texts = self.decode_sequences(
            gfn_policy.sample_trajectory(self.num_samples_for_diversity)[0]
        )
        unique_ratio, self_bleu = self.compute_diversity_metrics(diversity_texts)

        # Policy entropy
        entropy = self.compute_policy_entropy(gfn_policy, sequences)

        # Log Z estimate
        log_Z_estimate = self.estimate_log_Z(gfn_policy, target_oracle)

        metrics = EvaluationMetrics(
            kl_forward=kl_forward,
            kl_reverse=kl_reverse,
            kl_symmetric=(kl_forward + kl_reverse) / 2,
            tb_loss=tb_error.pow(2).mean().item(),
            tb_error_mean=tb_error.mean().item(),
            tb_error_std=tb_error.std().item(),
            log_reward_mean=log_reward.mean().item(),
            log_reward_std=log_reward.std().item(),
            log_reward_max=log_reward.max().item(),
            log_reward_min=log_reward.min().item(),
            log_pf_mean=log_pf.mean().item(),
            log_pf_std=log_pf.std().item(),
            entropy_mean=entropy,
            log_Z=log_Z.item(),
            log_Z_estimate=log_Z_estimate,
            unique_ratio=unique_ratio,
            self_bleu=self_bleu,
            epoch=epoch,
            step=step,
        )

        self.metrics_history.append(metrics)

        # Log to wandb
        if self.log_to_wandb:
            log_dict = metrics.to_dict()
            if additional_metrics:
                log_dict.update(additional_metrics)

            # Log sample texts
            log_dict["samples/generated_texts"] = wandb.Table(
                data=[[t] for t in texts[:10]],
                columns=["text"],
            )

            wandb.log(log_dict, step=step)

        return metrics

    def log_training_step(
        self,
        tb_loss: float,
        log_Z: float,
        log_pf_mean: float,
        log_reward_mean: float,
        step: int,
        additional: Optional[Dict[str, float]] = None,
    ):
        """Log metrics for a single training step."""
        if not self.log_to_wandb:
            return

        log_dict = {
            "train/tb_loss": tb_loss,
            "train/log_Z": log_Z,
            "train/log_pf_mean": log_pf_mean,
            "train/log_reward_mean": log_reward_mean,
        }

        if additional:
            log_dict.update({f"train/{k}": v for k, v in additional.items()})

        wandb.log(log_dict, step=step)

    def plot_convergence(self):
        """Create convergence plots in wandb."""
        if not self.log_to_wandb or len(self.metrics_history) < 2:
            return

        epochs = [m.epoch for m in self.metrics_history]

        # KL divergence over time
        data_kl = [[e, m.kl_forward, m.kl_reverse, m.kl_symmetric]
                   for e, m in zip(epochs, self.metrics_history)]
        wandb.log({
            "plots/kl_divergence": wandb.Table(
                data=data_kl,
                columns=["epoch", "kl_forward", "kl_reverse", "kl_symmetric"],
            )
        })

        # Log Z convergence
        data_z = [[e, m.log_Z, m.log_Z_estimate]
                  for e, m in zip(epochs, self.metrics_history)]
        wandb.log({
            "plots/log_Z": wandb.Table(
                data=data_z,
                columns=["epoch", "learned_log_Z", "estimated_log_Z"],
            )
        })


def setup_wandb(
    project_name: str = "dataset_sucker",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> None:
    """
    Initialize Weights & Biases logging.

    Args:
        project_name: W&B project name
        run_name: Name for this run
        config: Configuration dictionary
        tags: Tags for the run
    """
    wandb.init(
        project=project_name,
        name=run_name,
        config=config or {},
        tags=tags or [],
        reinit=True,
    )

    # Define custom charts
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")


def log_samples_table(
    samples: List[str],
    rewards: Optional[torch.Tensor] = None,
    log_pf: Optional[torch.Tensor] = None,
    name: str = "samples",
):
    """Log samples as a wandb table."""
    columns = ["sample"]
    data = [[s] for s in samples]

    if rewards is not None:
        columns.append("log_reward")
        for i, r in enumerate(rewards.tolist()):
            data[i].append(r)

    if log_pf is not None:
        columns.append("log_pf")
        for i, p in enumerate(log_pf.tolist()):
            data[i].append(p)

    wandb.log({f"samples/{name}": wandb.Table(data=data, columns=columns)})
