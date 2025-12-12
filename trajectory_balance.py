"""
Trajectory Balance Loss for GFlowNet Training

Implements the trajectory balance objective from:
"Trajectory Balance: Improved Credit Assignment in GFlowNets"

For autoregressive text generation:
- Forward policy: P_F(s_t | s_{<t}) at each step
- No backward policy needed (sequential generation is deterministic backward)
- Reward: R(x) = p(x) * p(y|x) where p(y|x) comes from LLM oracle

The trajectory balance condition:
    Z * P_F(tau) = R(x) * P_B(tau)

For sequential generation with deterministic backward (P_B = 1):
    Z * P_F(tau) = R(x)
    => log Z + log P_F(tau) = log R(x)
    => log Z + sum_t log P_F(s_t | s_{<t}) = log R(x)

Loss: (log Z + log P_F(tau) - log R(x))^2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TrajectoryBalanceOutput:
    """Output of trajectory balance computation."""

    loss: torch.Tensor
    log_Z: torch.Tensor
    log_pf: torch.Tensor
    log_reward: torch.Tensor
    tb_error: torch.Tensor  # log Z + log P_F - log R

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "tb_loss": self.loss.item(),
            "log_Z": self.log_Z.item(),
            "mean_log_pf": self.log_pf.mean().item(),
            "mean_log_reward": self.log_reward.mean().item(),
            "mean_tb_error": self.tb_error.mean().item(),
            "std_tb_error": self.tb_error.std().item(),
        }


class TrajectoryBalanceLoss(nn.Module):
    """
    Trajectory Balance loss for GFlowNet training.

    The loss is: E_tau [(log Z + log P_F(tau) - log R(x))^2]

    Where:
        - Z is the learnable partition function
        - P_F(tau) is the forward policy probability
        - R(x) is the reward (typically p(x) * p(y|x) for joint distribution)
    """

    def __init__(
        self,
        reward_temperature: float = 1.0,
        reward_clamp_min: float = -100.0,
        reward_clamp_max: float = 100.0,
        use_log_variance_loss: bool = False,
    ):
        """
        Args:
            reward_temperature: Temperature for reward scaling
            reward_clamp_min: Minimum value for log reward (for stability)
            reward_clamp_max: Maximum value for log reward
            use_log_variance_loss: Use log-variance loss variant for stability
        """
        super().__init__()
        self.reward_temperature = reward_temperature
        self.reward_clamp_min = reward_clamp_min
        self.reward_clamp_max = reward_clamp_max
        self.use_log_variance_loss = use_log_variance_loss

    def forward(
        self,
        log_Z: torch.Tensor,
        log_pf: torch.Tensor,
        log_reward: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> TrajectoryBalanceOutput:
        """
        Compute trajectory balance loss.

        Args:
            log_Z: Scalar log partition function
            log_pf: (batch,) log forward policy probabilities
            log_reward: (batch,) log rewards
            weights: (batch,) optional importance weights

        Returns:
            TrajectoryBalanceOutput with loss and diagnostics
        """
        # Clamp log reward for stability
        log_reward = torch.clamp(
            log_reward / self.reward_temperature,
            min=self.reward_clamp_min,
            max=self.reward_clamp_max,
        )

        # Trajectory balance error: log Z + log P_F - log R
        # At convergence, this should be 0 for all trajectories
        tb_error = log_Z + log_pf - log_reward

        if self.use_log_variance_loss:
            # Log-variance loss: more stable for large rewards
            # Minimize variance of (log Z + log P_F - log R)
            mean_error = tb_error.mean()
            loss = ((tb_error - mean_error) ** 2).mean()
        else:
            # Standard squared error loss
            loss = (tb_error ** 2)

            if weights is not None:
                loss = loss * weights
                loss = loss.sum() / weights.sum()
            else:
                loss = loss.mean()

        return TrajectoryBalanceOutput(
            loss=loss,
            log_Z=log_Z.detach(),
            log_pf=log_pf.detach(),
            log_reward=log_reward.detach(),
            tb_error=tb_error.detach(),
        )


class SubTrajectoryBalanceLoss(nn.Module):
    """
    Sub-Trajectory Balance loss for improved credit assignment.

    Instead of only enforcing balance at the full trajectory level,
    this enforces balance at every prefix, which provides denser
    gradient signal and faster convergence.

    For prefix s_{1:t}:
        Z * P_F(s_{1:t}) = F(s_t)

    Where F(s_t) is a learned flow function at state s_t.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 256,
        reward_clamp_min: float = -100.0,
        reward_clamp_max: float = 100.0,
    ):
        super().__init__()
        self.reward_clamp_min = reward_clamp_min
        self.reward_clamp_max = reward_clamp_max

        # Flow estimator network
        self.flow_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        log_Z: torch.Tensor,
        log_pf_per_step: torch.Tensor,  # (batch, seq_len)
        hidden_states: torch.Tensor,  # (batch, seq_len, d_model)
        log_reward: torch.Tensor,  # (batch,)
        attention_mask: torch.Tensor,  # (batch, seq_len)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute sub-trajectory balance loss.

        Args:
            log_Z: Scalar log partition function
            log_pf_per_step: (batch, seq_len) log P_F at each step
            hidden_states: (batch, seq_len, d_model) transformer hidden states
            log_reward: (batch,) final rewards
            attention_mask: (batch, seq_len) valid token mask

        Returns:
            loss: Scalar loss
            diagnostics: Dictionary of diagnostic values
        """
        batch_size, seq_len = log_pf_per_step.shape

        # Estimate flow at each state
        log_flows = self.flow_net(hidden_states).squeeze(-1)  # (batch, seq_len)

        # Cumulative log P_F up to each position
        cumsum_log_pf = torch.cumsum(log_pf_per_step, dim=1)  # (batch, seq_len)

        # Sub-trajectory balance at each step:
        # log Z + cumsum log P_F(1:t) = log F(s_t)
        stb_error = log_Z + cumsum_log_pf - log_flows

        # Mask out padding
        stb_error = stb_error * attention_mask

        # Loss: mean squared error over all valid positions
        stb_loss = (stb_error ** 2).sum() / attention_mask.sum()

        # Terminal constraint: log F(terminal) = log R
        # Get last valid position for each sequence
        lengths = attention_mask.sum(dim=1).long()
        batch_indices = torch.arange(batch_size, device=log_flows.device)
        terminal_log_flows = log_flows[batch_indices, lengths - 1]

        log_reward = torch.clamp(log_reward, min=self.reward_clamp_min, max=self.reward_clamp_max)
        terminal_loss = ((terminal_log_flows - log_reward) ** 2).mean()

        total_loss = stb_loss + terminal_loss

        diagnostics = {
            "stb_loss": stb_loss.detach(),
            "terminal_loss": terminal_loss.detach(),
            "mean_stb_error": stb_error.sum() / attention_mask.sum(),
        }

        return total_loss, diagnostics


class DetailedBalanceLoss(nn.Module):
    """
    Detailed Balance loss - enforces balance at each transition.

    For transition s_t -> s_{t+1}:
        F(s_t) * P_F(s_{t+1} | s_t) = F(s_{t+1}) * P_B(s_t | s_{t+1})

    For autoregressive generation, P_B = 1 (deterministic backward), so:
        F(s_t) * P_F(s_{t+1} | s_t) = F(s_{t+1})

    This is the most fine-grained balance condition.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Flow estimator for each state
        self.flow_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        log_pf_per_step: torch.Tensor,  # (batch, seq_len)
        hidden_states: torch.Tensor,  # (batch, seq_len, d_model)
        log_reward: torch.Tensor,  # (batch,)
        attention_mask: torch.Tensor,  # (batch, seq_len)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute detailed balance loss.

        Args:
            log_pf_per_step: (batch, seq_len) log P_F at each step
            hidden_states: (batch, seq_len, d_model) transformer hidden states
            log_reward: (batch,) final rewards
            attention_mask: (batch, seq_len) valid token mask

        Returns:
            loss: Scalar loss
            diagnostics: Dictionary of diagnostic values
        """
        batch_size, seq_len = log_pf_per_step.shape

        # Estimate log flow at each state
        log_flows = self.flow_net(hidden_states).squeeze(-1)  # (batch, seq_len)

        # Detailed balance for each transition:
        # log F(s_t) + log P_F(s_{t+1} | s_t) = log F(s_{t+1})
        # => log F(s_t) + log P_F(t+1) - log F(s_{t+1}) = 0
        log_F_t = log_flows[:, :-1]  # (batch, seq_len-1)
        log_F_next = log_flows[:, 1:]  # (batch, seq_len-1)
        log_pf_transition = log_pf_per_step[:, :-1]  # (batch, seq_len-1)

        db_error = log_F_t + log_pf_transition - log_F_next

        # Mask transitions (need both positions valid)
        transition_mask = attention_mask[:, :-1] * attention_mask[:, 1:]
        db_error = db_error * transition_mask

        db_loss = (db_error ** 2).sum() / (transition_mask.sum() + 1e-8)

        # Terminal constraint: log F(terminal) = log R
        lengths = attention_mask.sum(dim=1).long()
        batch_indices = torch.arange(batch_size, device=log_flows.device)
        terminal_log_flows = log_flows[batch_indices, lengths - 1]

        terminal_loss = ((terminal_log_flows - log_reward) ** 2).mean()

        # Initial flow = Z constraint
        log_Z = log_flows[:, 0].mean()  # Use first flow as Z estimate
        z_consistency_loss = ((log_flows[:, 0] - log_Z) ** 2).mean()

        total_loss = db_loss + terminal_loss + 0.1 * z_consistency_loss

        diagnostics = {
            "db_loss": db_loss.detach(),
            "terminal_loss": terminal_loss.detach(),
            "z_consistency_loss": z_consistency_loss.detach(),
            "mean_db_error": db_error.sum() / (transition_mask.sum() + 1e-8),
            "estimated_log_Z": log_Z.detach(),
        }

        return total_loss, diagnostics


def compute_kl_divergence(
    log_p_target: torch.Tensor,
    log_p_model: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute KL divergence KL(target || model) from samples.

    KL(p || q) = E_p[log p - log q]

    Args:
        log_p_target: (batch,) log probabilities under target
        log_p_model: (batch,) log probabilities under model
        weights: (batch,) optional importance weights

    Returns:
        kl: Scalar KL divergence estimate
    """
    kl_per_sample = log_p_target - log_p_model

    if weights is not None:
        kl = (kl_per_sample * weights).sum() / weights.sum()
    else:
        kl = kl_per_sample.mean()

    return kl


def compute_reverse_kl_divergence(
    log_p_target: torch.Tensor,
    log_p_model: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute reverse KL divergence KL(model || target) from samples.

    KL(q || p) = E_q[log q - log p]

    Note: This is estimated using samples from model (q), so importance
    weighting may be needed if samples are from target.

    Args:
        log_p_target: (batch,) log probabilities under target
        log_p_model: (batch,) log probabilities under model
        weights: (batch,) optional importance weights

    Returns:
        kl: Scalar reverse KL divergence estimate
    """
    kl_per_sample = log_p_model - log_p_target

    if weights is not None:
        kl = (kl_per_sample * weights).sum() / weights.sum()
    else:
        kl = kl_per_sample.mean()

    return kl
