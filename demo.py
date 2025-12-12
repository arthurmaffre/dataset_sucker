"""
Demo Script for Dataset Sucker

A minimal example showing GFlowNet training with trajectory balance
to learn the distribution of GPT-2.

This is a simplified version for quick testing and understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from typing import Tuple, List, Optional


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "d_model": 128,
    "nhead": 4,
    "num_layers": 2,
    "max_len": 32,
    "batch_size": 8,
    "epochs": 500,
    "lr": 5e-4,
    "lr_z": 1e-2,
    "eval_every": 25,
    "temperature": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
}


# ============================================================================
# GFlowNet Transformer Policy
# ============================================================================

class GFlowNetTransformer(nn.Module):
    """
    Simplified Transformer policy for GFlowNet.

    Generates sequences token-by-token with learnable log Z.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 64,
        pad_token_id: int = 0,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Learnable log partition function
        self.ln_Z = nn.Parameter(torch.tensor(0.0))

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for all positions."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(pos)

        # Transformer with causal mask
        mask = self._causal_mask(seq_len, device)
        x = self.transformer(x, mask=mask)

        # Project to vocabulary
        logits = self.output_proj(x)
        return logits

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample sequences from the policy.

        Returns:
            sequences: (batch, seq_len) token ids
            log_pf: (batch,) log forward probability
        """
        if max_len is None:
            max_len = self.max_len

        device = next(self.parameters()).device

        # Start with BOS
        sequences = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        log_pf = torch.zeros(batch_size, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            # Get logits for next token
            logits = self.forward(sequences)[:, -1, :] / temperature

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, 1)

            # Update log probability
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(1, next_tokens).squeeze(-1)
            log_pf = log_pf + token_log_probs * (~finished).float()

            # Check for EOS
            finished = finished | (next_tokens.squeeze(-1) == self.eos_token_id)

            # Append
            sequences = torch.cat([sequences, next_tokens], dim=1)

            if finished.all():
                break

        return sequences, log_pf

    def compute_log_pf(self, sequences: torch.Tensor) -> torch.Tensor:
        """Compute log P_F for given sequences."""
        if sequences.size(1) < 2:
            return torch.zeros(sequences.size(0), device=sequences.device)

        # Get logits
        logits = self.forward(sequences[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs of actual tokens
        targets = sequences[:, 1:]
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

        # Sum
        return token_log_probs.sum(dim=-1)


# ============================================================================
# Training
# ============================================================================

def compute_lm_log_prob(model, input_ids: torch.Tensor, pad_token_id: int = None) -> torch.Tensor:
    """Compute log p(sequence) under the LM."""
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

        # Per-sequence log prob: sum of log p(token_t | token_{<t})
        logits = outputs.logits[:, :-1, :]  # (batch, seq-1, vocab)
        targets = input_ids[:, 1:]  # (batch, seq-1)

        log_p = F.log_softmax(logits, dim=-1)
        token_log_p = log_p.gather(2, targets.unsqueeze(-1)).squeeze(-1)

        # Mask padding if pad_token_id provided
        if pad_token_id is not None:
            mask = (targets != pad_token_id).float()
            token_log_p = token_log_p * mask

        seq_log_p = token_log_p.sum(dim=-1)

    return seq_log_p


def train():
    """Main training function."""
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    # Load GPT-2 as the oracle (target distribution)
    print("Loading GPT-2 oracle...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    oracle = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    oracle.eval()

    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Create GFlowNet policy
    print("Creating GFlowNet policy...")
    gfn = GFlowNetTransformer(
        vocab_size=vocab_size,
        d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"],
        num_layers=CONFIG["num_layers"],
        max_len=CONFIG["max_len"],
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    ).to(device)

    num_params = sum(p.numel() for p in gfn.parameters())
    print(f"GFlowNet params: {num_params:,}")

    # Optimizers (separate lr for log Z)
    policy_params = [p for n, p in gfn.named_parameters() if n != "ln_Z"]
    optimizer = optim.AdamW(policy_params, lr=CONFIG["lr"], weight_decay=0.01)
    z_optimizer = optim.Adam([gfn.ln_Z], lr=CONFIG["lr_z"])

    # W&B init
    wandb.init(
        project="dataset_sucker",
        name=f"gfn-transformer-demo",
        config=CONFIG,
    )

    # Training loop
    print(f"\nTraining for {CONFIG['epochs']} epochs...")

    for epoch in tqdm(range(CONFIG["epochs"]), desc="Training"):
        gfn.train()

        # Sample from GFN
        sequences, log_pf = gfn.sample(
            batch_size=CONFIG["batch_size"],
            temperature=CONFIG["temperature"],
        )

        # Compute reward: log R(x) = log p_oracle(x)
        log_reward = compute_lm_log_prob(oracle, sequences, tokenizer.pad_token_id)

        # Trajectory Balance loss: (log Z + log P_F - log R)^2
        tb_error = gfn.ln_Z + log_pf - log_reward
        tb_loss = tb_error.pow(2).mean()

        # Backward
        optimizer.zero_grad()
        z_optimizer.zero_grad()
        tb_loss.backward()
        torch.nn.utils.clip_grad_norm_(gfn.parameters(), 1.0)
        optimizer.step()
        z_optimizer.step()

        # Log to wandb
        wandb.log({
            "train/tb_loss": tb_loss.item(),
            "train/log_Z": gfn.ln_Z.item(),
            "train/log_pf_mean": log_pf.mean().item(),
            "train/log_reward_mean": log_reward.mean().item(),
            "train/tb_error_mean": tb_error.mean().item(),
            "train/tb_error_std": tb_error.std().item(),
            "epoch": epoch,
        })

        # Evaluation
        if epoch % CONFIG["eval_every"] == 0:
            gfn.eval()

            # Sample from GFN
            with torch.no_grad():
                eval_seqs, eval_log_pf = gfn.sample(batch_size=32)
                eval_log_reward = compute_lm_log_prob(oracle, eval_seqs, tokenizer.pad_token_id)

            # KL divergence estimate: KL(gfn || oracle) ≈ E_gfn[log p_gfn - log p_oracle]
            log_p_gfn = eval_log_pf - gfn.ln_Z
            kl_reverse = (log_p_gfn - eval_log_reward).mean().item()

            # Sample from oracle for forward KL
            with torch.no_grad():
                oracle_input = torch.full((32, 1), tokenizer.eos_token_id, device=device)
                oracle_samples = oracle.generate(
                    oracle_input,
                    max_length=CONFIG["max_len"],
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
                oracle_log_p = compute_lm_log_prob(oracle, oracle_samples, tokenizer.pad_token_id)
                gfn_log_pf_oracle = gfn.compute_log_pf(oracle_samples)
                gfn_log_p_oracle = gfn_log_pf_oracle - gfn.ln_Z

            kl_forward = (oracle_log_p - gfn_log_p_oracle).mean().item()

            # Diversity: unique samples
            unique_ratio = len(set(tuple(s.tolist()) for s in eval_seqs)) / len(eval_seqs)

            wandb.log({
                "eval/kl_forward": kl_forward,
                "eval/kl_reverse": kl_reverse,
                "eval/kl_symmetric": (kl_forward + kl_reverse) / 2,
                "eval/unique_ratio": unique_ratio,
                "epoch": epoch,
            })

            # Decode and log samples
            sample_texts = [tokenizer.decode(s, skip_special_tokens=True)[:100] for s in eval_seqs[:5]]
            oracle_texts = [tokenizer.decode(s, skip_special_tokens=True)[:100] for s in oracle_samples[:5]]

            wandb.log({
                "samples/gfn": wandb.Table(data=[[t] for t in sample_texts], columns=["text"]),
                "samples/oracle": wandb.Table(data=[[t] for t in oracle_texts], columns=["text"]),
            })

            print(f"\n[Epoch {epoch}] TB Loss: {tb_loss.item():.4f}, "
                  f"log Z: {gfn.ln_Z.item():.4f}, "
                  f"KL(fwd): {kl_forward:.4f}, KL(rev): {kl_reverse:.4f}, "
                  f"Unique: {unique_ratio:.2%}")
            print(f"  GFN sample: {sample_texts[0][:80]}...")
            print(f"  Oracle sample: {oracle_texts[0][:80]}...")

    # Final evaluation
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final log Z: {gfn.ln_Z.item():.4f}")

    # Save model
    torch.save({
        "model_state_dict": gfn.state_dict(),
        "config": CONFIG,
    }, "gfn_demo_checkpoint.pt")
    print("Model saved to gfn_demo_checkpoint.pt")

    wandb.finish()
    print("\nDataset successfully sucked! 🧛")


if __name__ == "__main__":
    train()
