"""
Suck GPT-4's Distribution

Train a GFlowNet to sample contexts x such that p(x) * p(y|x) is maximized,
where p(y|x) is computed via the OpenAI API.

This is the "dataset resurrection" approach: we're learning what training
data distribution could have produced GPT-4's behavior.

Usage:
    export OPENAI_API_KEY="sk-..."
    python suck_gpt4.py --task classification --epochs 500
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from transformers import GPT2Tokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Model
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    max_len: int = 32  # Shorter sequences

    # Training
    batch_size: int = 16  # Larger batch for better gradient estimates
    epochs: int = 500
    lr: float = 3e-4
    lr_z: float = 0.1
    temperature: float = 1.0

    # API
    openai_model: str = "gpt-4o-mini"
    max_api_workers: int = 8  # Parallel API calls per batch
    llm_temperature: float = 1.0  # Temperature for GPT-4 generation

    # Evaluation
    eval_every: int = 5
    num_eval_samples: int = 8

    # Task-specific
    task: str = "raw"  # Default: no constraints, learn raw p(x)p(y|x)
    target_response: Optional[str] = None
    system_prompt: Optional[str] = None

    # Device
    device: str = "auto"

    # Checkpointing
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    save_every: int = 25  # Save checkpoint every N epochs


# ============================================================================
# Task Definitions
# ============================================================================

TASKS = {
    "raw": {
        "system_prompt": None,  # No system prompt - raw completion
        "target_response": None,
        "description": "Learn the raw joint distribution p(x)p(y|x) - no constraints",
    },
    "assistant": {
        "system_prompt": "You are a helpful assistant.",
        "target_response": None,
        "description": "Learn inputs that get helpful assistant responses",
    },
    "creative": {
        "system_prompt": "You are a creative writer.",
        "target_response": None,
        "description": "Learn inputs that get creative responses",
    },
    "code": {
        "system_prompt": None,  # No constraint, let it figure out code contexts
        "target_response": None,
        "description": "Learn contexts where GPT-4 naturally produces code",
    },
}


# ============================================================================
# GFlowNet Transformer
# ============================================================================

class GFlowNetTransformer(nn.Module):
    """Transformer-based GFlowNet policy."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        max_len: int = 64,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(0.1)

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
        self.ln_f = nn.LayerNorm(d_model)

        # Output
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Learnable log partition function
        self.ln_Z = nn.Parameter(torch.tensor(0.0))

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Transformer
        mask = self._causal_mask(seq_len, device)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)

        # Output logits
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample sequences with log probabilities."""
        if max_len is None:
            max_len = self.max_len

        device = next(self.parameters()).device

        # Start with BOS
        sequences = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        log_pf = torch.zeros(batch_size, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(sequences)[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, 1)

            # Log probability
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(1, next_tokens).squeeze(-1)
            log_pf = log_pf + token_log_probs * (~finished).float()

            # Check EOS
            finished = finished | (next_tokens.squeeze(-1) == self.eos_token_id)
            sequences = torch.cat([sequences, next_tokens], dim=1)

            if finished.all():
                break

        return sequences, log_pf

    def compute_log_pf(self, sequences: torch.Tensor) -> torch.Tensor:
        """Compute log P_F for given sequences."""
        if sequences.size(1) < 2:
            return torch.zeros(sequences.size(0), device=sequences.device)

        logits = self.forward(sequences[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)
        targets = sequences[:, 1:]
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=-1)


# ============================================================================
# OpenAI Reward Oracle
# ============================================================================

class GPT4RewardOracle:
    """
    Compute rewards using GPT-4 API.

    R(x) = p(x) * p(y|x)

    Where:
    - p(x) is computed using GPT-2 as prior
    - p(y|x) is computed via OpenAI API logprobs
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        target_response: Optional[str] = None,
        max_workers: int = 8,
        llm_temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.target_response = target_response
        self.max_workers = max_workers
        self.llm_temperature = llm_temperature
        self.device = torch.device(device)

        # OpenAI client
        self.client = OpenAI()

        # GPT-2 for prior p(x)
        from transformers import GPT2LMHeadModel
        self.prior_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.prior_model.to(self.device)
        self.prior_model.eval()

        # Cache
        self._cache: Dict[str, float] = {}

    @torch.no_grad()
    def compute_prior_log_prob(
        self,
        sequences: torch.Tensor,
        tokenizer,
    ) -> torch.Tensor:
        """Compute log p(x) using GPT-2."""
        log_probs = []

        for seq in sequences:
            # Truncate to GPT-2 max length
            seq = seq[:1024]
            input_ids = seq.unsqueeze(0).to(self.device)

            outputs = self.prior_model(input_ids, labels=input_ids)
            # Per-token log prob
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            log_p = F.log_softmax(logits, dim=-1)
            token_log_p = log_p.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            seq_log_p = token_log_p.sum().item()
            log_probs.append(seq_log_p)

        return torch.tensor(log_probs, device=self.device)

    def _call_api_single(self, text: str) -> Tuple[str, float, str]:
        """Make a single API call. Returns (text, log_prob, response)."""
        cache_key = f"{self.model}:{self.system_prompt}:{text}"

        # Check cache first
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return (text, cached["log_prob"], cached["response"])

        try:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": text})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=100,
                temperature=self.llm_temperature,
                logprobs=True,
                top_logprobs=5,
            )

            generated_text = response.choices[0].message.content.strip()

            # Sum log probs of generated tokens
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                total_log_prob = sum(
                    token.logprob for token in response.choices[0].logprobs.content
                )
            else:
                total_log_prob = 0.0

            # Cache the result
            self._cache[cache_key] = {"log_prob": total_log_prob, "response": generated_text}
            return (text, total_log_prob, generated_text)

        except Exception as e:
            print(f"API error for '{text[:30]}...': {e}")
            return (text, -10.0, f"[ERROR: {e}]")

    def compute_likelihood_log_prob(
        self,
        texts: List[str],
        return_responses: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[str]]]:
        """Compute log p(y|x) using OpenAI API with parallel requests."""

        # Use ThreadPoolExecutor for parallel API calls
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all API calls in parallel
            future_to_text = {
                executor.submit(self._call_api_single, text): text
                for text in texts
            }

            # Collect results as they complete
            for future in as_completed(future_to_text):
                text, log_prob, response = future.result()
                results[text] = {"log_prob": log_prob, "response": response}

        # Reconstruct results in original order
        log_probs = [results[text]["log_prob"] for text in texts]
        responses = [results[text]["response"] for text in texts] if return_responses else None

        return torch.tensor(log_probs, device=self.device), responses

    def compute_reward(
        self,
        sequences: torch.Tensor,
        texts: List[str],
        tokenizer,
        return_responses: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[str]]]:
        """
        Compute reward R(x) = p(x) * p(y|x).

        Returns:
            log_reward: log R(x)
            log_p_x: log p(x)
            log_p_y_given_x: log p(y|x)
            responses: (optional) list of GPT-4 responses
        """
        # Prior: log p(x)
        log_p_x = self.compute_prior_log_prob(sequences, tokenizer)

        # Likelihood: log p(y|x)
        log_p_y_given_x, responses = self.compute_likelihood_log_prob(texts, return_responses=return_responses)

        # Total reward
        log_reward = log_p_x + log_p_y_given_x

        return log_reward, log_p_x, log_p_y_given_x, responses


# ============================================================================
# Training
# ============================================================================

def train(config: Config):
    """Main training loop."""

    # Device
    if config.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(config.device)
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # Task configuration
    task_config = TASKS.get(config.task, TASKS["raw"])
    system_prompt = config.system_prompt or task_config["system_prompt"]
    target_response = config.target_response or task_config["target_response"]
    print(f"\nTask: {config.task}")
    print(f"Description: {task_config['description']}")
    print(f"System prompt: {system_prompt}")
    print(f"Target response: {target_response}")

    # Reward oracle
    print(f"\nInitializing GPT-4 reward oracle ({config.openai_model})...")
    print(f"Parallel API workers: {config.max_api_workers}")
    oracle = GPT4RewardOracle(
        model=config.openai_model,
        system_prompt=system_prompt,
        target_response=target_response,
        max_workers=config.max_api_workers,
        llm_temperature=config.llm_temperature,
        device=str(device),
    )

    # GFlowNet
    print("Creating GFlowNet policy...")
    gfn = GFlowNetTransformer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        max_len=config.max_len,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    ).to(device)

    num_params = sum(p.numel() for p in gfn.parameters())
    print(f"Parameters: {num_params:,}")

    # Optimizers for GFlowNet training
    policy_params = [p for n, p in gfn.named_parameters() if n != "ln_Z"]
    optimizer = optim.AdamW(policy_params, lr=config.lr, weight_decay=0.01)
    z_optimizer = optim.Adam([gfn.ln_Z], lr=config.lr_z)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')

    if config.resume_from and os.path.exists(config.resume_from):
        print(f"\nResuming from checkpoint: {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=device)
        gfn.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "z_optimizer_state_dict" in checkpoint:
            z_optimizer.load_state_dict(checkpoint["z_optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float('inf'))
        print(f"Resumed from epoch {start_epoch}, ln_Z = {gfn.ln_Z.item():.2f}, best_loss = {best_loss:.2f}")
    else:
        # Initialize ln_Z based on first batch
        print("\nInitializing ln_Z from first batch...")
        with torch.no_grad():
            init_seqs, init_log_pf = gfn.sample(batch_size=config.batch_size)
            init_texts = [tokenizer.decode(s, skip_special_tokens=True).strip() for s in init_seqs]
            init_log_reward, _, _, _ = oracle.compute_reward(init_seqs, init_texts, tokenizer)

            # ln_Z should be: log_reward - log_pf (at equilibrium)
            init_ln_z = (init_log_reward - init_log_pf).mean().item()
            gfn.ln_Z.data.fill_(init_ln_z)
            print(f"Initialized ln_Z to {init_ln_z:.2f}")

    # W&B
    wandb.init(
        project="dataset_sucker",
        name=f"suck-{config.openai_model}-{config.task}",
        config=vars(config),
        resume="allow" if config.resume_from else None,
    )

    # Training
    print(f"\nStarting training from epoch {start_epoch} to {config.epochs}...")
    print("="*60)

    for epoch in tqdm(range(start_epoch, config.epochs), desc="Sucking GPT-4", initial=start_epoch, total=config.epochs):
        gfn.train()

        # Sample from GFN
        sequences, log_pf = gfn.sample(
            batch_size=config.batch_size,
            temperature=config.temperature,
        )

        # Decode to text
        texts = [tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in sequences]

        # Compute reward
        log_reward, log_p_x, log_p_y_given_x, _ = oracle.compute_reward(
            sequences, texts, tokenizer
        )

        # Trajectory Balance loss
        tb_error = gfn.ln_Z + log_pf - log_reward
        tb_loss = tb_error.pow(2).mean()

        # Backward
        optimizer.zero_grad()
        z_optimizer.zero_grad()
        tb_loss.backward()
        optimizer.step()
        z_optimizer.step()

        # Log
        wandb.log({
            "train/tb_loss": tb_loss.item(),
            "train/log_Z": gfn.ln_Z.item(),
            "train/log_pf_mean": log_pf.mean().item(),
            "train/log_reward_mean": log_reward.mean().item(),
            "train/log_p_x_mean": log_p_x.mean().item(),
            "train/log_p_y_given_x_mean": log_p_y_given_x.mean().item(),
            "train/tb_error_mean": tb_error.mean().item(),
            "epoch": epoch,
        })

        # Evaluation
        if epoch % config.eval_every == 0:
            gfn.eval()

            with torch.no_grad():
                eval_seqs, eval_log_pf = gfn.sample(batch_size=config.num_eval_samples)
                eval_texts = [tokenizer.decode(s, skip_special_tokens=True).strip() for s in eval_seqs]

            eval_log_reward, eval_log_p_x, eval_log_p_y_given_x, eval_responses = oracle.compute_reward(
                eval_seqs, eval_texts, tokenizer, return_responses=True
            )

            # KL estimate
            log_p_gfn = eval_log_pf - gfn.ln_Z
            kl_reverse = (log_p_gfn - eval_log_reward).mean().item()

            # Diversity
            unique_ratio = len(set(eval_texts)) / len(eval_texts)

            # Detailed wandb logging
            wandb.log({
                "eval/kl_reverse": kl_reverse,
                "eval/unique_ratio": unique_ratio,
                "eval/log_reward_mean": eval_log_reward.mean().item(),
                "eval/log_reward_std": eval_log_reward.std().item(),
                "eval/log_p_x_mean": eval_log_p_x.mean().item(),
                "eval/log_p_y_given_x_mean": eval_log_p_y_given_x.mean().item(),
                # Samples table with GFN input and GPT-4 response
                "samples/gfn_generations": wandb.Table(
                    data=[
                        [t[:200], r, float(log_r), float(log_px), float(log_py)]
                        for t, r, log_r, log_px, log_py in zip(
                            eval_texts[:10],
                            eval_responses[:10],
                            eval_log_reward[:10],
                            eval_log_p_x[:10],
                            eval_log_p_y_given_x[:10],
                        )
                    ],
                    columns=["GFN_sample (x)", f"{config.openai_model}_response (y)", "log_R", "log_p(x)", "log_p(y|x)"],
                ),
                "epoch": epoch,
            })

            # Terminal output - clear and professional
            print(f"\n{'='*70}")
            print(f"[Epoch {epoch}] TB Loss: {tb_loss.item():.4f} | log Z: {gfn.ln_Z.item():.2f} | KL(rev): {kl_reverse:.4f} | Unique: {unique_ratio:.0%}")
            print(f"{'='*70}")
            print(f"  GFlowNet sample (x):     \"{eval_texts[0][:70]}...\"")
            print(f"  {config.openai_model} response (y): \"{eval_responses[0]}\"")
            print(f"  Reward breakdown:        log R = {eval_log_reward[0].item():.2f} = log p(x):{eval_log_p_x[0].item():.2f} + log p(y|x):{eval_log_p_y_given_x[0].item():.2f}")

            # Save best
            if tb_loss.item() < best_loss:
                best_loss = tb_loss.item()
                torch.save({
                    "model_state_dict": gfn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "z_optimizer_state_dict": z_optimizer.state_dict(),
                    "config": vars(config),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "ln_Z": gfn.ln_Z.item(),
                }, f"best_gfn_{config.task}.pt")

        # Periodic checkpoint save
        if epoch > 0 and epoch % config.save_every == 0:
            torch.save({
                "model_state_dict": gfn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "z_optimizer_state_dict": z_optimizer.state_dict(),
                "config": vars(config),
                "epoch": epoch,
                "best_loss": best_loss,
                "ln_Z": gfn.ln_Z.item(),
            }, f"checkpoint_gfn_{config.task}_epoch{epoch}.pt")
            print(f"  [Checkpoint saved: checkpoint_gfn_{config.task}_epoch{epoch}.pt]")

    # Final
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final log Z: {gfn.ln_Z.item():.4f}")

    # Generate final samples
    print("\nFinal generated samples:")
    gfn.eval()
    with torch.no_grad():
        final_seqs, _ = gfn.sample(batch_size=10)
        final_texts = [tokenizer.decode(s, skip_special_tokens=True).strip() for s in final_seqs]

    for i, text in enumerate(final_texts):
        print(f"  {i+1}. {text[:100]}")

    wandb.finish()
    print("\n🧛 Dataset successfully sucked from GPT-4!")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Suck GPT-4's distribution")

    parser.add_argument("--task", type=str, default="raw",
                        choices=list(TASKS.keys()),
                        help="Task type")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-z", type=float, default=0.1,
                        help="Learning rate for log Z")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--target-response", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., best_gfn_raw.pt)")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--max-api-workers", type=int, default=8,
                        help="Max parallel API workers per batch")
    parser.add_argument("--llm-temperature", type=float, default=1.0,
                        help="Temperature for GPT-4 generation (higher = more diverse)")

    args = parser.parse_args()

    config = Config(
        task=args.task,
        openai_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_z=args.lr_z,
        d_model=args.d_model,
        num_layers=args.num_layers,
        max_len=args.max_len,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
        target_response=args.target_response,
        device=args.device,
        resume_from=args.resume,
        save_every=args.save_every,
        max_api_workers=args.max_api_workers,
        llm_temperature=args.llm_temperature,
    )

    train(config)


if __name__ == "__main__":
    main()
