"""
Main Training Script for Dataset Sucker

GFlowNet training with trajectory balance to learn the joint distribution
p(x, y) = p(x) * p(y|x) where p(y|x) comes from an LLM oracle.

Usage:
    python train.py --oracle huggingface --model gpt2 --epochs 1000
    python train.py --oracle openai --model gpt-4o-mini --epochs 500
"""

import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Optional, Dict, Any
import os
from datetime import datetime
from tqdm import tqdm

# Local imports
from gflownet_policy import TransformerGFlowNetPolicy
from trajectory_balance import TrajectoryBalanceLoss
from reward_oracle import (
    create_reward_oracle,
    HuggingFaceRewardOracle,
    OpenAIRewardOracle,
)
from evaluation import (
    GFlowNetEvaluator,
    setup_wandb,
    log_samples_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GFlowNet for dataset sucking")

    # Oracle settings
    parser.add_argument("--oracle", type=str, default="huggingface",
                        choices=["huggingface", "openai", "joint"],
                        help="Type of reward oracle")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model for reward computation (gpt2, gpt-4o-mini, etc.)")
    parser.add_argument("--prior-model", type=str, default=None,
                        help="Model for p(x) prior (default: same as --model)")

    # GFlowNet architecture
    parser.add_argument("--d-model", type=int, default=256,
                        help="Transformer model dimension")
    parser.add_argument("--nhead", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--max-len", type=int, default=64,
                        help="Maximum sequence length")

    # Training settings
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--lr-z", type=float, default=1e-2,
                        help="Learning rate for log Z")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Warmup steps for scheduler")

    # Sampling settings
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling threshold")

    # Loss settings
    parser.add_argument("--reward-temp", type=float, default=1.0,
                        help="Temperature for reward scaling")
    parser.add_argument("--use-log-var-loss", action="store_true",
                        help="Use log-variance loss for stability")

    # Evaluation settings
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Evaluate every N epochs")
    parser.add_argument("--num-eval-samples", type=int, default=64,
                        help="Number of samples for evaluation")

    # Logging
    parser.add_argument("--project", type=str, default="dataset_sucker",
                        help="W&B project name")
    parser.add_argument("--run-name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda, mps)")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save checkpoint every N epochs")

    # Conditioning (optional)
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="System prompt for conditional generation")
    parser.add_argument("--target-response", type=str, default=None,
                        help="Target response for conditional reward")

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get the appropriate device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def create_tokenizer(model_name: str):
    """Create tokenizer based on model name."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def train_step(
    gfn_policy: TransformerGFlowNetPolicy,
    tb_loss_fn: TrajectoryBalanceLoss,
    optimizer: optim.Optimizer,
    z_optimizer: optim.Optimizer,
    reward_oracle,
    tokenizer,
    batch_size: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    grad_clip: float = 1.0,
    conditioning: Optional[str] = None,
    target_response: Optional[str] = None,
) -> Dict[str, float]:
    """
    Perform a single training step.

    Returns:
        Dictionary of metrics for logging
    """
    gfn_policy.train()
    device = next(gfn_policy.parameters()).device

    # Sample trajectories from GFN policy
    sequences, log_pf, lengths = gfn_policy.sample_trajectory(
        batch_size=batch_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Decode sequences to text
    texts = []
    for seq in sequences:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        texts.append(text.strip())

    # Compute rewards
    reward_output = reward_oracle.compute_reward(
        texts,
        conditioning=conditioning,
        response=target_response,
    )
    log_reward = reward_output.log_reward.to(device)

    # Compute trajectory balance loss
    tb_output = tb_loss_fn(
        log_Z=gfn_policy.ln_Z,
        log_pf=log_pf,
        log_reward=log_reward,
    )

    # Backward pass
    optimizer.zero_grad()
    z_optimizer.zero_grad()

    tb_output.loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(gfn_policy.parameters(), grad_clip)

    # Update parameters
    optimizer.step()
    z_optimizer.step()

    return {
        "tb_loss": tb_output.loss.item(),
        "log_Z": gfn_policy.ln_Z.item(),
        "log_pf_mean": log_pf.mean().item(),
        "log_pf_std": log_pf.std().item(),
        "log_reward_mean": log_reward.mean().item(),
        "log_reward_std": log_reward.std().item(),
        "tb_error_mean": tb_output.tb_error.mean().item(),
        "tb_error_std": tb_output.tb_error.std().item(),
        "log_p_x_mean": reward_output.log_p_x.mean().item(),
        "log_p_y_given_x_mean": reward_output.log_p_y_given_x.mean().item(),
    }


def save_checkpoint(
    gfn_policy: TransformerGFlowNetPolicy,
    optimizer: optim.Optimizer,
    z_optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_dir: str,
):
    """Save training checkpoint."""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": gfn_policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "z_optimizer_state_dict": z_optimizer.state_dict(),
        "metrics": metrics,
        "ln_Z": gfn_policy.ln_Z.item(),
    }

    path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def main():
    args = parse_args()

    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Setup wandb
    if not args.no_wandb:
        run_name = args.run_name or f"gfn_{args.oracle}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        setup_wandb(
            project_name=args.project,
            run_name=run_name,
            config=vars(args),
            tags=[args.oracle, args.model],
        )

    # Create tokenizer
    if args.oracle == "openai":
        # For OpenAI, use GPT-2 tokenizer as proxy
        tokenizer = create_tokenizer("gpt2")
    else:
        tokenizer = create_tokenizer(args.model)

    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id

    print(f"Vocabulary size: {vocab_size}")

    # Create reward oracle
    print(f"Creating {args.oracle} reward oracle with model {args.model}...")
    if args.oracle == "openai":
        reward_oracle = create_reward_oracle(
            oracle_type="openai",
            model=args.model,
        )
    elif args.oracle == "huggingface":
        reward_oracle = create_reward_oracle(
            oracle_type="huggingface",
            model_name=args.model,
            device=str(device),
        )
    elif args.oracle == "joint":
        prior_model = args.prior_model or args.model
        reward_oracle = create_reward_oracle(
            oracle_type="joint",
            prior_type="huggingface",
            prior_kwargs={"model_name": prior_model, "device": str(device)},
            likelihood_type="openai" if "gpt" in args.model.lower() else "huggingface",
            likelihood_kwargs={"model": args.model} if "gpt" in args.model.lower() else {"model_name": args.model},
        )

    # Create GFlowNet policy
    print("Creating GFlowNet policy...")
    gfn_policy = TransformerGFlowNetPolicy(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_len=args.max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    ).to(device)

    num_params = sum(p.numel() for p in gfn_policy.parameters())
    print(f"GFlowNet parameters: {num_params:,}")

    # Create optimizers
    # Separate optimizer for log Z with higher learning rate
    policy_params = [p for n, p in gfn_policy.named_parameters() if n != "ln_Z"]
    z_params = [gfn_policy.ln_Z]

    optimizer = optim.AdamW(
        policy_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    z_optimizer = optim.Adam(z_params, lr=args.lr_z)

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)

    # Loss function
    tb_loss_fn = TrajectoryBalanceLoss(
        reward_temperature=args.reward_temp,
        use_log_variance_loss=args.use_log_var_loss,
    )

    # Evaluator
    evaluator = GFlowNetEvaluator(
        tokenizer=tokenizer,
        log_to_wandb=not args.no_wandb,
        num_samples_for_kl=args.num_eval_samples,
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    global_step = 0

    for epoch in tqdm(range(args.epochs), desc="Training"):
        # Training step
        metrics = train_step(
            gfn_policy=gfn_policy,
            tb_loss_fn=tb_loss_fn,
            optimizer=optimizer,
            z_optimizer=z_optimizer,
            reward_oracle=reward_oracle,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            grad_clip=args.grad_clip,
            conditioning=args.system_prompt,
            target_response=args.target_response,
        )

        global_step += 1

        # Log training metrics
        evaluator.log_training_step(
            tb_loss=metrics["tb_loss"],
            log_Z=metrics["log_Z"],
            log_pf_mean=metrics["log_pf_mean"],
            log_reward_mean=metrics["log_reward_mean"],
            step=global_step,
            additional=metrics,
        )

        # Step scheduler
        scheduler.step()

        # Evaluation
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            print(f"\n[Epoch {epoch}] TB Loss: {metrics['tb_loss']:.4f}, "
                  f"log Z: {metrics['log_Z']:.4f}, "
                  f"log R mean: {metrics['log_reward_mean']:.4f}")

            eval_metrics = evaluator.evaluate(
                gfn_policy=gfn_policy,
                target_oracle=reward_oracle,
                epoch=epoch,
                step=global_step,
            )

            print(f"  KL(fwd): {eval_metrics.kl_forward:.4f}, "
                  f"KL(rev): {eval_metrics.kl_reverse:.4f}, "
                  f"Unique: {eval_metrics.unique_ratio:.2%}")

            # Sample and display some generations
            with torch.no_grad():
                samples, _, _ = gfn_policy.sample_trajectory(batch_size=5)
                sample_texts = [
                    tokenizer.decode(s, skip_special_tokens=True).strip()
                    for s in samples
                ]
                print("  Sample generations:")
                for i, text in enumerate(sample_texts[:3]):
                    print(f"    {i+1}. {text[:100]}...")

        # Save checkpoint
        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                gfn_policy, optimizer, z_optimizer,
                epoch, metrics, args.save_dir
            )

    # Final evaluation
    print("\n" + "="*50)
    print("Training complete! Running final evaluation...")

    final_metrics = evaluator.evaluate(
        gfn_policy=gfn_policy,
        target_oracle=reward_oracle,
        epoch=args.epochs,
        step=global_step,
    )

    print(f"\nFinal Results:")
    print(f"  TB Loss: {final_metrics.tb_loss:.4f}")
    print(f"  KL(target || gfn): {final_metrics.kl_forward:.4f}")
    print(f"  KL(gfn || target): {final_metrics.kl_reverse:.4f}")
    print(f"  Learned log Z: {final_metrics.log_Z:.4f}")
    print(f"  Estimated log Z: {final_metrics.log_Z_estimate:.4f}")
    print(f"  Sample diversity: {final_metrics.unique_ratio:.2%}")

    # Save final checkpoint
    save_checkpoint(
        gfn_policy, optimizer, z_optimizer,
        args.epochs, final_metrics.to_dict(), args.save_dir
    )

    # Plot convergence
    evaluator.plot_convergence()

    if not args.no_wandb:
        import wandb
        wandb.finish()

    print("\nDone! Dataset successfully sucked. 🧛")


if __name__ == "__main__":
    main()
