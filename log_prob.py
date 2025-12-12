import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import wandb
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, TensorDataset  # Not used, but for completeness

# Load pre-trained GPT2 as the "oracle" LM (no training needed)
print("Loading GPT2...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
device = torch.device('cpu')  # Use 'cuda' if available
lm_model.to(device)
vocab_size = tokenizer.vocab_size
max_len = 20  # Max tokens (e.g., ~100 chars)
bos_token_id = tokenizer.bos_token_id
pad_token_id = tokenizer.pad_token_id
print(f"Vocab size: {vocab_size}, Max len: {max_len}")

# Function to compute exact log p(seq) under LM: sum log p(token_t | <t)
def lm_log_prob(lm_model, input_ids):
    if len(input_ids) < 2:
        return torch.tensor(0.0, device=device)
    input_ids = torch.tensor([input_ids], device=device, dtype=torch.long)
    with torch.no_grad():
        outputs = lm_model(input_ids, labels=input_ids)
        # loss is -mean(log p) over positions 1:, so sum_logp = -loss * (len-1)
        num_valid = input_ids.numel() - 1  # Exclude first (bos)
        sum_logp = -outputs.loss * num_valid
    return sum_logp

# Function to decode tokens to text (strip bos/pad)
def decode_tokens(token_ids):
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text.strip()

# Function to sample from LM (autoregressive)
def sample_from_lm(lm_model, tokenizer, num_samples=1, max_new_tokens=19, temperature=1.0):  # max_len-1
    samples = []
    input_ids = torch.tensor([[bos_token_id]], device=device)
    with torch.no_grad():
        for _ in range(num_samples):
            output = lm_model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, pad_token_id=pad_token_id, eos_token_id=tokenizer.eos_token_id
            )
            # Take full seq including bos
            full_seq = output[0].tolist()  # First sample
            samples.append(full_seq[:max_len])  # Truncate to max_len
    return samples

# GFN Policy: Token-level AR policy using GRU
class GFNPolicy(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=1)
        self.policy_head = nn.Linear(hidden_dim, vocab_size)
        self.ln_z0 = nn.Parameter(torch.tensor(math.log(1.0)))  # log Z(s0)

    def sample_trajectory(self, max_len):
        prefix = [bos_token_id]
        log_pf_sum = torch.tensor(0.0, device=device)
        hidden = torch.zeros(1, 1, self.gru.hidden_size, device=device)

        # Embed bos
        emb_bos = self.embed(torch.tensor([[bos_token_id]], dtype=torch.long, device=device))
        _, hidden = self.gru(emb_bos)

        for _ in range(max_len - 1):
            h = hidden[0, 0]
            logits = self.policy_head(h)
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1).item()
            log_pf_sum += F.log_softmax(logits, dim=-1)[token]
            prefix.append(token)

            # Update hidden
            emb_token = self.embed(torch.tensor([[token]], dtype=torch.long, device=device))
            _, hidden = self.gru(emb_token, hidden)

        return prefix, log_pf_sum

    def log_policy_prob(self, tokens):
        if len(tokens) < 2:
            return torch.tensor(0.0, device=device)
        log_p = torch.tensor(0.0, device=device)
        hidden = torch.zeros(1, 1, self.gru.hidden_size, device=device)
        # Start with bos
        emb_bos = self.embed(torch.tensor([[bos_token_id]], dtype=torch.long, device=device))
        _, hidden = self.gru(emb_bos)
        for token in tokens[1:]:  # After bos
            h = hidden[0, 0]
            logits = self.policy_head(h)
            log_p += F.log_softmax(logits, dim=-1)[token]
            emb_token = self.embed(torch.tensor([[token]], dtype=torch.long, device=device))
            _, hidden = self.gru(emb_token, hidden)
        return log_p

# Init GFN and train with TB (R = p_LM(seq) ~ exp(lm_log_prob))
gfn = GFNPolicy(vocab_size).to(device)
gfn_opt = optim.Adam(gfn.parameters(), lr=5e-4)  # Smaller lr for large vocab

# WandB init
wandb.init(project="dataset_sucker", name="gfn-gpt2-tokenlevel-no-data")

batch_size = 8  # Smaller for CPU/memory
gfn_epochs = 500  # More epochs to see convergence
num_val = 16
for epoch in range(gfn_epochs):
    log_terms = []
    for _ in range(batch_size):
        prefix, log_pf = gfn.sample_trajectory(max_len)
        log_r = lm_log_prob(lm_model, prefix)
        log_term = gfn.ln_z0 + log_pf - log_r
        log_terms.append(log_term)
    tb_loss = torch.stack(log_terms).pow(2).mean()
    gfn_opt.zero_grad()
    tb_loss.backward()
    gfn_opt.step()

    wandb.log({"epoch": epoch, "tb_loss": tb_loss.item(), "ln_z0": gfn.ln_z0.item()})

    if epoch % 50 == 0:  # Less frequent eval for speed
        # Evaluation: est KL(p_LM || p_GFN) ≈ E_{~p_LM} [log p_LM - log p_GFN]
        # Note: At conv, log p_GFN ≈ log_pf (ln_z0 ≈ 0 for normalized)
        samples_lm = sample_from_lm(lm_model, tokenizer, num_val)
        kl = 0.0
        for samp in samples_lm:
            log_p_lm = lm_log_prob(lm_model, samp)
            log_pf_gfn = gfn.log_policy_prob(samp)
            # p_GFN(seq) ∝ exp(log_pf), but since Z learned via ln_z0, log p_GFN = log_pf - ln_z0
            # But for KL(p||q) = E_p [log p - log q], and if q normalizes to p, KL→0
            log_p_gfn = log_pf_gfn - gfn.ln_z0
            kl += (log_p_lm - log_p_gfn).item()
        kl /= num_val
        wandb.log({"kl_div": kl})

        # Sample from GFN and log as text
        gfn_samples = [gfn.sample_trajectory(max_len)[0] for _ in range(5)]
        gfn_texts = [decode_tokens(s) for s in gfn_samples]
        wandb.log({"generated_samples": wandb.Table(data=[[text] for text in gfn_texts], columns=["GFN Sample"])})

        # LM samples for comparison
        lm_samples = sample_from_lm(lm_model, tokenizer, 5)
        lm_texts = [decode_tokens(s) for s in lm_samples]
        wandb.log({"lm_samples": wandb.Table(data=[[text] for text in lm_texts], columns=["LM Sample"])})

        print(f"Epoch {epoch}: TB Loss {tb_loss:.4f}, KL {kl:.4f}, ln_z0 {gfn.ln_z0.item():.4f}")
        print("GFN samples:", gfn_texts[:2])
        print("LM samples:", lm_texts[:2])

wandb.finish()