"""
GFlowNet Policy Network for Dataset Sucking

Implements a Transformer-based policy with learnable ln_Z parameter
for trajectory balance training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerGFlowNetPolicy(nn.Module):
    """
    Transformer-based GFlowNet policy for sequential text generation.

    The policy generates sequences token-by-token, with a learnable ln_Z
    parameter that represents the log partition function.

    For trajectory balance: log Z + log P_F(tau) = log R(x)
    Where:
        - Z is the partition function (exp(ln_Z))
        - P_F(tau) is the forward policy probability of trajectory tau
        - R(x) = p(x) * p(y|x) is the reward (joint probability)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Transformer decoder (causal)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Learnable log partition function
        # Initialize to 0 (Z=1), will be learned via trajectory balance
        self.ln_Z = nn.Parameter(torch.tensor(0.0))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to get logits for all positions.

        Args:
            input_ids: (batch, seq_len) token ids
            attention_mask: (batch, seq_len) padding mask (1 = valid, 0 = pad)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embed tokens
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)

        # Padding mask (for transformer: True = ignore)
        if attention_mask is not None:
            # Convert from (1=valid, 0=pad) to (False=valid, True=ignore)
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # Transformer forward (self-attention only, no cross-attention)
        # Use a dummy memory tensor
        memory = torch.zeros(batch_size, 1, self.d_model, device=device)
        x = self.transformer(
            x,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask,
        )

        # Project to vocabulary
        logits = self.output_proj(x)

        return logits

    def get_next_token_logits(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get logits for the next token only (for efficient sampling).

        Args:
            input_ids: (batch, seq_len) token ids

        Returns:
            logits: (batch, vocab_size) logits for next token
        """
        logits = self.forward(input_ids)
        return logits[:, -1, :]

    @torch.no_grad()
    def sample_trajectory(
        self,
        batch_size: int = 1,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample trajectories from the policy.

        Args:
            batch_size: Number of trajectories to sample
            max_len: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold

        Returns:
            sequences: (batch, seq_len) sampled token ids
            log_pf: (batch,) log forward probability of each trajectory
            lengths: (batch,) actual lengths of each sequence
        """
        if max_len is None:
            max_len = self.max_len

        device = next(self.parameters()).device

        # Start with BOS token
        sequences = torch.full(
            (batch_size, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=device,
        )

        log_pf = torch.zeros(batch_size, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        lengths = torch.ones(batch_size, dtype=torch.long, device=device)

        for step in range(max_len - 1):
            # Get logits for next token
            logits = self.get_next_token_logits(sequences)

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                logits = torch.where(logits < threshold, float('-inf'), logits)

            # Apply nucleus (top-p) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            # Compute log probability of sampled tokens
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(1, next_tokens).squeeze(-1)

            # Update log_pf only for non-finished sequences
            log_pf = log_pf + token_log_probs * (~finished).float()

            # Update lengths for non-finished sequences
            lengths = lengths + (~finished).long()

            # Check for EOS
            finished = finished | (next_tokens.squeeze(-1) == self.eos_token_id)

            # Append tokens
            sequences = torch.cat([sequences, next_tokens], dim=1)

            # Stop if all sequences finished
            if finished.all():
                break

        return sequences, log_pf, lengths

    def compute_log_pf(
        self,
        sequences: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log forward probability of given sequences.

        Args:
            sequences: (batch, seq_len) token ids (including BOS)
            attention_mask: (batch, seq_len) mask for valid tokens

        Returns:
            log_pf: (batch,) log probability of each sequence
        """
        batch_size, seq_len = sequences.shape

        if seq_len < 2:
            return torch.zeros(batch_size, device=sequences.device)

        # Get logits for all positions
        logits = self.forward(sequences[:, :-1], attention_mask[:, :-1] if attention_mask is not None else None)

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs of actual next tokens
        # targets: (batch, seq_len-1)
        targets = sequences[:, 1:]
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

        # Mask out padding
        if attention_mask is not None:
            # Shift mask to align with targets
            target_mask = attention_mask[:, 1:].float()
            token_log_probs = token_log_probs * target_mask

        # Sum log probs
        log_pf = token_log_probs.sum(dim=-1)

        return log_pf


class GFlowNetWithConditioning(TransformerGFlowNetPolicy):
    """
    GFlowNet policy that can be conditioned on a context/prompt.

    This is useful for conditional generation where we want to sample
    x such that R(x) = p(x) * p(y|x) for some conditioning y.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        num_encoder_layers: int = 2,
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

        # Encoder for conditioning context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Cross-attention in decoder
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

    def encode_context(
        self,
        context_ids: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode conditioning context.

        Args:
            context_ids: (batch, context_len) token ids
            context_mask: (batch, context_len) attention mask

        Returns:
            context_encoding: (batch, context_len, d_model)
        """
        x = self.embedding(context_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        if context_mask is not None:
            src_key_padding_mask = (context_mask == 0)
        else:
            src_key_padding_mask = None

        context_encoding = self.context_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return context_encoding

    def forward_with_context(
        self,
        input_ids: torch.Tensor,
        context_encoding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with conditioning context.

        Args:
            input_ids: (batch, seq_len) token ids
            context_encoding: (batch, context_len, d_model) encoded context
            attention_mask: (batch, seq_len) padding mask
            context_mask: (batch, context_len) context padding mask

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embed tokens
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)

        # Key padding mask
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # Context key padding mask
        if context_mask is not None:
            context_key_padding_mask = (context_mask == 0)
        else:
            context_key_padding_mask = None

        # Transformer with cross-attention to context
        x = self.transformer(
            x,
            context_encoding,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=context_key_padding_mask,
        )

        # Additional cross-attention
        attn_out, _ = self.cross_attention(
            x, context_encoding, context_encoding,
            key_padding_mask=context_key_padding_mask,
        )
        x = self.cross_norm(x + attn_out)

        # Project to vocabulary
        logits = self.output_proj(x)

        return logits
