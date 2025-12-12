"""
Reward Oracle for GFlowNet Training

Computes rewards R(x) = p(x) * p(y|x) using LLM APIs.

The reward represents the joint probability of generating context x
and then having the LLM produce response y given x.

Supports:
- OpenAI API (GPT-4, GPT-4o, etc.)
- Local models via HuggingFace (GPT-2, LLaMA, etc.)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
import aiohttp
from functools import lru_cache
import os
import time


@dataclass
class RewardOutput:
    """Output from reward computation."""

    log_reward: torch.Tensor  # (batch,) log R(x) = log p(x) + log p(y|x)
    log_p_x: torch.Tensor  # (batch,) log p(x) - prior probability
    log_p_y_given_x: torch.Tensor  # (batch,) log p(y|x) - conditional from LLM
    metadata: Dict[str, Any]  # Additional info (tokens used, etc.)


class RewardOracle(ABC):
    """Abstract base class for reward oracles."""

    @abstractmethod
    def compute_reward(
        self,
        sequences: List[str],
        conditioning: Optional[str] = None,
    ) -> RewardOutput:
        """Compute rewards for a batch of sequences."""
        pass

    @abstractmethod
    def compute_log_p_y_given_x(
        self,
        contexts: List[str],
        response: str,
    ) -> torch.Tensor:
        """Compute log p(y|x) for given contexts and fixed response."""
        pass


class OpenAIRewardOracle(RewardOracle):
    """
    Reward oracle using OpenAI API.

    Computes log p(y|x) by:
    1. Sending context x as user message
    2. Getting logprobs for target response y
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 100,
        prior_model: Optional[str] = None,  # For computing p(x)
        device: str = "cpu",
        cache_size: int = 10000,
        rate_limit_delay: float = 0.1,
    ):
        """
        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (defaults to env var)
            temperature: Sampling temperature
            max_tokens: Max tokens in response
            prior_model: Model for p(x) computation (if None, uses uniform)
            device: Torch device
            cache_size: LRU cache size for API calls
            rate_limit_delay: Delay between API calls (seconds)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prior_model = prior_model
        self.device = torch.device(device)
        self.rate_limit_delay = rate_limit_delay

        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

        # Import OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        # Cache for API responses
        self._cache: Dict[str, Any] = {}
        self._cache_size = cache_size

    def _get_completion_with_logprobs(
        self,
        messages: List[Dict[str, str]],
        logprobs: bool = True,
        top_logprobs: int = 5,
    ) -> Dict[str, Any]:
        """Get completion with log probabilities."""
        cache_key = str(messages) + str(logprobs) + str(top_logprobs)

        if cache_key in self._cache:
            return self._cache[cache_key]

        time.sleep(self.rate_limit_delay)  # Rate limiting

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs if logprobs else None,
        )

        result = {
            "content": response.choices[0].message.content,
            "logprobs": response.choices[0].logprobs if logprobs else None,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }

        # Cache result
        if len(self._cache) < self._cache_size:
            self._cache[cache_key] = result

        return result

    def compute_log_p_y_given_x(
        self,
        contexts: List[str],
        response: str,
        system_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute log p(y|x) for each context.

        This asks the model to generate and checks the logprob of
        the actual generated tokens.

        Args:
            contexts: List of context strings (x)
            response: Target response (y) - not used directly, we measure
                     what the model generates
        """
        log_probs = []

        for context in contexts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": context})

            result = self._get_completion_with_logprobs(messages)

            # Sum log probs of generated tokens
            if result["logprobs"] and result["logprobs"].content:
                total_log_prob = sum(
                    token_info.logprob
                    for token_info in result["logprobs"].content
                )
            else:
                total_log_prob = 0.0

            log_probs.append(total_log_prob)

        return torch.tensor(log_probs, device=self.device)

    def compute_reward(
        self,
        sequences: List[str],
        conditioning: Optional[str] = None,
        log_p_x: Optional[torch.Tensor] = None,
    ) -> RewardOutput:
        """
        Compute rewards R(x) = p(x) * p(y|x).

        Args:
            sequences: List of generated contexts (x)
            conditioning: Optional system prompt/task description
            log_p_x: Pre-computed log p(x), if available

        Returns:
            RewardOutput with log rewards and components
        """
        # Compute log p(y|x) for each sequence
        # Here y is the model's response to context x
        log_p_y_given_x = self.compute_log_p_y_given_x(
            sequences,
            response="",  # We measure actual generation
            system_prompt=conditioning,
        )

        # Compute log p(x) - prior probability of context
        if log_p_x is None:
            # Default: uniform prior (log p(x) = 0)
            # In practice, you might want to use a language model
            log_p_x = torch.zeros(len(sequences), device=self.device)

        # Total reward: R(x) = p(x) * p(y|x)
        log_reward = log_p_x + log_p_y_given_x

        return RewardOutput(
            log_reward=log_reward,
            log_p_x=log_p_x,
            log_p_y_given_x=log_p_y_given_x,
            metadata={
                "model": self.model,
                "num_sequences": len(sequences),
            }
        )


class HuggingFaceRewardOracle(RewardOracle):
    """
    Reward oracle using HuggingFace models locally.

    More efficient than API calls, supports batching,
    and provides exact log probabilities.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        batch_size: int = 8,
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: Torch device
            max_length: Maximum sequence length
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size

        # Load model and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def compute_sequence_log_prob(
        self,
        sequences: List[str],
    ) -> torch.Tensor:
        """
        Compute log p(sequence) for each sequence.

        Args:
            sequences: List of text sequences

        Returns:
            log_probs: (batch,) log probabilities
        """
        all_log_probs = []

        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i:i + self.batch_size]

            # Tokenize
            encodings = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            # Per-sequence loss (negative log prob)
            # outputs.loss is mean over all tokens
            # We need per-sequence
            logits = outputs.logits[:, :-1, :]  # (batch, seq-1, vocab)
            targets = input_ids[:, 1:]  # (batch, seq-1)
            mask = attention_mask[:, 1:]  # (batch, seq-1)

            # Compute log probs
            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

            # Sum over sequence (masked)
            seq_log_probs = (target_log_probs * mask).sum(dim=-1)
            all_log_probs.append(seq_log_probs)

        return torch.cat(all_log_probs, dim=0)

    @torch.no_grad()
    def compute_conditional_log_prob(
        self,
        contexts: List[str],
        responses: List[str],
    ) -> torch.Tensor:
        """
        Compute log p(response | context) for each pair.

        Args:
            contexts: List of context strings
            responses: List of response strings

        Returns:
            log_probs: (batch,) conditional log probabilities
        """
        all_log_probs = []

        for i in range(0, len(contexts), self.batch_size):
            batch_contexts = contexts[i:i + self.batch_size]
            batch_responses = responses[i:i + self.batch_size]

            # Combine context + response
            full_texts = [c + r for c, r in zip(batch_contexts, batch_responses)]

            # Tokenize full text
            full_encodings = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            # Tokenize context only to get lengths
            context_encodings = self.tokenizer(
                batch_contexts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            input_ids = full_encodings["input_ids"].to(self.device)
            attention_mask = full_encodings["attention_mask"].to(self.device)
            context_lengths = context_encodings["attention_mask"].sum(dim=-1)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]

            # Compute log probs
            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

            # Mask: only count response tokens (after context)
            batch_size_actual, seq_len = targets.shape
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            response_mask = position_ids >= (context_lengths.unsqueeze(1) - 1)
            response_mask = response_mask & (attention_mask[:, 1:] == 1)

            # Sum log probs over response only
            seq_log_probs = (target_log_probs * response_mask.float()).sum(dim=-1)
            all_log_probs.append(seq_log_probs)

        return torch.cat(all_log_probs, dim=0)

    def compute_log_p_y_given_x(
        self,
        contexts: List[str],
        response: str,
    ) -> torch.Tensor:
        """Compute log p(y|x) for fixed response y and varying contexts x."""
        responses = [response] * len(contexts)
        return self.compute_conditional_log_prob(contexts, responses)

    def compute_reward(
        self,
        sequences: List[str],
        conditioning: Optional[str] = None,
        response: Optional[str] = None,
    ) -> RewardOutput:
        """
        Compute rewards R(x) = p(x) * p(y|x).

        Args:
            sequences: List of generated contexts (x)
            conditioning: Not used for HF models
            response: Target response y (if None, uses empty string)

        Returns:
            RewardOutput with log rewards
        """
        # Compute log p(x) - sequence probability
        log_p_x = self.compute_sequence_log_prob(sequences)

        # Compute log p(y|x) - conditional probability
        if response is None:
            # If no target response, log p(y|x) = 0 (just use p(x))
            log_p_y_given_x = torch.zeros_like(log_p_x)
        else:
            log_p_y_given_x = self.compute_log_p_y_given_x(sequences, response)

        log_reward = log_p_x + log_p_y_given_x

        return RewardOutput(
            log_reward=log_reward,
            log_p_x=log_p_x,
            log_p_y_given_x=log_p_y_given_x,
            metadata={
                "model": self.model_name,
                "num_sequences": len(sequences),
            }
        )


class JointRewardOracle(RewardOracle):
    """
    Reward oracle that combines multiple models.

    Uses one model for p(x) (prior) and another for p(y|x) (likelihood).
    This is the full formulation for dataset resurrection:
    R(x) = p(x) * p(y|x)
    """

    def __init__(
        self,
        prior_oracle: RewardOracle,
        likelihood_oracle: RewardOracle,
        prior_weight: float = 1.0,
        likelihood_weight: float = 1.0,
    ):
        """
        Args:
            prior_oracle: Oracle for computing p(x)
            likelihood_oracle: Oracle for computing p(y|x)
            prior_weight: Weight for log p(x) in reward
            likelihood_weight: Weight for log p(y|x) in reward
        """
        self.prior_oracle = prior_oracle
        self.likelihood_oracle = likelihood_oracle
        self.prior_weight = prior_weight
        self.likelihood_weight = likelihood_weight

    def compute_log_p_y_given_x(
        self,
        contexts: List[str],
        response: str,
    ) -> torch.Tensor:
        """Delegate to likelihood oracle."""
        return self.likelihood_oracle.compute_log_p_y_given_x(contexts, response)

    def compute_reward(
        self,
        sequences: List[str],
        conditioning: Optional[str] = None,
        response: Optional[str] = None,
    ) -> RewardOutput:
        """
        Compute joint reward R(x) = p(x)^alpha * p(y|x)^beta.

        Args:
            sequences: Generated contexts
            conditioning: System prompt for likelihood
            response: Target response for likelihood
        """
        # Get p(x) from prior
        prior_output = self.prior_oracle.compute_reward(sequences)
        log_p_x = prior_output.log_p_x

        # Get p(y|x) from likelihood oracle
        if response is not None:
            log_p_y_given_x = self.likelihood_oracle.compute_log_p_y_given_x(
                sequences, response
            )
        else:
            # Use the likelihood oracle's generation-based reward
            likelihood_output = self.likelihood_oracle.compute_reward(
                sequences, conditioning
            )
            log_p_y_given_x = likelihood_output.log_p_y_given_x

        # Weighted combination
        log_reward = (
            self.prior_weight * log_p_x +
            self.likelihood_weight * log_p_y_given_x
        )

        return RewardOutput(
            log_reward=log_reward,
            log_p_x=log_p_x,
            log_p_y_given_x=log_p_y_given_x,
            metadata={
                "prior_model": getattr(self.prior_oracle, "model_name", "unknown"),
                "likelihood_model": getattr(self.likelihood_oracle, "model", "unknown"),
                "prior_weight": self.prior_weight,
                "likelihood_weight": self.likelihood_weight,
            }
        )


def create_reward_oracle(
    oracle_type: str = "huggingface",
    **kwargs,
) -> RewardOracle:
    """
    Factory function to create reward oracles.

    Args:
        oracle_type: "openai", "huggingface", or "joint"
        **kwargs: Arguments for the specific oracle

    Returns:
        RewardOracle instance
    """
    if oracle_type == "openai":
        return OpenAIRewardOracle(**kwargs)
    elif oracle_type == "huggingface":
        return HuggingFaceRewardOracle(**kwargs)
    elif oracle_type == "joint":
        prior_kwargs = kwargs.pop("prior_kwargs", {})
        likelihood_kwargs = kwargs.pop("likelihood_kwargs", {})
        prior_type = kwargs.pop("prior_type", "huggingface")
        likelihood_type = kwargs.pop("likelihood_type", "openai")

        prior_oracle = create_reward_oracle(prior_type, **prior_kwargs)
        likelihood_oracle = create_reward_oracle(likelihood_type, **likelihood_kwargs)

        return JointRewardOracle(
            prior_oracle=prior_oracle,
            likelihood_oracle=likelihood_oracle,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown oracle type: {oracle_type}")
