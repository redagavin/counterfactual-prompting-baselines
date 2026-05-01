#!/usr/bin/env python3
"""
ABOUTME: Custom stopping criteria for HuggingFace transformers
ABOUTME: Detects </think> tag to stop reasoning model generation
"""

from transformers import StoppingCriteria
import torch

class ThinkTagStoppingCriteria(StoppingCriteria):
    """
    Stops generation when </think> tag appears in decoded output

    This is specifically for DeepSeek R1 reasoning models that wrap
    their chain-of-thought in <think>...</think> tags.
    """

    def __init__(self, tokenizer, stop_string="</think>"):
        """
        Args:
            tokenizer: HuggingFace tokenizer for decoding
            stop_string: String to detect (default: "</think>")
        """
        self.tokenizer = tokenizer
        self.stop_string = stop_string

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if stop string appears in generated text

        Args:
            input_ids: Tensor of shape [batch_size, seq_len]
            scores: Logits (not used, but required by interface)

        Returns:
            bool: True if should stop, False otherwise
        """
        # Decode only the newly generated tokens (not the prompt)
        # input_ids[0] is the first (and only) sequence in batch
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)

        return self.stop_string in decoded
