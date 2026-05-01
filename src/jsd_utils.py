#!/usr/bin/env python3
"""
ABOUTME: JSD (Jensen-Shannon Divergence) calculation utilities
ABOUTME: Provides probability distribution analysis for model outputs
"""

import numpy as np
from scipy.spatial.distance import jensenshannon
import torch

def extract_abcd_logits(logits_tensor, token_ids_dict):
    """
    Extract logits for answer tokens A, B, C, D from full vocabulary logits

    Args:
        logits_tensor: torch.Tensor of shape [vocab_size]
        token_ids_dict: dict mapping 'A', 'B', 'C', 'D' to token IDs

    Returns:
        dict: {'A': float, 'B': float, 'C': float, 'D': float}
    """
    return {
        option: logits_tensor[token_id].item()
        for option, token_id in token_ids_dict.items()
    }

def logits_to_probs(logits_dict):
    """
    Convert logits dictionary to probability distribution via softmax

    Args:
        logits_dict: dict with keys 'A', 'B', 'C', 'D' and logit values

    Returns:
        dict: probability distribution over A, B, C, D (sums to 1.0)
    """
    logits = np.array([logits_dict[k] for k in ['A', 'B', 'C', 'D']])
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    probs = exp_logits / exp_logits.sum()
    return {k: probs[i] for i, k in enumerate(['A', 'B', 'C', 'D'])}

def calculate_jsd(probs_1, probs_2):
    """
    Calculate Jensen-Shannon Divergence between two probability distributions

    Args:
        probs_1: dict {'A': p1, 'B': p2, 'C': p3, 'D': p4}
        probs_2: dict {'A': q1, 'B': q2, 'C': q3, 'D': q4}

    Returns:
        float: JSD value (0 = identical, higher = more different)
    """
    p = np.array([probs_1[k] for k in ['A', 'B', 'C', 'D']])
    q = np.array([probs_2[k] for k in ['A', 'B', 'C', 'D']])

    # scipy's jensenshannon returns sqrt of JSD, so square it
    return jensenshannon(p, q) ** 2

def calculate_aggregate_stats(jsd_list):
    """
    Calculate aggregate statistics for JSD values

    Args:
        jsd_list: list of JSD values

    Returns:
        dict: {'median': float, '95th_percentile': float, 'mean': float}
    """
    jsd_array = np.array(jsd_list)
    return {
        'median': np.median(jsd_array),
        '95th_percentile': np.percentile(jsd_array, 95),
        'mean': np.mean(jsd_array)
    }
