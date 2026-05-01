# ABOUTME: Token-level edit distance calculation
# ABOUTME: Computes Levenshtein distance on tokenized sequences for perturbation matching


def levenshtein_distance(seq1, seq2):
    """
    Calculate Levenshtein edit distance between two sequences.

    Args:
        seq1: First sequence (list of token IDs)
        seq2: Second sequence (list of token IDs)

    Returns:
        int: Minimum number of edits (insert, delete, substitute) to transform seq1 to seq2
    """
    m, n = len(seq1), len(seq2)

    # Create distance matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )

    return dp[m][n]


def token_edit_distance_percent(original, perturbed, tokenizer):
    """
    Calculate percentage of tokens changed using Levenshtein edit distance.

    Args:
        original: Original text string
        perturbed: Perturbed text string
        tokenizer: HuggingFace tokenizer (same as model being analyzed)

    Returns:
        float: Percentage of tokens changed (edit_distance / original_length * 100)
    """
    orig_tokens = tokenizer.encode(original, add_special_tokens=False)
    pert_tokens = tokenizer.encode(perturbed, add_special_tokens=False)

    if len(orig_tokens) == 0:
        return 0.0 if len(pert_tokens) == 0 else 100.0

    edit_dist = levenshtein_distance(orig_tokens, pert_tokens)

    return (edit_dist / len(orig_tokens)) * 100
