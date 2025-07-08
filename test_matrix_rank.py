#!/usr/bin/env python3
"""
Test script to compare min_fp approach with torch.linalg.matrix_rank
"""

import torch
import numpy as np


def test_rank_computation():
    """Compare min_fp vs matrix_rank approaches"""

    # Create a test matrix with known rank
    m, n = 100, 100
    rank_true = 50

    # Create a matrix with specific rank
    U = torch.randn(m, rank_true)
    V = torch.randn(rank_true, n)
    A = U @ V  # This matrix has rank = rank_true

    # Add some noise to make it more realistic
    noise = torch.randn(m, n) * 1e-10
    A_noisy = A + noise

    print(f"True rank: {rank_true}")
    print(f"Matrix shape: {A_noisy.shape}")

    # Method 1: min_fp approach (old method)
    min_fp = torch.finfo(torch.float32).tiny
    U_svd, S_svd, V_svd = torch.linalg.svd(A_noisy)
    non_zero_count = (S_svd > min_fp).sum().item()
    print(f"min_fp approach: {non_zero_count} non-zero singular values")

    # Method 2: torch.linalg.matrix_rank (new method)
    matrix_rank = torch.linalg.matrix_rank(A_noisy).item()
    print(f"torch.linalg.matrix_rank: {matrix_rank}")

    # Method 3: torch.linalg.matrix_rank with custom tolerances
    matrix_rank_atol = torch.linalg.matrix_rank(A_noisy, atol=1e-12).item()
    print(f"torch.linalg.matrix_rank (atol=1e-12): {matrix_rank_atol}")

    matrix_rank_rtol = torch.linalg.matrix_rank(A_noisy, rtol=1e-10).item()
    print(f"torch.linalg.matrix_rank (rtol=1e-10): {matrix_rank_rtol}")

    # Show singular values for comparison
    print(f"\nFirst 10 singular values: {S_svd[:10]}")
    print(f"min_fp threshold: {min_fp}")

    return {
        "true_rank": rank_true,
        "min_fp_count": non_zero_count,
        "matrix_rank": matrix_rank,
        "matrix_rank_atol": matrix_rank_atol,
        "matrix_rank_rtol": matrix_rank_rtol,
    }


if __name__ == "__main__":
    results = test_rank_computation()
    print(f"\nSummary:")
    print(f"True rank: {results['true_rank']}")
    print(f"min_fp count: {results['min_fp_count']}")
    print(f"matrix_rank: {results['matrix_rank']}")
    print(f"matrix_rank (atol=1e-12): {results['matrix_rank_atol']}")
    print(f"matrix_rank (rtol=1e-10): {results['matrix_rank_rtol']}")
