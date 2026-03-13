"""Utility functions for quantization benchmarking.

Provides model size estimation, quantization simulation, metric computation,
and formatting utilities.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def estimate_model_size_mb(model: nn.Module) -> float:
    """Estimate model size in megabytes from parameter dtypes.

    Args:
        model: PyTorch model to measure.

    Returns:
        Estimated size in megabytes.
    """
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    return total_bytes / (1024 * 1024)


def count_parameters(model: nn.Module) -> int:
    """Count total number of model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Total parameter count.
    """
    return sum(p.numel() for p in model.parameters())


def estimate_quantized_size_mb(
    model: nn.Module,
    num_bits: int,
    group_size: int = 128,
) -> float:
    """Estimate quantized model size accounting for scale/zero-point overhead.

    For group quantization, each group of `group_size` weights shares
    one scale and one zero-point value (both float16).

    Args:
        model: Original model.
        num_bits: Target bit width.
        group_size: Group size for quantization.

    Returns:
        Estimated quantized size in megabytes.
    """
    total_bits = 0
    for param in model.parameters():
        num_elements = param.nelement()
        # Weight bits
        total_bits += num_elements * num_bits
        # Scale + zero-point overhead (16 bits each per group)
        num_groups = max(1, num_elements // group_size)
        total_bits += num_groups * 32  # 16-bit scale + 16-bit zero
    return total_bits / 8 / (1024 * 1024)


def compute_mse(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
) -> float:
    """Compute mean squared error between two tensors.

    Args:
        tensor_a: First tensor.
        tensor_b: Second tensor.

    Returns:
        Scalar MSE value.
    """
    return float(torch.mean((tensor_a - tensor_b) ** 2).item())


def compute_cosine_similarity(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
) -> float:
    """Compute cosine similarity between flattened tensors.

    Args:
        tensor_a: First tensor.
        tensor_b: Second tensor.

    Returns:
        Cosine similarity in [-1, 1].
    """
    flat_a = tensor_a.flatten().float()
    flat_b = tensor_b.flatten().float()
    dot = torch.dot(flat_a, flat_b)
    norm_a = torch.norm(flat_a)
    norm_b = torch.norm(flat_b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float((dot / (norm_a * norm_b)).item())


def compute_snr_db(
    original: torch.Tensor,
    quantized: torch.Tensor,
) -> float:
    """Compute signal-to-noise ratio in decibels.

    SNR = 10 * log10(signal_power / noise_power)

    Args:
        original: Original tensor (signal).
        quantized: Quantized tensor (signal + noise).

    Returns:
        SNR in decibels. Higher is better.
    """
    signal_power = float(torch.mean(original.float() ** 2).item())
    noise = original.float() - quantized.float()
    noise_power = float(torch.mean(noise ** 2).item())
    if noise_power < 1e-10:
        return 100.0  # Effectively perfect
    return float(10 * np.log10(signal_power / noise_power))


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute perplexity from logits and target token IDs.

    Perplexity = exp(cross_entropy_loss)

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab).
        targets: Target token IDs of shape (batch, seq_len).

    Returns:
        Perplexity value (lower is better).
    """
    # Flatten for cross-entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    return float(torch.exp(loss).item())


def time_function(
    func: Callable[..., Any],
    *args: Any,
    warmup: int = 3,
    iterations: int = 10,
    **kwargs: Any,
) -> dict[str, float]:
    """Time a function with warmup and multiple iterations.

    Args:
        func: Function to time.
        *args: Positional arguments to pass.
        warmup: Number of warmup calls.
        iterations: Number of timed calls.
        **kwargs: Keyword arguments to pass.

    Returns:
        Dictionary with mean, std, p50, p99 latencies in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Benchmark
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)

    times_arr = np.array(times_ms)
    return {
        "mean_ms": float(np.mean(times_arr)),
        "std_ms": float(np.std(times_arr)),
        "p50_ms": float(np.percentile(times_arr, 50)),
        "p99_ms": float(np.percentile(times_arr, 99)),
        "min_ms": float(np.min(times_arr)),
        "max_ms": float(np.max(times_arr)),
    }


def format_report_table(reports: list[dict[str, Any]]) -> str:
    """Format benchmark reports as an ASCII table.

    Args:
        reports: List of summary row dictionaries.

    Returns:
        Formatted table string.
    """
    if not reports:
        return "No results to display."

    headers = list(reports[0].keys())
    col_widths = {h: max(len(h), 10) for h in headers}

    # Compute column widths
    for row in reports:
        for h in headers:
            val_str = _format_value(row[h])
            col_widths[h] = max(col_widths[h], len(val_str))

    # Header line
    header_line = " | ".join(h.rjust(col_widths[h]) for h in headers)
    separator = "-+-".join("-" * col_widths[h] for h in headers)

    # Data lines
    lines = [header_line, separator]
    for row in reports:
        line = " | ".join(
            _format_value(row[h]).rjust(col_widths[h]) for h in headers
        )
        lines.append(line)

    return "\n".join(lines)


def _format_value(value: Any) -> str:
    """Format a single value for table display.

    Args:
        value: Value to format.

    Returns:
        Formatted string.
    """
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
