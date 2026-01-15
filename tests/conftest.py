"""Shared fixtures for quantization lab tests."""

from __future__ import annotations

import pytest
import torch

from model_quantization_lab.core import BenchmarkModel, ModelQuantizer
from model_quantization_lab.models import (
    BenchmarkConfig,
    QuantizationConfig,
    QuantizationMethod,
)

VOCAB_SIZE = 100
HIDDEN_DIM = 64
NUM_LAYERS = 2
SEQ_LENGTH = 32
BATCH_SIZE = 4


@pytest.fixture
def benchmark_config() -> BenchmarkConfig:
    """Small benchmark config for testing."""
    return BenchmarkConfig(
        model_dim=HIDDEN_DIM,
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        seq_length=SEQ_LENGTH,
        eval_samples=8,
        warmup_iterations=1,
        benchmark_iterations=3,
        seed=42,
    )


@pytest.fixture
def small_model() -> BenchmarkModel:
    """Small model for testing."""
    torch.manual_seed(42)
    return BenchmarkModel(
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    )


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Sample input tensor."""
    torch.manual_seed(42)
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))


@pytest.fixture
def dynamic_config() -> QuantizationConfig:
    """Dynamic 8-bit quantization config."""
    return QuantizationConfig(
        method=QuantizationMethod.DYNAMIC,
        num_bits=8,
    )


@pytest.fixture
def static_config() -> QuantizationConfig:
    """Static 4-bit quantization config."""
    return QuantizationConfig(
        method=QuantizationMethod.STATIC,
        num_bits=4,
    )
