"""Data models for quantization benchmarking configuration and results.

Defines the configuration for quantization methods, benchmark parameters,
and structured result types used throughout the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_NUM_BITS = 4
DEFAULT_GROUP_SIZE = 128
DEFAULT_CALIBRATION_SAMPLES = 128
DEFAULT_EVAL_SAMPLES = 256
DEFAULT_SEQ_LENGTH = 512
DEFAULT_WARMUP_ITERATIONS = 3
DEFAULT_BENCHMARK_ITERATIONS = 10
MIN_BITS = 2
MAX_BITS = 8


class QuantizationMethod(str, Enum):
    """Supported quantization methods."""

    GPTQ = "gptq"
    AWQ = "awq"
    DYNAMIC = "dynamic"
    STATIC = "static"
    FLOAT16 = "float16"
    NONE = "none"


class QuantizationConfig(BaseModel):
    """Configuration for a single quantization method.

    Attributes:
        method: Quantization algorithm to use.
        num_bits: Target bit width for weight quantization.
        group_size: Number of weights sharing one scale factor.
        symmetric: Whether to use symmetric or asymmetric quantization.
        calibration_samples: Number of calibration samples for PTQ methods.
        desc_act: Whether to use descending activation order (GPTQ specific).
    """

    method: QuantizationMethod = QuantizationMethod.DYNAMIC
    num_bits: int = Field(default=DEFAULT_NUM_BITS, ge=MIN_BITS, le=MAX_BITS)
    group_size: int = Field(default=DEFAULT_GROUP_SIZE, ge=1)
    symmetric: bool = True
    calibration_samples: int = Field(default=DEFAULT_CALIBRATION_SAMPLES, ge=1)
    desc_act: bool = False


class BenchmarkConfig(BaseModel):
    """Configuration for the benchmark harness.

    Attributes:
        methods: List of quantization methods to compare.
        model_dim: Hidden dimension of the model to benchmark.
        vocab_size: Vocabulary size for the test model.
        num_layers: Number of transformer layers.
        seq_length: Sequence length for evaluation.
        eval_samples: Number of samples for quality evaluation.
        warmup_iterations: Number of warmup iterations before benchmarking.
        benchmark_iterations: Number of timed iterations.
        seed: Random seed for reproducibility.
    """

    methods: list[QuantizationConfig] = Field(default_factory=list)
    model_dim: int = Field(default=256, ge=1)
    vocab_size: int = Field(default=1000, ge=1)
    num_layers: int = Field(default=4, ge=1)
    seq_length: int = Field(default=DEFAULT_SEQ_LENGTH, ge=1)
    eval_samples: int = Field(default=DEFAULT_EVAL_SAMPLES, ge=1)
    warmup_iterations: int = Field(default=DEFAULT_WARMUP_ITERATIONS, ge=0)
    benchmark_iterations: int = Field(default=DEFAULT_BENCHMARK_ITERATIONS, ge=1)
    seed: int = 42

    @model_validator(mode="after")
    def ensure_methods_not_empty(self) -> BenchmarkConfig:
        """Add default methods if none specified."""
        if not self.methods:
            self.methods = [
                QuantizationConfig(method=QuantizationMethod.NONE),
                QuantizationConfig(method=QuantizationMethod.DYNAMIC, num_bits=8),
                QuantizationConfig(method=QuantizationMethod.DYNAMIC, num_bits=4),
                QuantizationConfig(method=QuantizationMethod.STATIC, num_bits=4),
            ]
        return self


@dataclass
class QuantizationResult:
    """Result from quantizing a model with a specific method.

    Attributes:
        method: Quantization method used.
        num_bits: Bit width used.
        original_size_mb: Original model size in megabytes.
        quantized_size_mb: Quantized model size in megabytes.
        compression_ratio: Ratio of original to quantized size.
        quantization_time_sec: Time to quantize in seconds.
        num_parameters: Total number of model parameters.
    """

    method: str = ""
    num_bits: int = DEFAULT_NUM_BITS
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    compression_ratio: float = 1.0
    quantization_time_sec: float = 0.0
    num_parameters: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary.

        Returns:
            Dictionary of field name to value.
        """
        return {
            "method": self.method,
            "num_bits": self.num_bits,
            "original_size_mb": self.original_size_mb,
            "quantized_size_mb": self.quantized_size_mb,
            "compression_ratio": self.compression_ratio,
            "quantization_time_sec": self.quantization_time_sec,
            "num_parameters": self.num_parameters,
        }


@dataclass
class QualityMetrics:
    """Quality metrics for a quantized model.

    Attributes:
        perplexity: Model perplexity on evaluation data (lower is better).
        mse: Mean squared error between original and quantized outputs.
        cosine_similarity: Cosine similarity between output distributions.
        max_abs_error: Maximum absolute error across all outputs.
        snr_db: Signal-to-noise ratio in decibels.
    """

    perplexity: float = 0.0
    mse: float = 0.0
    cosine_similarity: float = 1.0
    max_abs_error: float = 0.0
    snr_db: float = 0.0


@dataclass
class PerformanceMetrics:
    """Performance metrics from inference benchmarking.

    Attributes:
        mean_latency_ms: Mean inference latency in milliseconds.
        p50_latency_ms: 50th percentile latency.
        p99_latency_ms: 99th percentile latency.
        throughput_samples_per_sec: Inference throughput.
        peak_memory_mb: Peak memory usage during inference.
    """

    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    peak_memory_mb: float = 0.0


@dataclass
class BenchmarkReport:
    """Complete benchmark report for one quantization method.

    Attributes:
        quantization: Quantization metadata and compression results.
        quality: Quality degradation metrics.
        performance: Inference performance metrics.
    """

    quantization: QuantizationResult = field(default_factory=QuantizationResult)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    def summary_row(self) -> dict[str, Any]:
        """Generate a summary row for tabular display.

        Returns:
            Dictionary with key metrics for comparison.
        """
        return {
            "method": self.quantization.method,
            "bits": self.quantization.num_bits,
            "size_mb": self.quantization.quantized_size_mb,
            "compression": self.quantization.compression_ratio,
            "perplexity": self.quality.perplexity,
            "cosine_sim": self.quality.cosine_similarity,
            "snr_db": self.quality.snr_db,
            "latency_p50_ms": self.performance.p50_latency_ms,
            "latency_p99_ms": self.performance.p99_latency_ms,
            "throughput": self.performance.throughput_samples_per_sec,
            "memory_mb": self.performance.peak_memory_mb,
        }
