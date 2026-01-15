"""Tests for data models and configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from model_quantization_lab.models import (
    BenchmarkConfig,
    BenchmarkReport,
    PerformanceMetrics,
    QualityMetrics,
    QuantizationConfig,
    QuantizationMethod,
    QuantizationResult,
)


class TestQuantizationConfig:
    """Tests for quantization configuration."""

    def test_default_values(self) -> None:
        """Default config should use dynamic quantization with 4 bits."""
        config = QuantizationConfig()
        assert config.method == QuantizationMethod.DYNAMIC
        assert config.num_bits == 4

    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
    def test_valid_bit_widths(self, bits: int) -> None:
        """All bit widths from 2-8 should be valid."""
        config = QuantizationConfig(num_bits=bits)
        assert config.num_bits == bits

    @pytest.mark.parametrize("bits", [0, 1, 9, 16, 32])
    def test_invalid_bit_widths(self, bits: int) -> None:
        """Bit widths outside 2-8 should be rejected."""
        with pytest.raises(ValidationError):
            QuantizationConfig(num_bits=bits)

    def test_all_methods_valid(self) -> None:
        """All enum methods should be accepted."""
        for method in QuantizationMethod:
            config = QuantizationConfig(method=method)
            assert config.method == method


class TestBenchmarkConfig:
    """Tests for benchmark configuration."""

    def test_default_methods_populated(self) -> None:
        """Empty methods list should be populated with defaults."""
        config = BenchmarkConfig()
        assert len(config.methods) > 0

    def test_custom_methods_preserved(self) -> None:
        """Custom methods list should not be overwritten."""
        custom = [QuantizationConfig(method=QuantizationMethod.GPTQ)]
        config = BenchmarkConfig(methods=custom)
        assert len(config.methods) == 1
        assert config.methods[0].method == QuantizationMethod.GPTQ

    def test_invalid_model_dim_rejected(self) -> None:
        """Zero or negative model dim should be rejected."""
        with pytest.raises(ValidationError):
            BenchmarkConfig(model_dim=0)


class TestQuantizationResult:
    """Tests for quantization result dataclass."""

    def test_to_dict(self) -> None:
        """Result should serialize to dictionary."""
        result = QuantizationResult(
            method="gptq",
            num_bits=4,
            original_size_mb=100.0,
            quantized_size_mb=25.0,
            compression_ratio=4.0,
        )
        d = result.to_dict()
        assert d["method"] == "gptq"
        assert d["compression_ratio"] == 4.0
        assert len(d) == 7


class TestBenchmarkReport:
    """Tests for complete benchmark report."""

    def test_summary_row_keys(self) -> None:
        """Summary row should have all expected keys."""
        report = BenchmarkReport()
        row = report.summary_row()
        expected_keys = {
            "method", "bits", "size_mb", "compression",
            "perplexity", "cosine_sim", "snr_db",
            "latency_p50_ms", "latency_p99_ms",
            "throughput", "memory_mb",
        }
        assert set(row.keys()) == expected_keys

    def test_default_quality_is_perfect(self) -> None:
        """Default quality metrics should indicate no degradation."""
        metrics = QualityMetrics()
        assert metrics.cosine_similarity == 1.0
        assert metrics.mse == 0.0
