"""Tests for core quantization benchmark components."""

from __future__ import annotations

import pytest
import torch

from model_quantization_lab.core import (
    BenchmarkModel,
    ModelQuantizer,
    PerformanceProfiler,
    QualityEvaluator,
    QuantizationBenchmark,
)
from model_quantization_lab.models import (
    BenchmarkConfig,
    BenchmarkReport,
    QuantizationConfig,
    QuantizationMethod,
)
from model_quantization_lab.utils import (
    compute_cosine_similarity,
    compute_mse,
    compute_perplexity,
    compute_snr_db,
    count_parameters,
    estimate_model_size_mb,
    estimate_quantized_size_mb,
)

VOCAB_SIZE = 100
HIDDEN_DIM = 64
NUM_LAYERS = 2
SEQ_LENGTH = 32


class TestBenchmarkModel:
    """Tests for the benchmark model."""

    def test_forward_shape(self, small_model: BenchmarkModel, sample_input: torch.Tensor) -> None:
        """Model should produce logits of correct shape."""
        logits = small_model(sample_input)
        batch_size, seq_len = sample_input.shape
        assert logits.shape == (batch_size, seq_len, VOCAB_SIZE)

    def test_parameter_count(self, small_model: BenchmarkModel) -> None:
        """Model should have a non-trivial number of parameters."""
        num_params = count_parameters(small_model)
        assert num_params > 0


class TestModelQuantizer:
    """Tests for the quantizer."""

    def test_none_quantization_preserves_model(
        self, small_model: BenchmarkModel, sample_input: torch.Tensor
    ) -> None:
        """None quantization should produce identical outputs."""
        config = QuantizationConfig(method=QuantizationMethod.NONE)
        quantizer = ModelQuantizer(config)
        quantized, result = quantizer.quantize(small_model)

        for m in small_model.modules():
            m.training = False
        for m in quantized.modules():
            m.training = False
        with torch.no_grad():
            orig_out = small_model(sample_input)
            quant_out = quantized(sample_input)
        assert torch.allclose(orig_out, quant_out, atol=1e-5)
        assert result.compression_ratio == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize(
        "method,bits",
        [
            (QuantizationMethod.DYNAMIC, 8),
            (QuantizationMethod.DYNAMIC, 4),
            (QuantizationMethod.STATIC, 4),
            (QuantizationMethod.GPTQ, 4),
            (QuantizationMethod.AWQ, 4),
        ],
    )
    def test_quantization_produces_output(
        self,
        small_model: BenchmarkModel,
        sample_input: torch.Tensor,
        method: QuantizationMethod,
        bits: int,
    ) -> None:
        """All quantization methods should produce valid outputs."""
        config = QuantizationConfig(method=method, num_bits=bits)
        quantizer = ModelQuantizer(config)
        quantized, result = quantizer.quantize(small_model)

        output = quantized(sample_input)
        assert output.shape == small_model(sample_input).shape
        assert not torch.isnan(output).any()
        assert result.quantization_time_sec > 0

    def test_compression_ratio_increases_with_lower_bits(
        self, small_model: BenchmarkModel
    ) -> None:
        """Lower bit widths should give higher compression ratios."""
        q8 = ModelQuantizer(QuantizationConfig(method=QuantizationMethod.DYNAMIC, num_bits=8))
        q4 = ModelQuantizer(QuantizationConfig(method=QuantizationMethod.DYNAMIC, num_bits=4))

        _, result_8 = q8.quantize(small_model)
        _, result_4 = q4.quantize(small_model)

        assert result_4.compression_ratio > result_8.compression_ratio

    def test_float16_halves_size(self, small_model: BenchmarkModel) -> None:
        """Float16 quantization should roughly halve model size."""
        config = QuantizationConfig(method=QuantizationMethod.FLOAT16)
        quantizer = ModelQuantizer(config)
        _, result = quantizer.quantize(small_model)

        assert result.compression_ratio == pytest.approx(2.0, abs=0.1)


class TestQualityEvaluator:
    """Tests for quality evaluation."""

    def test_identical_models_perfect_quality(
        self, small_model: BenchmarkModel, benchmark_config: BenchmarkConfig
    ) -> None:
        """Comparing a model to itself should show perfect quality."""
        evaluator = QualityEvaluator(benchmark_config)
        quality = evaluator.evaluate(small_model, small_model)

        assert quality.cosine_similarity == pytest.approx(1.0, abs=1e-4)
        assert quality.mse == pytest.approx(0.0, abs=1e-6)
        assert quality.snr_db > 90

    def test_quantized_model_has_lower_quality(
        self, small_model: BenchmarkModel, benchmark_config: BenchmarkConfig
    ) -> None:
        """Quantized model should have measurably lower quality."""
        config = QuantizationConfig(method=QuantizationMethod.STATIC, num_bits=4)
        quantizer = ModelQuantizer(config)
        quantized, _ = quantizer.quantize(small_model)

        evaluator = QualityEvaluator(benchmark_config)
        quality = evaluator.evaluate(small_model, quantized)

        # Quantization should introduce some error
        assert quality.mse > 0
        assert quality.cosine_similarity < 1.0
        # But not catastrophic (should still be > 0.5)
        assert quality.cosine_similarity > 0.5


class TestPerformanceProfiler:
    """Tests for performance profiling."""

    def test_profiling_returns_positive_values(
        self, small_model: BenchmarkModel, benchmark_config: BenchmarkConfig
    ) -> None:
        """Profiling should return positive latency and throughput."""
        profiler = PerformanceProfiler(benchmark_config)
        perf = profiler.profile(small_model)

        assert perf.mean_latency_ms > 0
        assert perf.p50_latency_ms > 0
        assert perf.p99_latency_ms > 0
        assert perf.throughput_samples_per_sec > 0
        assert perf.peak_memory_mb > 0

    def test_p50_less_than_p99(
        self, small_model: BenchmarkModel, benchmark_config: BenchmarkConfig
    ) -> None:
        """P50 latency should be at most P99."""
        profiler = PerformanceProfiler(benchmark_config)
        perf = profiler.profile(small_model)

        assert perf.p50_latency_ms <= perf.p99_latency_ms


class TestUtilityFunctions:
    """Tests for utility math functions."""

    def test_mse_identical_tensors(self) -> None:
        """MSE of identical tensors should be zero."""
        t = torch.randn(10, 10)
        assert compute_mse(t, t) == pytest.approx(0.0, abs=1e-8)

    def test_cosine_similarity_identical(self) -> None:
        """Cosine similarity of identical tensors should be 1."""
        t = torch.randn(10, 10)
        assert compute_cosine_similarity(t, t) == pytest.approx(1.0, abs=1e-4)

    def test_cosine_similarity_orthogonal(self) -> None:
        """Cosine similarity of orthogonal tensors should be near 0."""
        t1 = torch.tensor([1.0, 0.0])
        t2 = torch.tensor([0.0, 1.0])
        assert abs(compute_cosine_similarity(t1, t2)) < 0.01

    def test_snr_identical_is_high(self) -> None:
        """SNR of identical tensors should be very high."""
        t = torch.randn(10, 10)
        snr = compute_snr_db(t, t)
        assert snr >= 90

    def test_estimate_quantized_smaller(self, small_model: BenchmarkModel) -> None:
        """Quantized estimate should be smaller than original."""
        orig = estimate_model_size_mb(small_model)
        quant = estimate_quantized_size_mb(small_model, num_bits=4)
        assert quant < orig

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_lower_bits_smaller_estimate(self, small_model: BenchmarkModel, bits: int) -> None:
        """Lower bit width should give smaller size estimate."""
        size = estimate_quantized_size_mb(small_model, num_bits=bits)
        size_8 = estimate_quantized_size_mb(small_model, num_bits=8)
        if bits < 8:
            assert size < size_8


class TestQuantizationBenchmark:
    """Integration tests for the full benchmark."""

    def test_benchmark_runs_all_methods(self, benchmark_config: BenchmarkConfig) -> None:
        """Benchmark should produce one report per method."""
        benchmark = QuantizationBenchmark(benchmark_config)
        reports = benchmark.run()
        assert len(reports) == len(benchmark_config.methods)

    def test_reports_are_complete(self, benchmark_config: BenchmarkConfig) -> None:
        """Each report should have all three sections."""
        benchmark = QuantizationBenchmark(benchmark_config)
        reports = benchmark.run()
        for report in reports:
            assert report.quantization.num_parameters > 0
            assert report.performance.mean_latency_ms > 0
            assert report.quality.perplexity > 0
