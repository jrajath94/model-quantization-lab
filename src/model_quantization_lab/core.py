"""Core quantization benchmarking components.

The benchmark harness is decomposed into four components:
  1. ModelQuantizer — applies quantization methods to models
  2. QualityEvaluator — measures quality degradation (perplexity, MSE, SNR)
  3. PerformanceProfiler — measures inference latency and memory
  4. QuantizationBenchmark — orchestrates the full comparison pipeline

Each component can be used independently for targeted analysis.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from model_quantization_lab.exceptions import (
    EvaluationError,
    ProfilingError,
    QuantizationFailedError,
)
from model_quantization_lab.models import (
    BenchmarkConfig,
    BenchmarkReport,
    PerformanceMetrics,
    QualityMetrics,
    QuantizationConfig,
    QuantizationMethod,
    QuantizationResult,
)
from model_quantization_lab.utils import (
    compute_cosine_similarity,
    compute_mse,
    compute_perplexity,
    compute_snr_db,
    count_parameters,
    estimate_model_size_mb,
    estimate_quantized_size_mb,
    set_seed,
    time_function,
)

logger = logging.getLogger(__name__)


class SimpleTransformerBlock(nn.Module):
    """A simplified transformer block for benchmarking.

    Uses linear projections instead of multi-head attention for
    deterministic, fast benchmarking without external dependencies.

    Args:
        hidden_dim: Hidden dimension size.
        dropout: Dropout probability.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self._qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self._out_proj = nn.Linear(hidden_dim, hidden_dim)
        self._ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self._norm1 = nn.LayerNorm(hidden_dim)
        self._norm2 = nn.LayerNorm(hidden_dim)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Output tensor of same shape.
        """
        # Simplified self-attention (linear projection only)
        normed = self._norm1(x)
        qkv = self._qkv_proj(normed)
        q, k, v = qkv.chunk(3, dim=-1)
        # Simplified: use v as the attention output
        attn_out = self._out_proj(v)
        x = x + self._dropout(attn_out)

        # Feed-forward
        normed = self._norm2(x)
        ff_out = self._ffn(normed)
        x = x + self._dropout(ff_out)
        return x


class BenchmarkModel(nn.Module):
    """A lightweight model for quantization benchmarking.

    Designed to exercise the same operations as a real transformer
    (embeddings, linear layers, layer norms, activations) while being
    small enough to benchmark quickly.

    Args:
        vocab_size: Vocabulary size.
        hidden_dim: Hidden dimension.
        num_layers: Number of transformer blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, hidden_dim)
        self._blocks = nn.ModuleList([
            SimpleTransformerBlock(hidden_dim) for _ in range(num_layers)
        ])
        self._head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass producing logits.

        Args:
            token_ids: Input token IDs of shape (batch, seq_len).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        x = self._embedding(token_ids)
        for block in self._blocks:
            x = block(x)
        return self._head(x)


class ModelQuantizer:
    """Applies quantization methods to PyTorch models.

    Supports dynamic quantization, static quantization, and simulated
    GPTQ/AWQ-style weight quantization. Real GPTQ/AWQ require the
    optional `quantization` dependency.

    Args:
        config: Quantization configuration.
    """

    def __init__(self, config: QuantizationConfig) -> None:
        self._config = config

    def quantize(self, model: nn.Module) -> tuple[nn.Module, QuantizationResult]:
        """Quantize a model according to the configuration.

        Args:
            model: Original PyTorch model.

        Returns:
            Tuple of (quantized_model, quantization_result).

        Raises:
            QuantizationFailedError: If quantization fails.
        """
        original_size = estimate_model_size_mb(model)
        num_params = count_parameters(model)

        start_time = time.perf_counter()

        try:
            if self._config.method == QuantizationMethod.NONE:
                quantized = copy.deepcopy(model)
                quantized_size = original_size
            elif self._config.method == QuantizationMethod.FLOAT16:
                quantized = copy.deepcopy(model).half()
                quantized_size = original_size / 2
            elif self._config.method == QuantizationMethod.DYNAMIC:
                quantized = self._apply_dynamic_quantization(model)
                quantized_size = estimate_quantized_size_mb(
                    model, self._config.num_bits, self._config.group_size
                )
            elif self._config.method == QuantizationMethod.STATIC:
                quantized = self._apply_static_quantization(model)
                quantized_size = estimate_quantized_size_mb(
                    model, self._config.num_bits, self._config.group_size
                )
            elif self._config.method in (QuantizationMethod.GPTQ, QuantizationMethod.AWQ):
                quantized = self._apply_simulated_weight_quantization(model)
                quantized_size = estimate_quantized_size_mb(
                    model, self._config.num_bits, self._config.group_size
                )
            else:
                raise QuantizationFailedError(
                    f"Unsupported method: {self._config.method}"
                )
        except QuantizationFailedError:
            raise
        except Exception as exc:
            raise QuantizationFailedError(
                f"Quantization failed for {self._config.method}: {exc}"
            ) from exc

        elapsed = time.perf_counter() - start_time
        compression = original_size / quantized_size if quantized_size > 0 else 1.0

        result = QuantizationResult(
            method=self._config.method.value,
            num_bits=self._config.num_bits,
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=compression,
            quantization_time_sec=elapsed,
            num_parameters=num_params,
        )

        logger.info(
            "Quantized with %s (%d-bit): %.2f MB -> %.2f MB (%.1fx compression) in %.2fs",
            self._config.method.value,
            self._config.num_bits,
            original_size,
            quantized_size,
            compression,
            elapsed,
        )

        return quantized, result

    def _apply_dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply PyTorch dynamic quantization.

        Args:
            model: Model to quantize.

        Returns:
            Dynamically quantized model.
        """
        quantized = copy.deepcopy(model)
        # Simulate dynamic quantization effect on weights
        with torch.no_grad():
            for param in quantized.parameters():
                if param.dim() >= 2:
                    self._quantize_tensor_inplace(param, self._config.num_bits)
        return quantized

    def _apply_static_quantization(self, model: nn.Module) -> nn.Module:
        """Apply simulated static quantization.

        Args:
            model: Model to quantize.

        Returns:
            Statically quantized model (simulated).
        """
        quantized = copy.deepcopy(model)
        with torch.no_grad():
            for param in quantized.parameters():
                self._quantize_tensor_inplace(param, self._config.num_bits)
        return quantized

    def _apply_simulated_weight_quantization(self, model: nn.Module) -> nn.Module:
        """Simulate GPTQ/AWQ-style group weight quantization.

        Quantizes weights per group with separate scale and zero-point.

        Args:
            model: Model to quantize.

        Returns:
            Model with quantized weights.
        """
        quantized = copy.deepcopy(model)
        with torch.no_grad():
            for param in quantized.parameters():
                if param.dim() >= 2:
                    self._group_quantize_inplace(
                        param,
                        self._config.num_bits,
                        self._config.group_size,
                    )
        return quantized

    @staticmethod
    def _quantize_tensor_inplace(
        tensor: torch.Tensor,
        num_bits: int,
    ) -> None:
        """Quantize a tensor in-place using uniform quantization.

        Args:
            tensor: Tensor to quantize.
            num_bits: Target bit width.
        """
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1

        abs_max = tensor.abs().max()
        if abs_max < 1e-10:
            return

        scale = abs_max / qmax
        quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
        tensor.copy_(quantized * scale)

    @staticmethod
    def _group_quantize_inplace(
        tensor: torch.Tensor,
        num_bits: int,
        group_size: int,
    ) -> None:
        """Quantize a tensor per group for fine-grained scale factors.

        Args:
            tensor: Tensor to quantize (must be 2D+).
            num_bits: Target bit width.
            group_size: Number of elements per group.
        """
        original_shape = tensor.shape
        flat = tensor.flatten()
        num_elements = flat.numel()

        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1

        for start in range(0, num_elements, group_size):
            end = min(start + group_size, num_elements)
            group = flat[start:end]

            abs_max = group.abs().max()
            if abs_max < 1e-10:
                continue

            scale = abs_max / qmax
            quantized = torch.clamp(torch.round(group / scale), qmin, qmax)
            flat[start:end] = quantized * scale

        tensor.copy_(flat.reshape(original_shape))


class QualityEvaluator:
    """Evaluates quality degradation from quantization.

    Compares original and quantized model outputs on the same inputs
    to measure perplexity, MSE, cosine similarity, and SNR.

    Args:
        config: Benchmark configuration.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config

    @torch.no_grad()
    def evaluate(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        eval_data: Optional[torch.Tensor] = None,
    ) -> QualityMetrics:
        """Compare quality between original and quantized model.

        Args:
            original_model: The unquantized reference model.
            quantized_model: The quantized model to evaluate.
            eval_data: Optional evaluation token IDs. Generated if None.

        Returns:
            QualityMetrics with degradation measurements.

        Raises:
            EvaluationError: If evaluation fails.
        """
        try:
            if eval_data is None:
                eval_data = torch.randint(
                    0, self._config.vocab_size,
                    (self._config.eval_samples, self._config.seq_length),
                )

            # Get outputs from both models
            # Set all submodules to inference mode
            for module in original_model.modules():
                module.training = False
            for module in quantized_model.modules():
                module.training = False

            orig_logits = original_model(eval_data)
            quant_logits = quantized_model(eval_data.to(
                next(quantized_model.parameters()).dtype
                if next(quantized_model.parameters()).is_floating_point()
                else eval_data.dtype
            ) if False else eval_data)

            # Handle float16 model output
            if quant_logits.dtype == torch.float16:
                quant_logits = quant_logits.float()

            # Compute metrics
            mse = compute_mse(orig_logits, quant_logits)
            cosine_sim = compute_cosine_similarity(orig_logits, quant_logits)
            snr = compute_snr_db(orig_logits, quant_logits)
            max_abs_err = float(
                torch.max(torch.abs(orig_logits - quant_logits)).item()
            )

            # Perplexity using original model's predictions as targets
            targets = torch.argmax(orig_logits, dim=-1)
            perplexity = compute_perplexity(quant_logits, targets)

            return QualityMetrics(
                perplexity=perplexity,
                mse=mse,
                cosine_similarity=cosine_sim,
                max_abs_error=max_abs_err,
                snr_db=snr,
            )

        except Exception as exc:
            raise EvaluationError(f"Quality evaluation failed: {exc}") from exc


class PerformanceProfiler:
    """Profiles inference performance of quantized models.

    Measures latency, throughput, and memory usage.

    Args:
        config: Benchmark configuration.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config

    @torch.no_grad()
    def profile(
        self,
        model: nn.Module,
        input_data: Optional[torch.Tensor] = None,
    ) -> PerformanceMetrics:
        """Profile inference performance.

        Args:
            model: Model to profile.
            input_data: Optional input data. Generated if None.

        Returns:
            PerformanceMetrics with latency and throughput.

        Raises:
            ProfilingError: If profiling fails.
        """
        try:
            if input_data is None:
                input_data = torch.randint(
                    0, self._config.vocab_size,
                    (1, self._config.seq_length),
                )

            for module in model.modules():
                module.training = False

            def inference_fn() -> None:
                model(input_data)

            timing = time_function(
                inference_fn,
                warmup=self._config.warmup_iterations,
                iterations=self._config.benchmark_iterations,
            )

            # Estimate memory
            peak_memory = estimate_model_size_mb(model)

            return PerformanceMetrics(
                mean_latency_ms=timing["mean_ms"],
                p50_latency_ms=timing["p50_ms"],
                p99_latency_ms=timing["p99_ms"],
                throughput_samples_per_sec=1000.0 / timing["mean_ms"],
                peak_memory_mb=peak_memory,
            )

        except Exception as exc:
            raise ProfilingError(f"Performance profiling failed: {exc}") from exc


class QuantizationBenchmark:
    """Orchestrates the full quantization comparison pipeline.

    For each configured method: quantize -> evaluate quality -> profile performance.

    Args:
        config: Benchmark configuration.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config
        set_seed(config.seed)
        self._model = BenchmarkModel(
            config.vocab_size, config.model_dim, config.num_layers
        )
        self._evaluator = QualityEvaluator(config)
        self._profiler = PerformanceProfiler(config)
        self._reports: list[BenchmarkReport] = []

        logger.info(
            "Benchmark initialized: model_dim=%d, layers=%d, vocab=%d, params=%d",
            config.model_dim,
            config.num_layers,
            config.vocab_size,
            count_parameters(self._model),
        )

    @property
    def model(self) -> nn.Module:
        """Access the original benchmark model."""
        return self._model

    @property
    def reports(self) -> list[BenchmarkReport]:
        """Access all benchmark reports."""
        return self._reports

    def run(self) -> list[BenchmarkReport]:
        """Run the full benchmark across all configured methods.

        Returns:
            List of BenchmarkReport, one per quantization method.
        """
        logger.info(
            "Starting benchmark with %d methods", len(self._config.methods)
        )

        # Generate shared evaluation data
        eval_data = torch.randint(
            0, self._config.vocab_size,
            (self._config.eval_samples, self._config.seq_length),
        )

        for quant_config in self._config.methods:
            report = self._benchmark_method(quant_config, eval_data)
            self._reports.append(report)

        self._log_summary()
        return self._reports

    def _benchmark_method(
        self,
        quant_config: QuantizationConfig,
        eval_data: torch.Tensor,
    ) -> BenchmarkReport:
        """Benchmark a single quantization method.

        Args:
            quant_config: Configuration for this method.
            eval_data: Shared evaluation data.

        Returns:
            Complete BenchmarkReport.
        """
        method_name = f"{quant_config.method.value}-{quant_config.num_bits}bit"
        logger.info("Benchmarking: %s", method_name)

        # Quantize
        quantizer = ModelQuantizer(quant_config)
        quantized_model, quant_result = quantizer.quantize(self._model)

        # Evaluate quality
        quality = self._evaluator.evaluate(
            self._model, quantized_model, eval_data
        )

        # Profile performance
        performance = self._profiler.profile(quantized_model)

        report = BenchmarkReport(
            quantization=quant_result,
            quality=quality,
            performance=performance,
        )

        logger.info(
            "  %s: perplexity=%.2f, cosine=%.4f, snr=%.1f dB, "
            "latency_p50=%.1f ms, compression=%.1fx",
            method_name,
            quality.perplexity,
            quality.cosine_similarity,
            quality.snr_db,
            performance.p50_latency_ms,
            quant_result.compression_ratio,
        )

        return report

    def _log_summary(self) -> None:
        """Log a summary table of all benchmark results."""
        logger.info("\n" + "=" * 80)
        logger.info("  QUANTIZATION BENCHMARK RESULTS")
        logger.info("=" * 80)

        header = (
            f"  {'Method':<20s} {'Bits':>4s} {'Size(MB)':>8s} "
            f"{'Compr':>6s} {'PPL':>8s} {'Cosine':>8s} "
            f"{'SNR(dB)':>8s} {'P50(ms)':>8s}"
        )
        logger.info(header)
        logger.info("-" * 80)

        for report in self._reports:
            q = report.quantization
            ql = report.quality
            p = report.performance
            logger.info(
                "  %-20s %4d %8.2f %6.1fx %8.2f %8.4f %8.1f %8.1f",
                q.method,
                q.num_bits,
                q.quantized_size_mb,
                q.compression_ratio,
                ql.perplexity,
                ql.cosine_similarity,
                ql.snr_db,
                p.p50_latency_ms,
            )
        logger.info("=" * 80)
