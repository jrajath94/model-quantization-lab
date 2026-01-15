"""Quickstart: Run a quantization comparison benchmark."""

import logging

from model_quantization_lab.core import QuantizationBenchmark
from model_quantization_lab.models import BenchmarkConfig, QuantizationConfig, QuantizationMethod


def main() -> None:
    """Run a small quantization benchmark comparing 4 methods."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    config = BenchmarkConfig(
        model_dim=128,
        vocab_size=500,
        num_layers=2,
        seq_length=64,
        eval_samples=32,
        warmup_iterations=2,
        benchmark_iterations=5,
        methods=[
            QuantizationConfig(method=QuantizationMethod.NONE),
            QuantizationConfig(method=QuantizationMethod.DYNAMIC, num_bits=8),
            QuantizationConfig(method=QuantizationMethod.DYNAMIC, num_bits=4),
            QuantizationConfig(method=QuantizationMethod.STATIC, num_bits=4),
            QuantizationConfig(method=QuantizationMethod.GPTQ, num_bits=4),
            QuantizationConfig(method=QuantizationMethod.AWQ, num_bits=4),
        ],
    )

    benchmark = QuantizationBenchmark(config)
    reports = benchmark.run()

    # Print comparison table
    print("\n" + "=" * 90)
    print("  COMPARISON TABLE")
    print("=" * 90)
    print(f"  {'Method':<15s} {'Bits':>4s} {'Size(MB)':>8s} {'Compr':>6s} "
          f"{'PPL':>8s} {'Cos':>8s} {'SNR(dB)':>8s} {'P50(ms)':>8s} {'Tput':>8s}")
    print("-" * 90)
    for r in reports:
        q = r.quantization
        ql = r.quality
        p = r.performance
        compr_str = f"{q.compression_ratio:.1f}x"
        print(f"  {q.method:<15s} {q.num_bits:>4d} {q.quantized_size_mb:>8.2f} "
              f"{compr_str:>6s} {ql.perplexity:>8.2f} "
              f"{ql.cosine_similarity:>8.4f} {ql.snr_db:>8.1f} "
              f"{p.p50_latency_ms:>8.1f} {p.throughput_samples_per_sec:>8.1f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
