"""Benchmark the quantization lab on a medium-sized model."""

import logging
import sys
import time

from model_quantization_lab.core import QuantizationBenchmark
from model_quantization_lab.models import BenchmarkConfig, QuantizationConfig, QuantizationMethod

logger = logging.getLogger(__name__)


def main() -> int:
    """Run the full benchmark and print results.

    Returns:
        Exit code.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = BenchmarkConfig(
        model_dim=256,
        vocab_size=1000,
        num_layers=4,
        seq_length=128,
        eval_samples=64,
        warmup_iterations=3,
        benchmark_iterations=10,
        methods=[
            QuantizationConfig(method=QuantizationMethod.NONE),
            QuantizationConfig(method=QuantizationMethod.FLOAT16),
            QuantizationConfig(method=QuantizationMethod.DYNAMIC, num_bits=8),
            QuantizationConfig(method=QuantizationMethod.DYNAMIC, num_bits=4),
            QuantizationConfig(method=QuantizationMethod.STATIC, num_bits=8),
            QuantizationConfig(method=QuantizationMethod.STATIC, num_bits=4),
            QuantizationConfig(method=QuantizationMethod.GPTQ, num_bits=4),
            QuantizationConfig(method=QuantizationMethod.AWQ, num_bits=4),
        ],
    )

    start = time.perf_counter()
    benchmark = QuantizationBenchmark(config)
    reports = benchmark.run()
    total_time = time.perf_counter() - start

    # Summary
    print("\n" + "=" * 100)
    print("  FULL BENCHMARK RESULTS")
    print("=" * 100)
    print(f"  {'Method':<15s} {'Bits':>4s} {'Size(MB)':>8s} {'Compr':>6s} "
          f"{'PPL':>8s} {'Cos':>8s} {'SNR(dB)':>8s} "
          f"{'P50(ms)':>8s} {'P99(ms)':>8s} {'Tput':>8s}")
    print("-" * 100)
    for r in reports:
        q = r.quantization
        ql = r.quality
        p = r.performance
        compr_str = f"{q.compression_ratio:.1f}x"
        print(f"  {q.method:<15s} {q.num_bits:>4d} {q.quantized_size_mb:>8.2f} "
              f"{compr_str:>6s} {ql.perplexity:>8.2f} "
              f"{ql.cosine_similarity:>8.4f} {ql.snr_db:>8.1f} "
              f"{p.p50_latency_ms:>8.1f} {p.p99_latency_ms:>8.1f} "
              f"{p.throughput_samples_per_sec:>8.1f}")
    print("=" * 100)
    print(f"  Total benchmark time: {total_time:.1f}s")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
