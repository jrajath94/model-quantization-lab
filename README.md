# model-quantization-lab

> Apples-to-apples comparison of LLM quantization methods -- same models, same datasets, same metrics, no marketing.

[![CI](https://github.com/jrajath94/model-quantization-lab/workflows/CI/badge.svg)](https://github.com/jrajath94/model-quantization-lab/actions)
[![Coverage](https://codecov.io/gh/jrajath94/model-quantization-lab/branch/master/graph/badge.svg)](https://codecov.io/gh/jrajath94/model-quantization-lab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## The Problem

Quantization is how you make large language models small enough to run on consumer hardware. GPTQ squashes weights to 4-bit integers. AWQ quantizes weights using activation distributions. GGML uses custom formats optimized for CPU inference. All of them claim to be "fast" and "accurate."

But nobody compares them fairly. GPTQ papers report GPTQ results on their chosen datasets. AWQ papers report AWQ results on different datasets with different configurations. Blog posts cherry-pick one model, one metric, one hardware setup. When you need to pick a quantization method for production deployment, you're comparing apples to oranges -- different models, different evaluation prompts, different hardware.

The economics make this matter. An A100 80GB at fp16 serves one Llama 2 70B. With 4-bit quantization, that same GPU serves the model with room to spare for KV cache. That is a 2-4x reduction in serving cost per token. For a service handling millions of requests per day, picking the wrong quantization method costs tens of thousands of dollars per month in GPU spend. Yet the decision is made on vibes, not data.

I built a benchmarking harness that runs every method on the same model with the same inputs and measures the same metrics: perplexity, SNR, cosine similarity, latency, and compression ratio. Plug in any quantization method, get directly comparable numbers. The results surprised me.

## What This Project Does

A unified benchmarking pipeline that evaluates quantization methods under identical conditions, ensuring true apples-to-apples comparison.

- **Standardized evaluation harness** -- run GPTQ, AWQ, dynamic, and static quantization through the same pipeline
- **Six quality metrics** -- perplexity, cosine similarity, SNR (dB), compression ratio, latency (p50/p99), and memory footprint
- **Group quantization simulation** -- captures the core quality advantage of GPTQ/AWQ without requiring GPU-specific libraries
- **Configurable calibration** -- test how calibration data affects quantization quality (it matters more than you think)
- **Reproducible benchmarks** -- shared evaluation data, fixed seeds, deterministic output

## Architecture

```mermaid
graph TD
    A[BenchmarkConfig] -->|methods list| B[QuantizationBenchmark]
    B -->|for each method| C[ModelQuantizer]
    C -->|quantized model| D[QualityEvaluator]
    C -->|quantized model| E[PerformanceProfiler]
    D -->|QualityMetrics| F[BenchmarkReport]
    E -->|PerformanceMetrics| F
    C -->|QuantizationResult| F

    style B fill:#f96,stroke:#333
    style C fill:#9cf,stroke:#333
    style D fill:#6f6,stroke:#333
    style E fill:#6f6,stroke:#333
```

The pipeline follows a config-driven harness pattern. `BenchmarkConfig` defines which methods to compare (with bit width, group size, and calibration parameters). `ModelQuantizer` applies each method to the same base model. `QualityEvaluator` and `PerformanceProfiler` run independently -- quality and speed are separate concerns measured under identical conditions. Results aggregate into a `BenchmarkReport` with side-by-side comparison tables.

## Quick Start

```bash
git clone https://github.com/jrajath94/model-quantization-lab.git
cd model-quantization-lab
make install && make run
```

## How to Use

Run benchmarks on a custom model by configuring quantization methods:

```bash
# Small benchmark (local machine)
python examples/quickstart.py

# Full benchmark with 8 methods
make bench
```

The benchmarking pipeline measures:
- **Quality metrics**: perplexity, cosine similarity, SNR (dB), MSE
- **Performance metrics**: latency (p50/p99), throughput (samples/sec), peak memory
- **Compression**: size reduction and compression ratio

See `/benchmarks/bench_core.py` for a full example benchmark configuration with 8 quantization methods.

## Design Decisions

| Decision                                            | Rationale                                                                               | Alternative Considered       | Tradeoff                                                                  |
| --------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------- |
| Simulated quantization                              | Enables benchmarking without GPU-specific libraries (AutoGPTQ/AutoAWQ)                  | Real library quantization    | Trades production fidelity for portability and CI-testability             |
| Group quantization simulation                       | Captures the key quality advantage of GPTQ/AWQ (per-group scale factors)                | Per-tensor quantization only | Adds complexity but matches real-world results much more closely          |
| SNR as primary metric                               | More interpretable than MSE for comparing degradation across methods                    | MSE alone                    | MSE is scale-dependent; SNR normalizes across different weight magnitudes |
| Shared evaluation data                              | True apples-to-apples: identical inputs across all methods                              | Random data per method       | Requires more memory but eliminates a major confound                      |
| Separate profiler from evaluator                    | Quality and speed are independent concerns that shouldn't couple                        | Single monolithic evaluator  | Slightly more code, but each component is testable and swappable          |
| Pydantic for config, dataclass for hot-path results | Pydantic validates user input; dataclasses avoid overhead on measurement-critical paths | Pydantic everywhere          | Two model types to maintain, but measurably faster in profiling loops     |

## How It Works

The core insight behind this project is that quantization methods differ in one fundamental dimension: how they distribute quantization error across weights.

**GPTQ** (Frantar et al., 2022) works column-by-column through each layer's weight matrix. It computes the Hessian H = 2X^TX from calibration data, quantizes one column to the nearest grid point, then distributes the error across remaining columns using the inverse Hessian. Later columns absorb earlier errors. This needs only 128 calibration samples and runs in minutes.

**AWQ** (Lin et al., 2023) takes the opposite approach. Instead of compensating for error after quantization, it identifies which weight channels matter most _before_ quantizing. A small fraction of weights (roughly 1%) correspond to large activation magnitudes. AWQ scales these salient channels up before quantization so they land on more precise grid points, then scales activations down equivalently at runtime. Mathematically equivalent output, but quantization error on important channels drops dramatically.

**GGML** uses custom formats designed for CPU inference. K-quant variants (Q4_K_M, Q3_K_S) use per-block scaling factors with different bit allocations for scale vs. data bits, trading precision for portability.

The benchmarking harness implements each method's core mechanism: `ModelQuantizer` applies the quantization transform, `QualityEvaluator` measures perplexity (via cross-entropy loss) and cosine similarity (embedding-space fidelity) against the fp16 baseline, and `PerformanceProfiler` records inference latency percentiles and peak memory. Group quantization is simulated by partitioning weight matrices into groups of `group_size` weights, each with its own scale and zero-point -- this is the mechanism that gives GPTQ and AWQ their quality edge over naive per-tensor quantization.

One key design insight: calibration data matters more than the algorithm choice. The harness lets you test how different calibration datasets affect quantization quality, since this choice significantly impacts generalization to downstream tasks. If you are deploying for a specific domain, calibrate on representative data from that domain.

## Testing

```bash
make test    # 44 tests
make bench   # Full benchmark with 8 methods
make lint    # Ruff + mypy
```

## Project Structure

```
model-quantization-lab/
├── src/model_quantization_lab/
│   ├── models.py         # BenchmarkConfig, QuantizationConfig, result types
│   ├── utils.py          # ModelQuantizer, QualityEvaluator, PerformanceProfiler
│   ├── cli.py            # Click-based CLI for running benchmarks
│   └── exceptions.py     # QuantizationError, CalibrationError
├── tests/                # 44 unit + integration tests
├── benchmarks/           # Full benchmark harness
├── examples/             # Quick-start comparison
└── docs/                 # Architecture and interview prep
```

## What I'd Improve

- **Broader benchmark coverage.** HellaSwag, ARC, and TruthfulQA each measure different capabilities. I would add long-context evaluation since quantization sometimes degrades on sequences beyond the calibration length.
- **Newer methods.** SqueezeLLM (non-uniform quantization), SpQR (outlier isolation at higher precision), and QuIP (incoherence processing) are showing promising results. The harness is designed to plug in new methods easily.
- **End-to-end latency.** Token-per-second does not tell the full story when you factor in model loading from disk, CUDA graph capture, and real serving overhead with concurrent requests.

## License

MIT -- Rajath John
