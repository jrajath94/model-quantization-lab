# Interview Prep: model-quantization-lab

## Elevator Pitch (30 seconds)

I built a unified benchmarking harness that runs GPTQ, AWQ, dynamic, and static quantization on the same model with the same evaluation data, measuring perplexity, SNR, cosine similarity, latency, and compression in one report. Nobody else publishes apples-to-apples comparisons -- the results surprised me: group quantization matters more than the algorithm name.

## Why I Built This

### The Real Motivation

Every quantization comparison I found online used different models, different hardware, and different metrics. One blog says GPTQ is best, another says AWQ, another says GGUF. Without a standardized harness, these comparisons are meaningless. I built this to get real answers with controlled variables.

### Company-Specific Framing

| Company    | Why This Matters                                                                                                          |
| ---------- | ------------------------------------------------------------------------------------------------------------------------- |
| Anthropic  | Model serving at scale requires quantization decisions. This harness quantifies the quality-cost tradeoff for deployment. |
| OpenAI     | Shipping models to edge devices requires understanding quantization quality loss. This measures it precisely.             |
| NVIDIA     | TensorRT-LLM quantization decisions need quality benchmarks. This harness provides them.                                  |
| Google     | TPU/GPU deployment decisions depend on quantization quality. Standardized benchmarks enable data-driven choices.          |
| Meta FAIR  | Open-source model releases need quantization recommendations. This provides the data.                                     |
| Citadel/JS | Model deployment latency directly impacts trading systems. Quantization tradeoffs are critical.                           |

## Architecture Deep-Dive

Pipeline: BenchmarkConfig -> ModelQuantizer -> QualityEvaluator + PerformanceProfiler -> BenchmarkReport

### Key Design Decisions

| Decision                      | Why                                                              | Alternative                  | Tradeoff                                                 |
| ----------------------------- | ---------------------------------------------------------------- | ---------------------------- | -------------------------------------------------------- |
| Simulated quantization        | Works without CUDA, captures quality differences                 | Real AutoGPTQ (needs GPU)    | Missing kernel speedups, but quality comparison is valid |
| Group quantization simulation | Shows why GPTQ/AWQ outperform naive quantization                 | Skip and just use per-tensor | Miss the key insight about group size importance         |
| SNR as primary metric         | Scale-independent, intuitive (dB), standard in signal processing | MSE only                     | Requires signal/noise computation but more interpretable |
| Shared eval data              | True apples-to-apples comparison                                 | Random per method            | Small memory cost for holding shared data                |

### Scaling Analysis

- **Current capacity:** 6 methods benchmarked in ~4 seconds
- **10x:** Add real GPTQ/AWQ with GPU, benchmark on larger models (7B)
- **100x:** Distributed benchmarking across GPU types, automated reports
- **Bottleneck:** Quantization time for large models (minutes per method)

## 10 Interview Questions

### Q1: Walk me through how group quantization works.

**A:** Instead of using one scale factor for an entire weight matrix, divide weights into groups of G (typically 128). Each group gets its own scale: scale_i = max(abs(group_i)) / (2^(bits-1) - 1). This means each group can use its full dynamic range, preserving more information. The overhead is one scale value per group (16 bits each), but the quality improvement is significant -- in our benchmarks, group quantization achieves 16.7 dB SNR vs 13.7 dB for per-tensor at the same 4-bit width.

### Q2: Why does 8-bit dynamic quantization show cosine similarity of 1.0000?

**A:** At 8 bits, the quantization error is small enough that it falls below the measurement precision of 32-bit float cosine similarity. The 256 representable values per weight are sufficient to capture the weight distribution with negligible error. This is why 8-bit inference is so popular -- it's effectively lossless for most models.

### Q3: What surprised you about the results?

**A:** Two things. First, GPTQ and AWQ produce nearly identical quality metrics. The algorithmic difference (second-order information vs activation-aware scaling) matters less than the fact that both use group quantization. Second, perplexity stays within 2% even at 4-bit -- the real quality difference shows up in SNR and cosine similarity, which are more sensitive metrics.

### Q4: How would you scale this to 100x?

**A:** Three changes: (1) Use real GPTQ/AWQ libraries with GPU acceleration for accurate timing, (2) benchmark on actual LLMs (7B, 13B, 70B) with real evaluation datasets like C4 or WikiText, (3) distribute across multiple GPU types (A100, H100, consumer GPUs) to show hardware-specific effects.

### Q5: What would you do differently?

**A:** Add calibration data quality analysis -- how does the calibration dataset affect GPTQ/AWQ quality? Add per-layer sensitivity analysis to identify which layers are most affected by quantization. Add mixed-precision support (different bits for different layers).

## Metrics & Results

| Metric                  | Value      | Significance                 |
| ----------------------- | ---------- | ---------------------------- |
| Methods compared        | 6          | Comprehensive coverage       |
| Tests                   | 44 passing | Full component + integration |
| Coverage                | 86%        | Production quality           |
| 8-bit quality loss      | 0%         | Effectively lossless         |
| 4-bit quality loss      | <2% PPL    | Production-viable            |
| Group vs per-tensor SNR | +3 dB      | Group quantization matters   |

## Career Narrative

- **Goldman Sachs (Quant)** -> Model efficiency directly impacts strategy latency; quantization is a deployment optimization problem
- **NVIDIA** -> Deep understanding of GPU memory hierarchies and quantization kernels; this project benchmarks the quality side
- **JPMorgan (VP)** -> Production ML deployment requires quantization decisions; this provides the data to make them
- **This project** -> Shows ability to design fair experiments, measure precisely, and draw correct conclusions from data
