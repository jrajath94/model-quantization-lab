# Architecture: model-quantization-lab

## Overview

The benchmark harness follows a pipeline pattern: configure methods, quantize, evaluate quality, profile performance, report. Each stage is an independent component.

## Components

### ModelQuantizer

- Applies quantization to PyTorch models
- Supports: none, float16, dynamic, static, simulated GPTQ, simulated AWQ
- Key method: group quantization that mimics GPTQ/AWQ scale-per-group behavior

### QualityEvaluator

- Compares original vs quantized model outputs
- Metrics: perplexity, MSE, cosine similarity, max absolute error, SNR
- Uses shared evaluation data for fair comparison

### PerformanceProfiler

- Measures inference latency and throughput
- Reports: mean, p50, p99 latency, throughput, memory estimate
- Includes warmup iterations to avoid cold-start bias

### QuantizationBenchmark

- Orchestrator that runs all methods and collects reports
- Generates shared evaluation data once
- Produces formatted summary table

## Quantization Simulation

Rather than requiring GPU-specific libraries (AutoGPTQ, AutoAWQ), this harness simulates the quantization effect:

1. **Dynamic/Static**: Uniform quantization per-tensor
2. **GPTQ/AWQ**: Group quantization with per-group scale factors

The simulation captures the key quality difference: group quantization preserves more information than per-tensor because each group of weights gets its own scale factor.
