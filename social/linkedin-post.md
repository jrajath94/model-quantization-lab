# LinkedIn Post: model-quantization-lab

I just open-sourced model-quantization-lab -- a unified benchmarking harness that settles the GPTQ vs AWQ vs GGUF debate with controlled experiments.

The problem: every quantization comparison online uses different models, different hardware, and different evaluation metrics. When one blog says GPTQ is best and another says AWQ, neither is wrong -- they're just not comparable. There's no standardized harness that runs every method on the same model with the same inputs and measures the same quality metrics.

My approach: one model, one evaluation dataset, six methods. The harness measures perplexity (language quality), SNR (signal fidelity), cosine similarity (output distribution), latency, and compression ratio. The surprising result: GPTQ and AWQ produce nearly identical quality metrics. What matters isn't the algorithm name -- it's whether you use group quantization (per-group scale factors). Group quantization adds 3 dB of SNR over naive per-tensor quantization at the same bit width.

The project includes 44 tests at 86% coverage, Pydantic-validated configurations, and a pipeline architecture that makes it easy to add new quantization methods. All results are reproducible with `make run`.

GitHub: github.com/jrajath94/model-quantization-lab

#AI #MachineLearning #Quantization #ModelCompression #OpenSource #LLM
