# X Thread: model-quantization-lab

## Tweet 1

"GPTQ is better than AWQ" -- based on what?

Every quantization comparison online uses different models, different hardware, different metrics.

I built a harness that runs them all on the same model. The results surprised me.

Code: github.com/jrajath94/model-quantization-lab

## Tweet 2

The problem: you can't compare quantization methods fairly when:

- Blog A tests GPTQ on Llama-7B
- Blog B tests AWQ on Mistral-7B
- Blog C tests GGUF on a different GPU

Different models + different hardware = meaningless comparison.

## Tweet 3

My approach: one model, one eval dataset, six methods.

Measured: perplexity, cosine similarity, SNR, latency, compression.

The only variables that change are the quantization method and bit width.

## Tweet 4

The surprising finding: GPTQ and AWQ produce IDENTICAL quality metrics.

Both use group quantization (per-group scale factors). The algorithmic difference (second-order info vs activation-aware) matters less than the group size.

Group quantization > algorithm choice.

## Tweet 5

Numbers that matter:

- 8-bit: 3.9x compression, zero quality loss (cosine=1.0000)
- 4-bit naive: 7.5x compression, SNR=13.7 dB
- 4-bit group: 7.5x compression, SNR=16.7 dB (+3 dB)
- All 4-bit: perplexity within 2% of baseline

## Tweet 6

Star it if useful. What quantization method do you use?

github.com/jrajath94/model-quantization-lab

#AI #MachineLearning #Quantization #LLM #OpenSource #BuildInPublic
