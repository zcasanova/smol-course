# Preference Alignment

This module covers techniques for aligning language models with human preferences. While supervised fine-tuning helps models learn tasks, preference alignment encourages outputs to match human expectations and values.

## Overview

Typical alignment methods involve multiple stages:
1. Supervised Fine-Tuning (SFT) to adapt models to specific domains
2. Preference alignment (like RLHF or DPO) to improve response quality

Alternative approaches like ORPO combine instruction tuning and preference alignment into a single process. Here, we wil focus on DPO and ORPO algorithms.

If you would like to learn more about the different alignment techniques, you can read more about them in the [Argilla Blog](https://argilla.io/blog/mantisnlp-rlhf-part-8). 

### [Direct Preference Optimization (DPO)](./dpo.md)

DPO simplifies preference alignment by directly optimizing models using preference data, eliminating the need for separate reward models and complex reinforcement learning. This makes it more stable and efficient than traditional RLHF.

Key benefits:
- No separate reward model needed
- More stable training process
- Lower computational requirements

### [Odds Ratio Preference Optimization (ORPO)](./orpo.md)

ORPO introduces a combined approach to instruction tuning and preference alignment in a single process. It modifies the standard language modeling objective by combining negative log-likelihood loss with an odds ratio term on a token level.

Key innovations:
- Unified single-stage training process
- Reference model-free architecture
- Improved computational efficiency

ORPO has shown impressive results across various benchmarks. Better performance on AlpacaEval compared to traditional methods. Strong results on MT-Bench, even without multi-turn training. Effective across different model sizes (125M to 1.3B parameters).

## Next Steps

1. Start with [DPO](./dpo.md) for a simpler introduction to preference alignment.
2. Explore [ORPO](./orpo.md) for a unified approach
3. Try the practical tutorials in the `notebooks/` directory

## Resources
- [TRL Documentation](https://huggingface.co/docs/trl/index)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [ORPO Paper](https://arxiv.org/abs/2402.01714)
- [Argilla RLHF Guide](https://argilla.io/blog/mantisnlp-rlhf-part-8/)

