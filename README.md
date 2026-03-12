# Exploring Gate Behavior in Gated Attention: An Empirical Notebook

> **A hands-on look at the pretrained models from [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708) (NeurIPS 2025 Oral, Best Paper)**
>
> This notebook pokes around inside the 1B-parameter gated attention models released by the [Qwen team](https://github.com/qiuzh20/gated_attention), pulls out the learned gate values, plots what they look like, and sees what breaks when they are overridden.

> **What this is and isn't:** This is an exploratory weekend project, not a research paper. Everything runs on a free Colab T4 in about 35 minutes. Evaluation is on WikiText-2 only (roughly 25K tokens). No training is done, and no proper controls are run. Think of this as "let's see what's going on inside the model" rather than "here are rigorous scientific findings." The [Limitations](#limitations-and-open-questions) section is honest about what can and can't be concluded.

---

## Table of Contents

- [What's This About?](#whats-this-about)
- [Setup and Models](#setup-and-models)
- [Experiment 1: What Do the Gates Look Like?](#experiment-1-what-do-the-gates-look-like)
  - [1.1 The Big Picture](#11-the-big-picture)
  - [1.2 Which Heads and Layers Are Open vs Closed?](#12-which-heads-and-layers-are-open-vs-closed)
  - [1.3 The First Token Gets Suppressed](#13-the-first-token-gets-suppressed)
  - [1.4 Gate Scores Token by Token](#14-gate-scores-token-by-token)
  - [1.5 Do Different Inputs Get Different Gating? (Very Preliminary)](#15-do-different-inputs-get-different-gating-very-preliminary)
  - [1.6 How Gate Distributions Change With Depth](#16-how-gate-distributions-change-with-depth)
  - [1.7 Headwise vs Elementwise](#17-headwise-vs-elementwise)
- [Experiment 2: What Breaks When the Gates Are Overridden?](#experiment-2-what-breaks-when-the-gates-are-overridden)
  - [2.1 Force All Gates to a Single Value](#21-force-all-gates-to-a-single-value)
  - [2.2 Sweeping Across Fixed Values](#22-sweeping-across-fixed-values)
  - [2.3 Breaking One Layer at a Time](#23-breaking-one-layer-at-a-time)
  - [2.4 Killing One Head at a Time](#24-killing-one-head-at-a-time)
- [Summary of Observations](#summary-of-observations)
- [Limitations and Open Questions](#limitations-and-open-questions)
- [What Would Make This Proper Research](#what-would-make-this-proper-research)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

---

## What's This About?

The original paper shows that sticking a simple gate (a sigmoid function) after the attention computation makes language models better in several ways. The gate learns to filter each attention head's output, letting useful information through and blocking noise.

The paper reports some overall statistics about these gates (average values, attention sink percentages), but doesn't go into detail about what individual layers and heads are doing, or what happens if the gates are messed with after training. That's what this notebook explores.

The approach uses standard tools that have been around since at least 2019: hooking into the model's internals during a forward pass, grabbing the gate values, and examining them. For the ablation experiments, the learned gate values are replaced with fixed numbers to see how much the model's quality drops (measured by perplexity, where lower means the model is better at predicting text).

Nothing about the method is new. The only thing being added here is applying these techniques to the specific case of gated attention models and writing up what was found.

---

## Setup and Models

Three pretrained 1B-parameter models from [QwQZh/gated_attention](https://huggingface.co/QwQZh/gated_attention) are used:

| Model | What's Different | How It Works |
|-------|-----------------|--------------|
| `1B_baseline` | No gates at all | Normal attention |
| `1B_gate_headwise` | One gate value per head | Each of the 16 heads gets a single on/off dial |
| `1B_gate_elementwise` | 128 gate values per head | Each dimension inside each head gets its own dial |

All three have 28 layers, 16 attention heads, and were trained on 3.5 trillion tokens.

**How the gate values are grabbed:** The model packs the gate scores into the same output as the query vectors (inside `q_proj`). A hook is attached to that layer, the output is intercepted, and the gate portion is split off using the exact same logic as the [model's source code](https://github.com/qiuzh20/gated_attention/blob/main/modeling_qwen3.py). For ablation experiments, the hook replaces the gate values with whatever fixed number is needed before the rest of the model sees them.

**How quality is measured:** Perplexity on WikiText-2 (a standard text dataset), using 50 chunks of 512 tokens each, so about 25K tokens total. That's a pretty small evaluation surface, which is discussed in [Limitations](#limitations-and-open-questions).

---

## Experiment 1: What Do the Gates Look Like?

### 1.1 The Big Picture

<p align="center">
<img src="figures/headwise_gate_distribution.png" width="100%">
</p>

**Headwise model numbers:**

| What Was Measured | Value |
|-----------------|-------|
| Average gate score | 0.1852 |
| Median gate score | 0.1412 |
| Standard deviation | 0.1522 |
| Gates nearly closed (below 0.1) | 36.64% |
| Gates nearly open (above 0.9) | 0.15% |

Most gate values are small. Over a third are below 0.1, and almost none are above 0.9. The gates are doing a lot more closing than opening, which fits what the paper describes as "sparse gating."

The middle chart shows that early layers (0 through 3) have the lowest gate values (around 0.08 to 0.12), middle layers are the most open (around 0.20 to 0.25), and the final layer gets sparse again. The right chart shows the same thing from a sparsity angle: Layer 0 has 77% of its gates nearly closed, while Layer 14 only has 16%.

So the model seems to filter aggressively at the start, open up in the middle where most of the processing happens, and then tighten up again at the end.

### 1.2 Which Heads and Layers Are Open vs Closed?

<p align="center">
<img src="figures/headwise_per_head_heatmap.png" width="85%">
</p>

| Most Closed Heads (averaged across layers) | Score | Most Open Heads | Score |
|--------------------------------------------|-------|-----------------|-------|
| Head 3 | 0.1572 | Head 12 | 0.1935 |
| Head 2 | 0.1659 | Head 1 | 0.1955 |
| Head 8 | 0.1668 | Head 15 | 0.2152 |

**Most closed layers:** Layer 0 (0.079), Layer 2 (0.092), Layer 1 (0.100)

**Most open layers:** Layer 18 (0.253), Layer 11 (0.244), Layer 26 (0.279)

The difference between the most closed head and the most open head is only about 1.4x, so no head is dramatically different from the others when averaged across all layers. The interesting stuff shows up at specific (layer, head) combinations, which can be seen in the heatmap. Some cells are very dark (nearly off) while others in the same layer are brighter.

### 1.3 The First Token Gets Suppressed

<p align="center">
<img src="figures/headwise_first_token_comparison.png" width="85%">
</p>

| Position | Average Gate Score |
|----------|-------------------|
| First token ("The") | 0.0684 |
| All other tokens | 0.1790 |
| How much lower the first token is | 2.6x |

The first token gets noticeably lower gate scores across every single layer, and the gap gets wider in the middle and late layers.

This connects to something the paper discusses called the "attention sink." In regular transformers, when the model doesn't have anything useful to attend to, it tends to dump all the leftover attention weight onto the first token (because the math forces the weights to add up to 1). The gate gives the model another option: just turn down the volume on the output for that token instead. The consistently low gate scores on token 0 suggest the model has learned to do exactly that.

**Caveat:** This was only tested on a handful of prompts. To be confident about this pattern, hundreds of diverse inputs would be needed, along with checks on whether it's about the position (always token 0) or the specific word ("The").

### 1.4 Gate Scores Token by Token

<p align="center">
<img src="figures/headwise_token_gate_heatmap.png" width="85%">
</p>

This heatmap shows gate scores for one prompt ("The transformer architecture uses multi-head attention to process sequences in parallel.") across all 28 layers.

The first token is a dark column the whole way down, matching what was found above. Content words in the middle of the sentence tend to get higher scores, especially in middle layers. The period at the end also gets relatively low scores.

This is just one sentence though, so treat it as a picture of what's happening rather than proof of a general pattern.

### 1.5 Do Different Inputs Get Different Gating? (Very Preliminary)

<p align="center">
<img src="figures/headwise_cross_prompt_comparison.png" width="100%">
</p>

| Input Type | Average Gate | Nearly Closed (below 0.1) |
|-----------|-------------|--------------------------|
| Factual | 0.1589 | 44.3% |
| Technical | 0.1705 | 42.2% |
| Conversational | 0.1732 | 39.0% |
| Repetitive | 0.2024 | 33.8% |
| Code-like | 0.2116 | 27.7% |

> **Important: exactly one sentence per category was used.** That means these numbers could be entirely about the specific sentences picked (how long they are, what words they use, how they tokenize) and not about the category at all. No real conclusions can be drawn from n=1. This is included because the per-layer profiles are interesting to look at, not because it proves code "opens gates more than prose" or anything like that.

What is interesting is that all five inputs follow a similar shape across layers (low at the start, rising, oscillating, dropping at the end) but at different heights. Whether that reflects something real about how different content types are processed would require testing with hundreds of examples per category and proper statistics.

### 1.6 How Gate Distributions Change With Depth

<p align="center">
<img src="figures/headwise_per_layer_distributions.png" width="100%">
</p>

| Layer | Average | Nearly Closed | What It Looks Like |
|-------|---------|---------------|--------------------|
| 0 (first) | 0.080 | 77% | Almost everything is near zero |
| 7 (early-mid) | 0.225 | 12% | A smooth curve centered around 0.15 to 0.25 |
| 14 (middle) | 0.238 | 16% | The widest spread, values from 0.05 all the way to 0.55 |
| 27 (last) | 0.137 | 64% | Two humps: most near zero, but a small cluster reaching toward 1.0 |

The shape of the distribution changes a lot depending on where you are in the model. Layer 0 is basically "off." The middle layers have the widest spread, meaning the gate is making the most varied decisions about what to let through. The last layer has an interesting two-hump pattern: most gates are nearly closed, but a small group is open, as if the model only needs a few heads active at the very end.

### 1.7 Headwise vs Elementwise

<p align="center">
<img src="figures/headwise_vs_elementwise_comparison.png" width="100%">
</p>

| What Was Measured | Headwise | Elementwise |
|-----------------|----------|-------------|
| Average gate score | 0.1852 | 0.1038 |
| Median gate score | 0.1412 | 0.0434 |
| Below 0.1 | 36.64% | 70.35% |
| Above 0.9 | 0.15% | 0.32% |

The elementwise model is way sparser. 70% of its gate values are below 0.1, compared to 37% for headwise. This makes sense: when there are 128 separate dials per head instead of 1, the model can be much more selective about exactly which pieces of information to keep and which to throw away.

**Looking inside a single head (Layer 14, elementwise):**

| Pattern | How Common |
|---------|-----------|
| Always closed (below 0.1 on average) | 31.8% of dimensions |
| Always open (above 0.9 on average) | 0.0% of dimensions |
| Changes depending on input (high variance) | 37.4% of dimensions |

About a third of the dimensions inside each head are permanently turned off no matter what text is fed in. Another third actively changes based on the input. And none are permanently fully open. Every dimension gets at least some filtering.

---

## Experiment 2: What Breaks When the Gates Are Overridden?

Experiment 1 was about looking at the gates. Experiment 2 is about breaking them on purpose to see what happens.

**A big caveat before starting:** This model was trained with gates. That means every other weight in the model learned to expect gated outputs. So when the gates are forced to weird values, the model breaks. But that would probably happen if any other learned component were tampered with too, like the normalization layers or the feed-forward weights. **Those other components were not tested**, so it's unclear whether the gate is special or if this is just what happens when any part of a trained model is interfered with. The original paper establishes that gating helps through proper training experiments; these ablation experiments can't make that claim on their own.

### 2.1 Force All Gates to a Single Value

| What Was Done | Perplexity | How Much Worse |
|------------|-----------|----------------|
| **Leave gates alone (baseline)** | **12.39** | **n/a** |
| Force all gates to 0.116 (the paper's average) | 276.97 | 22x worse |
| Force all gates to 1.0 (fully open, no filtering) | 251,076 | Broken |
| Force all gates to 0.0 (fully closed, kill attention) | 303,654 | Broken |

Forcing gates to either extreme (all open or all closed) completely destroys the model. The fixed average value (0.116) is much better than the extremes but still 22x worse than the actual learned gates. This suggests that the token-by-token pattern of the gates matters a lot to how this trained model works, not just the overall level of sparsity.

### 2.2 Sweeping Across Fixed Values

<p align="center">
<img src="figures/progressive_gate_sweep.png" width="85%">
</p>

| Fixed Gate Value | Perplexity |
|-----------------|-----------|
| 0.00 | 303,654 |
| 0.05 | 47,582 |
| 0.10 | 621 |
| **0.15** | **112** (best fixed value) |
| 0.20 | 175 |
| 0.30 | 9,415 |
| 0.50 | 36,392 |
| 1.00 | 251,076 |

There's a very narrow sweet spot around 0.15. Step outside the 0.10 to 0.20 range and things fall apart fast. The damage is also lopsided: going from 0.15 to 0.30 (making gates too open) adds about 9,300 to perplexity, while going from 0.15 to 0.10 (making gates too closed) only adds about 500. The model is much more sensitive to gates being too open than too closed, which fits with the idea that the gate's main job is suppression.

Even the best fixed value (0.15, perplexity 112) is 9x worse than the learned gates (12.39). That gap represents how much the per-token, per-head, per-layer pattern contributes on top of just being at the right average level.

### 2.3 Breaking One Layer at a Time

<p align="center">
<img src="figures/layer_ablation_gates_open.png" width="100%">
</p>

Each layer's gates were forced fully open (to 1.0) one layer at a time and the quality drop was measured.

| Layer | Perplexity | How Much It Changed |
|-------|-----------|-------------------|
| **0** | **148,419** | **+148,407** |
| 1 | 245.62 | +233.23 |
| 2 | 97.92 | +85.53 |
| 4 | 22.28 | +9.90 |
| 3 | 20.39 | +8.00 |
| 5 | 16.38 | +3.99 |
| 19 | 15.86 | +3.47 |
| ... | ... | ... |
| 9 | 13.04 | +0.65 |
| 26 | 13.00 | +0.61 |

Layer 0 is in a category by itself. Breaking just its gates accounts for 59% of the total damage from breaking all gates at once. Layers 0 through 2 together account for almost everything. Layers 6 through 27 barely matter (each one adds less than 3.5 to perplexity).

Layer 0 is also the most aggressively closed layer (77% of gates below 0.1). So the layer that does the most filtering is also the one where filtering matters the most. That makes intuitive sense: at this depth the model is working with raw token embeddings, and a lot of the attention head outputs are probably not very useful. The gate learned to suppress them, and undoing that suppression is catastrophic.

**Important note:** Early transformer layers are generally more sensitive to interference of any kind because the representations haven't built up redundancy yet. It was not tested whether the same kind of concentration would appear if normalization layers or feed-forward weights were broken at Layer 0, so it's unclear if this is about the gate specifically or just about early layers being fragile.

### 2.4 Killing One Head at a Time

<p align="center">
<img src="figures/head_ablation_top_layers.png" width="100%">
</p>

For the three most important layers (0, 1, 2), each head was killed one at a time by forcing its gate to zero to see what happened.

**Heads that actually improved when killed:**

| Layer | Head | Change in Perplexity |
|-------|------|---------------------|
| 1 | 1 | -0.06 (better without it) |
| 0 | 15 | -0.05 |
| 2 | 1 | -0.04 |
| 0 | 7 | -0.02 |
| 2 | 11 | -0.02 |
| 1 | 4 | -0.02 |
| 0 | 14 | -0.01 |
| 2 | 5 | -0.01 |

**Heads that hurt the most when killed:**

| Layer | Head | Change in Perplexity |
|-------|------|---------------------|
| 0 | 12 | +0.36 |
| 0 | 4 | +0.30 |
| 2 | 9 | +0.17 |
| 1 | 8 | +0.12 |
| 1 | 15 | +0.11 |

8 out of 48 heads tested (17%) could be killed with no quality loss or even a slight improvement. On the other end, the most important individual head (Layer 0, Head 12) only caused a 0.36 increase when killed. Compare that to Layer 0 as a whole, which caused a 148,407 increase when all its gates were opened. The takeaway: Layer 0's sensitivity comes from all its heads working together, not from any single head being a bottleneck.

**Caveat:** This is measured on one dataset only. A head that looks useless on WikiText-2 might be essential for code generation, math, or other tasks. Testing across many benchmarks would be needed before actually removing any heads, and ideally retraining to confirm.

---

## Summary of Observations

| What Was Seen | How Confident | Why |
|------------|---------------------|-----|
| Gates are mostly closed, with values clustered near 0 | **High** | Large sample of values, consistent with the paper |
| Early layers are more closed than middle layers | **High** | Clear pattern across all 28 layers |
| First token gets lower gate scores than other tokens | **Moderate** | Consistent across layers but only tested on a few prompts |
| Elementwise gating is sparser than headwise | **High** | Expected from the design, confirmed by the numbers |
| Overriding gates with fixed values severely hurts quality | **Moderate** | Expected for any co-trained parameter, and controls were not tested |
| Gate sensitivity is concentrated in the first few layers | **Moderate** | Could be generic early-layer fragility rather than gate-specific |
| Some heads can be killed without quality loss on WikiText-2 | **Low** | Single dataset, no statistics, no retraining |
| Different input types produce different gate profiles | **Very low** | One example per category, not meaningful statistically |

---

## Limitations and Open Questions

To be upfront about what's missing:

1. **No controls.** The gates were broken but no other parts of the model were broken in the same way. So when it's observed that "Layer 0's gate is critical," it might just be that Layer 0 is critical in general, gate or no gate. There's no way to know from this alone.

2. **One dataset.** Everything is measured on WikiText-2, which is about 25K tokens of English Wikipedia text. There's no way to know if these patterns hold on code, math, multilingual text, or anything else.

3. **One example per input type.** Section 1.5 uses a single sentence per category. Nothing statistical can be concluded from that. The plots are interesting to look at, but the numbers are not meaningful.

4. **No training.** Only the already-trained model was examined. There's no way to tell from this alone if the gate is architecturally valuable on its own; only that this particular model has learned to depend on it. The original paper answers the architectural question through proper training experiments.

5. **No error bars.** Every perplexity number is a single measurement. No bootstrapping or multiple splits were done to get confidence intervals.

6. **The co-training problem.** When a model is trained with gates, everything else in the model adapts to expect gated outputs. So of course breaking the gates breaks the model. The same thing would probably happen if the normalization layers were clamped to fixed values. This doesn't prove gating is special.

7. **One model size.** These are all 1B parameter models. The patterns might look completely different at 7B or 70B.

---

## What Would Make This Proper Research

To turn these observations into real findings:

1. **Run the same experiments on non-gate components** (normalization, feed-forward layers, etc.) to see if the patterns found are gate-specific or generic.

2. **Use hundreds of examples per input category** from established benchmarks, with proper statistical tests and confidence intervals.

3. **Evaluate on multiple datasets and tasks**, not just WikiText-2 perplexity.

4. **Train models from scratch** with the suggested modifications (only gating early layers, removing prunable heads, initializing gate biases differently) and see if they actually help.

5. **Test at multiple scales** (1B, 7B, 15B+) to see what's consistent and what changes.

---

## Reproducibility

| | |
|---|---|
| **Hardware** | Single NVIDIA T4 GPU (16 GB), Google Colab free tier |
| **Software** | `transformers==4.51.0`, PyTorch 2.x, Python 3.12 |
| **Models** | [QwQZh/gated_attention](https://huggingface.co/QwQZh/gated_attention) |
| **Eval data** | WikiText-2 test (`wikitext-2-raw-v1`), 50 x 512-token chunks |
| **Runtime** | Experiment 1: about 5 min, Experiment 2: about 30 min |

Notebooks:
- [`gated_attention_gate_score_analysis.ipynb`](gated_attention_gate_score_analysis.ipynb) : Experiment 1
- [`phase1_exp2_gate_ablation.ipynb`](phase1_exp2_gate_ablation.ipynb) : Experiment 2

---

## Citation

This analysis builds on:

```bibtex
@inproceedings{qiu2025gated,
  title     = {Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free},
  author    = {Zihan Qiu and Zekun Wang and Bo Zheng and Zeyu Huang and Kaiyue Wen and Songlin Yang and Rui Men and Le Yu and Fei Huang and Suozhi Huang and Dayiheng Liu and Jingren Zhou and Junyang Lin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2505.06708},
}
```
