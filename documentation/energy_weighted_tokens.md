# Energy-Weighted Tokens (e-Tokens)

## Summary

- **e-tokens** are standardized, comparable, energy-normalized tokens.
- They calculate the full energy consumed from the moment a query is received by an LLM to the moment its last output is generated, and assign it to either the input or the output tokens, amortising the net energy footprint for comparisons.
- **e-token (input)** expresses net energy per unit of input intention.
- **e-token (output)** expresses net energy per unit of generated content.
- Both derive from the same physical measurement (Wh).
- Together, they give a **complete** view of model and task energy efficiency.

## Overview

This repository introduces **energy-weighted tokens ("e-tokens")**, a standardized unit for expressing total inference energy in a way that allows:

- **Fair comparison across models**
- **Fair comparison across tasks**
- **Fair comparison across hardware**
- **Task intensity analysis**
- **Prediction of relative energy cost at scale**

E-tokens convert raw energy measurements (Wh) into a **token-normalised metric**, enabling apples-to-apples comparison even when tasks generate very different volumes of text.

There are two complementary variants:

- **e-token (input)** — normalizes total energy by input tokens
- **e-token (output)** — normalizes total energy by output tokens

Both represent *the same physical energy*, simply expressed through different denominators.

---

## What Is an e-Token?

An **e-token** is an *energy-weighted token unit*:

> **A normalized representation of total inference energy (preprocess + prefill + decode) expressed per 1,000 tokens, using either input or output tokens as the denominator.**

It is NOT a claim about "the energy cost of one input token" or "one output token."

Rather, e-tokens express **overall energy intensity** under a chosen perspective:

- **Input-normalized perspective** → energy cost per 1000 query tokens
- **Output-normalized perspective** → energy cost per 1000 response tokens

In reality, input tokens correspond to the energy consumption of the prefill phase of LLM inference, while output tokens correspond to the energy consumption of the decode phase.

### Prefill Phase (Input → Key/Value Cache)

- Fully parallelizable across tokens
- Much cheaper per token
- Dominated by large matrix multiplications
- Utilizes GPU throughput efficiently

$$
\text{energy/token (prefill)} \ll \text{energy/token (decode)}
$$

### Decode Phase (Autoregressive Generation)

- **Serial** — each output token depends on previous outputs
- High latency
- KV cache grows with sequence length
- Expensive per token

$$
\text{energy/token (decode)} \gg \text{energy/token (prefill)}
$$

This means that tying energy to a given token count is problematic, since the energy cost of 1000 input tokens is very different than the energy cost of 1000 output tokens.

E-tokens reconcile these differences by assigning the complete energy cost to just the input tokens or just the output tokens, normalising Wh/1000 tokens for the purposes of measurements and comparisons.

- Input normalization spreads decode cost over inputs.
- Output normalization spreads the same energy over generated tokens.

Both are correct and both transmit the total energy consumption of a query and its response, but they illuminate different structural aspects of inference.

---

## Why e-Tokens Matter

E-tokens provide:

- ✔ A universal scale for comparing models
- ✔ A universal scale for comparing tasks
- ✔ A universal scale for comparing hardware
- ✔ A way to express task intensity ("how heavy is this task?")
- ✔ A predictive metric for relative cost at scale

Examples:

- Two models perform the same task on the same hardware, with the same input and output tokens. The model with the lower e-token (input) value will be more energy-efficient.
- One model performs two tasks on the same hardware, each task yielding a different e-token value. The task with the lower e-token (input) value will be more energy-efficient.
- The same model runs the same task with the same input and output tokens on different hardware. The hardware yielding the lower e-token (input) value will be more energy-efficient.

In each of these examples the e-token value is a **direct metric** of the total energy consumed by the query and its response.

**E-tokens turn raw Wh measurements into human-interpretable, model-comparable quantities**.

---

## Mathematical Definitions

Let:

$$
E = \text{total inference energy (Wh), including preprocess + prefill + decode}
$$

$$
T_{\text{input}} = \text{number of input tokens}
$$

$$
T_{\text{output}} = \text{number of output tokens}
$$

### e-token (input)

Energy per 1,000 input-normalized e-tokens:

$$
\text{e-token (input)} = \left( \frac{E}{T_{\text{input}}} \right) \times 1000
$$

### e-token (output)

Energy per 1,000 output-normalized e-tokens:

$$
\text{e-token (output)} = \left( \frac{E}{T_{\text{output}}} \right) \times 1000
$$

### Relationship Between the Two Metrics

Because both use the same numerator (total energy):

$$
\frac{\text{e-token (input)}}{\text{e-token (output)}} = \frac{T_{\text{output}}}{T_{\text{input}}}
$$

This identity is a feature, not a flaw — it means:

- e-token (input) emphasizes the **user-driven input burden**
- e-token (output) emphasizes the **model-generated work**

Neither is "more correct."
Both are *perspectives* on the same total energy.

---

## Visual Intuition: How the Normalisation Behaves

### ASCII Chart: Energy vs Input/Output Normalisation

```
                SAME ENERGY (E)
                ───────────────────────
                    |          |
        Normalize by inputs   Normalize by outputs
                    |          |
     e-token (input)         e-token (output)
             ↑                    ↓
      Larger value            Smaller value
        when output            when output
        dominates               dominates
```

Interpretation:

- If output ≫ input, input-normalized e-tokens will be larger.
- If input ≫ output, output-normalized e-tokens will be larger.

Both remain perfectly valid expressions of the *same* energy.

---

## Comparison Table (Different I/O Ratios)

Assume **same energy (E)** for all tasks.

| I/O Ratio (Input : Output) | Example Task          | e-token (input) | e-token (output) | Interpretation                                             |
| -------------------------- | --------------------- | --------------- | ---------------- | ---------------------------------------------------------- |
| **1 : 0.1**                | Classification        | LOW             | HIGH             | Output is tiny → energy per output token appears larger    |
| **1 : 1**                  | Chat, summarization   | MODERATE        | MODERATE         | Balanced → both metrics converge                           |
| **1 : 5**                  | Reasoning, expansions | HIGH            | LOW              | Many outputs → energy per input token appears higher       |
| **1 : 50**                 | Creative writing      | VERY HIGH       | VERY LOW         | Huge decode → cost looks large per input, small per output |

Again, **neither metric is distorted** — they simply highlight different aspects of the task.

---

## When to Use Each Metric

| Use Case                             | Best Metric          | Why                                          |
| ------------------------------------ | -------------------- | -------------------------------------------- |
| API pricing, billing, user budgeting | **e-token (input)**  | Users control input size                     |
| Model architecture comparison        | **e-token (output)** | Decode cost reveals architectural efficiency |
| RAG, search, embedding-heavy tasks   | **e-token (input)**  | Prefill dominates                            |
| Creative generation                  | **e-token (output)** | Expanders generate lots of tokens            |
| Task intensity analysis              | **e-token (input)**  | Standardised across tasks                    |
| Benchmark ranking                    | **Both**             | Complementary views                          |

---

## Strengths

### e-tokens retain full physical energy information

All amortization preserves total Wh exactly.

### e-tokens are comparable across tasks

Because the denominator is explicit and consistent.

### e-tokens enable cost-to-solve analysis

You can directly compare:

- "Which task is more computationally intense?"
- "Which model is more efficient for this input size?"

### e-tokens enable meaningful hardware comparison

Same model + task → differences reflect **hardware efficiency**.

### Suitable for order-of-magnitude predictions

While not exact predictors, e-tokens reliably preserve relative ranking:

- If model A uses half the Wh of model B in the benchmark,
- It will *roughly* use half the Wh on most tasks of similar IO character.

---

## Limitations (Conceptual and Methodological)

- E-tokens are not "energy per intrinsic token"; they express normalized **total inference energy**, not "the energy cost of a single token."
- Different prompt distributions produce different e-token values.
- Model rankings may shift under extreme I/O.
- Hugging Face benchmarks only report input tokens, so e-token (output) requires measuring or estimating output tokens separately.
- Prefill and decode physics differ; linear scaling is approximate, not exact.
