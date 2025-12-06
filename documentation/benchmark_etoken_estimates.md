# E-Token Benchmark Estimates

## Benchmark Sources and Comparison (Hugging Face vs Jegham et al.)

Two main external benchmark sources are used to derive e-token style energy metrics:

- **Hugging Face AI Energy Score (HF)** – direct GPU energy measurements on fixed hardware.
- **Jegham et al. (2025)** – API-level benchmarks with **modelled** energy estimates based on assumed infrastructure.

At a high level:

- **HF AI Energy Score**
  - Measures **actual GPU energy (Wh)** for open-weight models on an NVIDIA H100 80GB GPU.
  - Reports **Wh per 1,000 queries**, which we convert to **Wh per 1,000 input tokens** and then to **e-token (input)**.
  - Reflects on-the-ground hardware measurements under a specific benchmark workload.

- **Jegham et al. (2025)** – ["Large Language Models in the Real World: Energy, Cost and Latency"](https://arxiv.org/html/2505.09598v6)
  - Runs both **closed and open-source models via public APIs**.
  - Observes usage (token counts, latency, pricing) and combines this with **assumed hardware, utilisation, environment and PUE** to derive **estimated Wh per query**.
  - Uses three synthetic prompt sizes with a standardised **1,000 input / 1,000 output** configuration (plus two other I/O profiles), which aligns naturally with the **e-token** methodology for both input and output.

The key methodological difference is:

- HF values are **ground-truth GPU energy readings** for a known benchmark and hardware.
- Jegham values for open-source models are **proxies** based on **inferred infrastructure plus modelling assumptions**:
  - **Not** from watt-meters or GPU power logs.
  - **Not guaranteed** to match the true energy of any particular deployment.
  - Their validity depends on assumed GPU type, utilisation, environment, PUE, etc.

Because of this, Jegham et al.’s energy numbers – especially for open-source models – should be treated as **plausible proxies, not ground-truth measurements**. At the same time, they:

- Provide **coverage for many closed models** that do not appear in the Hugging Face dataset.
- Apply a **consistent 1,000 input / 1,000 output** workload across models, directly compatible with the e-token framework for **both** input and output.

In this lab, HF and Jegham benchmarks are therefore **complementary**:

- HF anchors **physically measured energy** for open models.
- Jegham extends coverage to **closed models and additional I/O regimes**, under clearly modelled assumptions.

## Hugging Face Benchmarks

The LLM Energy Lab derives per-model energy metrics from the
[AI Energy Score](https://huggingface.github.io/AIEnergyScore/) project by
Hugging Face. The goal is to express the AI Energy Score benchmark in a form
that can be consumed programmatically (JSON) and, optionally, normalised per
token rather than per query.

### Data sources

We rely on three primary sources:

1. **Methodology documentation**

   The AI Energy Score methodology defines the benchmark tasks, datasets and
   energy metrics. In particular:

   - For each task, a custom dataset of **1,000 data points (queries)** is
     created by sampling from three existing datasets.
   - Table 1 in the documentation reports the **total number of input tokens**
     for each task. For the *Text generation and reasoning* task, the table
     lists `369,139` input tokens for the 1,000-query dataset.
   - The primary energy metric is defined as **GPU energy consumption in
     watt-hours per 1,000 queries**, where GPU energy is summed across
     `preprocess`, `prefill`, and `decode` and averaged over 10 runs.

   Source: AI Energy Score methodology documentation.

2. **Benchmark datasets**

   The text-generation dataset used for evaluation is published as the
   `AIEnergyScore/text_generation` dataset on the Hugging Face Hub. The dataset
   card shows a `train` split with `1k rows`, which aligns with the “1,000
   data points” described in the methodology.

   We therefore interpret the Table 1 "Input Tokens" value (`369,139`) as the
   **total number of input tokens across the 1,000 queries** used to benchmark
   text-generation and reasoning models.

3. **Leaderboard energy CSVs and app**

   The AI Energy Score leaderboard is implemented as the
   `AIEnergyScore/Leaderboard` Space. Per-task energy data are stored in CSV
   files under `data/energy/`, such as:

   - `data/energy/text_generation.csv`

   Each CSV has the columns:

   - `model` – Hugging Face model id
   - `total_gpu_energy` – average GPU energy in Wh **per query**
   - `energy_score` – star rating (1–5)
   - `class` – hardware class (A/B/C)
   - `test date` – date of the benchmark run

   The Space’s `app.py` reads these CSVs and multiplies `total_gpu_energy` by
   1,000 to produce `gpu_energy_numeric`, which is displayed in the UI as
   **“GPU Energy (Wh) per 1k Queries”**.

### Metric definitions

For each model in `data/energy/text_generation.csv` we define:

- **Total GPU energy per query (Wh)**

  ```text
  total_gpu_energy_per_query_wh = total_gpu_energy  # from CSV
  ```

- **GPU energy (Wh) per 1,000 queries**

  ```text
  wh_per_1000_queries = total_gpu_energy_per_query_wh * 1000
  ```

  This matches the value shown in the AI Energy Score leaderboard for the
  text-generation task.

- **GPU energy (Wh) per 1,000 input tokens (optional)**

  The AI Energy Score methodology states that the Text generation and
  reasoning task’s dataset consists of 1,000 queries and 369,139 input
  tokens. We therefore normalise `wh_per_1000_queries` by the token count:

  ```text
  input_tokens_per_1000_queries = 369139
  wh_per_1000_tokens_factor = 1000 / input_tokens_per_1000_queries  # ≈ 0.0027090066
  wh_per_1000_tokens = wh_per_1000_queries * wh_per_1000_tokens_factor
  ```

In the benchmark JSON used by this lab (`data/benchmark_data/hugging_face.json`), this factor is stored as `wh_per_1000_tokens_factor`, and each model row includes a precomputed
`wh_per_1000_input_etokens` value, which applies the same factor to that model's `wh_per_1000_queries`. These correspond directly to the **e-token (input)** metric derived above.

### What energy is being assigned to each token?

In line with the AI Energy Score methodology, `wh_per_1000_queries` includes:

- preprocess energy
- prefill (input / context) energy
- decode (generation) energy

All measured on an NVIDIA H100 80GB GPU, averaged over 10 full runs of the
1,000-query dataset.

When we convert from Wh per 1,000 queries to Wh per 1,000 tokens, we are
amortising the complete GPU energy of the end-to-end inference process
(preprocess, prefill, and decode) over the input tokens in the benchmark
dataset. This implements the e-token (input) metric. In other words:

> `wh_per_1000_tokens` answers: “On this benchmark, how many watt-hours of GPU
> energy does the model consume for every 1,000 input tokens it processes,
> including all overhead and the generation of its outputs?”

This implicitly bakes in both:

- the average input length distribution of the benchmark, and
- the average output lengths of the models on that benchmark.

### Assumptions and limitations

- **Benchmark-specific token mix** – The normalisation is valid for the
  AI Energy Score text-generation dataset (369,139 tokens over 1,000 queries).
  Different workloads with different prompt lengths or styles may lead to
  different per-token energy.
- **Output tokens are not counted explicitly** – The “Input Tokens” values in
  Table 1 refer only to input tokens. Decode energy is therefore implicitly
  amortised onto input tokens. We do not attempt to separate prefill vs decode
  energy at the per-token level.
- **Linear scaling assumption** – We assume that average energy per query
  scales linearly with the number of queries when we rescale from per 1,000
  queries to per 1,000 tokens. This is consistent with how the AI Energy Score
  defines its metrics (Wh per 1,000 queries), but real systems may exhibit
  some nonlinearities due to fixed overheads.

Despite these limitations, this normalisation provides a consistent way to
compare models’ energy intensity per unit of input text using the AI
Energy Score benchmark as a reference.

### Why this is legitimate

- It preserves the physics (Wh).
- It standardizes across models and tasks.
- It reflects *true* inference energy.
- It matches user intuition (“cost per input token”).
- It enables comparisons across models and hardware.

## Jegham et al. Benchmarks

Jegham et al. (2025), ["Large Language Models in the Real World: Energy, Cost and Latency"](https://arxiv.org/html/2505.09598v6), provide a complementary view of model energy by running a wide set of **closed and open-source models** through public APIs and estimating energy per query under a standardised infrastructure assumption.

### Table 4 – Energy per Query Across Prompt Sizes

Table 4 of the paper reports **mean ± standard deviation** of **energy consumption in watt-hours (Wh)** for three prompt sizes:

- **100 input / 300 output tokens**
- **1,000 input / 1,000 output tokens**
- **10,000 input / 1,500 output tokens**

The table below reproduces those values:

| Model                   | Energy (100 in / 300 out) Wh | Energy (1k in / 1k out) Wh | Energy (10k in / 1.5k out) Wh |
|-------------------------|------------------------------|----------------------------|--------------------------------|
| GPT-4.1                 | 0.871 ± 0.302                | 3.161 ± 0.515              | 4.833 ± 0.650                  |
| GPT-4.1 mini            | 0.450 ± 0.081                | 1.545 ± 0.211              | 2.122 ± 0.348                  |
| GPT-4.1 nano            | 0.207 ± 0.047                | 0.575 ± 0.108              | 0.827 ± 0.094                  |
| o4-mini (high)          | 3.649 ± 1.468                | 7.380 ± 2.177              | 7.237 ± 1.674                  |
| o3                      | 1.177 ± 0.224                | 5.153 ± 2.107              | 12.222 ± 1.082                 |
| o3-mini (high)          | 3.012 ± 0.991                | 6.865 ± 1.330              | 5.389 ± 1.183                  |
| o3-mini                 | 0.674 ± 0.015                | 2.423 ± 0.237              | 3.525 ± 0.168                  |
| o1                      | 2.268 ± 0.654                | 4.047 ± 0.497              | 6.181 ± 0.877                  |
| o1-mini                 | 0.535 ± 0.182                | 1.547 ± 0.405              | 2.317 ± 0.530                  |
| GPT-4o (Mar ’25)        | 0.423 ± 0.085                | 1.215 ± 0.241              | 2.875 ± 0.421                  |
| GPT-4o mini             | 0.577 ± 0.139                | 1.897 ± 0.570              | 3.098 ± 0.639                  |
| GPT-4 Turbo             | 1.699 ± 0.355                | 5.940 ± 1.441              | 9.877 ± 1.304                  |
| GPT-4                   | 1.797 ± 0.259                | 6.925 ± 1.553              | —                              |
| DeepSeek-R1 (DS)*       | 19.251 ± 9.449               | 24.596 ± 9.400             | 29.078 ± 9.725                 |
| DeepSeek-V3 (DS)*       | 2.777 ± 0.223                | 8.864 ± 0.724              | 13.162 ± 1.126                 |
| DeepSeek-R1 (AZ)†       | 2.353 ± 1.129                | 4.331 ± 1.695              | 7.410 ± 2.159                  |
| DeepSeek-V3 (AZ)†       | 0.742 ± 0.125                | 2.165 ± 0.578              | 3.696 ± 0.221                  |
| Claude-3.7 Sonnet       | 0.950 ± 0.040                | 2.989 ± 0.201              | 5.671 ± 0.302                  |
| Claude-3.5 Sonnet       | 0.973 ± 0.066                | 3.638 ± 0.256              | 7.772 ± 0.345                  |
| Claude-3.5 Haiku        | 0.975 ± 0.063                | 4.464 ± 0.283              | 8.010 ± 0.338                  |
| LLaMA-3-8B              | 0.108 ± 0.002                | 0.370 ± 0.005              | —                              |
| LLaMA-3-70B             | 0.861 ± 0.022                | 2.871 ± 0.094              | —                              |
| LLaMA-3.1-8B            | 0.052 ± 0.008                | 0.172 ± 0.015              | 0.443 ± 0.028                  |
| LLaMA-3.1-70B           | 1.271 ± 0.020                | 4.525 ± 0.053              | 19.183 ± 0.560                 |
| LLaMA-3.1-405B          | 2.226 ± 0.142                | 9.042 ± 0.385              | 25.202 ± 0.526                 |
| LLaMA-3.2 1B            | 0.109 ± 0.013                | 0.342 ± 0.025              | 0.552 ± 0.059                  |
| LLaMA-3.2 3B            | 0.143 ± 0.006                | 0.479 ± 0.017              | 0.707 ± 0.020                  |
| LLaMA-3.2-vision 11B    | 0.078 ± 0.021                | 0.242 ± 0.071              | 1.087 ± 0.060                  |
| LLaMA-3.2-vision 90B    | 1.235 ± 0.054                | 4.534 ± 0.448              | 6.852 ± 0.780                  |
| LLaMA-3.3 70B           | 0.237 ± 0.023                | 0.760 ± 0.079              | 1.447 ± 0.188                  |

*DS – DeepSeek host, †AZ – Microsoft Azure host.

### How Jegham et al. Relate to e-Tokens

Jegham et al. report **Wh per query** for fixed input/output sizes. For the
**1,000 input / 1,000 output** setting:

- Each query implicitly corresponds to a **2,000-token workload**.
- Dividing Wh by 2,000 tokens and scaling to 1,000 tokens yields an
  **e-token-style intensity** that is directly comparable across models.
- Because inputs and outputs are both fixed and symmetric (1k / 1k), the same
  per-1,000-token value can be read as both an **input-normalised** and an
  **output-normalised** e-token proxy.

However, unlike HF:

- These are **not direct hardware measurements**.
- Open-source models are evaluated via **public APIs plus inferred hardware
  assumptions**, rather than on a known benchmark rig with power meters.

So Jegham et al. values should be treated as **modelled proxies**:

- They offer **broad, comparable coverage** of many closed models that are
  absent from the HF dataset.
- They implement a **standardised 1k/1k token query** that aligns cleanly with
  the e-token framework.
- But they are **not ground truth** for any specific deployment; real-world
  energy will depend on the actual hardware, utilisation profile, and data
  centre efficiency where the model is hosted.
