# Energy & Alignment Testing Lab

A specialized testing environment for measuring how prompt injections and tool integrations affect LLM energy consumption and response alignment.

## Setup

### Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

## Features

### Energy Testing (`/energy`)
- **Energy Consumption Tracking**: Real-time measurement of energy usage in watt-hours
- **Carbon Footprint Calculation**: Estimate environmental impact
- **Benchmark Comparisons**: Compare against different hardware configurations
- **Dynamic Benchmark Switching**: Recalculate energy consumption with different benchmarks after tests
- **Custom Benchmarks**: Add your own hardware configurations with Wh/1000 tokens
- **Modification Impact Analysis**: See how injections and tools affect energy costs

### Alignment Testing (`/alignment`)
- **Alignment Metrics**: Goal adherence, consistency, relevance, factual accuracy
- **Risk Assessment**: Hallucination detection, injection bleed, tool interference
- **Quality Analysis**: Coherence, completeness, and misalignment detection
- **Modification Impact**: Understand how changes affect response quality

## Quick Start

### 1. Start the Energy/Alignment Lab
```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
python3 app_energy_alignment.py
```

### 2. Access the Interfaces
- **Energy Testing**: http://localhost:8001/energy
- **Alignment Testing**: http://localhost:8001/alignment

### 3. Run Tests
1. Select a model from the dropdown
2. Configure prompt injections and tool integrations
3. Click "Run [Energy/Alignment] Test"
4. View detailed metrics and analysis

## Architecture

### Core Modules
- `energy_tracker.py` - Energy consumption measurement and benchmarking
- `alignment_analyzer.py` - Response quality and alignment analysis
- `prompt_injection.py` - Prompt modification techniques
- `tool_integration.py` - Tool output integration strategies

### Data Handling
- **JSON Serialization**: All datetime objects are automatically converted to ISO strings
- **Session Persistence**: Energy readings and alignment scores are tracked across tests
- **Export Functionality**: Results can be exported to JSON for further analysis

### Independent Variables (What You Control)
- **Prompt Injection Types**:
  - None (baseline)
  - System modification (concise/detailed/creative/factual)
  - User augmentation (step-by-step, evidence-based, etc.)
  - Context injection
  - Chain-of-thought instructions

- **Tool Integration Methods**:
  - None (baseline)
  - Direct insertion (raw tool outputs)
  - Summarized insertion (compressed outputs)
  - Filtered insertion (relevant info only)
  - Staged insertion (incremental addition)

### Dependent Variables (What Gets Measured)

#### Energy Testing
- Total energy consumption (Wh)
- Energy per 1000 tokens
- Carbon footprint (gCO2)
- Modification overhead impact

#### Alignment Testing
- Goal adherence score (0-1)
- Response consistency
- Relevance to prompt
- Factual accuracy
- Hallucination risk
- Injection bleed effect
- Tool interference level

## Benchmark Assumptions & CO2 Conversion

### Default Energy Benchmarks
The system includes several pre-configured energy benchmarks based on measured power consumption during LLM inference:

- **Conservative Estimate** (0.50 Wh/1000 tokens): Baseline estimate for typical LLM workloads
- **NVIDIA RTX 4090** (0.75 Wh/1000 tokens): High-end gaming GPU
- **NVIDIA A100** (2.50 Wh/1000 tokens): Data center GPU (400W TDP)
- **Apple M2** (0.15 Wh/1000 tokens): Mobile chip (20W TDP)

### Custom Benchmarks
You can add your own benchmarks by specifying:
- **Name**: Descriptive identifier
- **Wh/1000 tokens**: Measured energy consumption per 1000 tokens
- **Description**: Hardware specifications
- **Source**: Where the measurement came from

### CO2 Conversion Methodology
- **Global Average**: 400 gCO2/kWh (IEA 2023 Global Energy Review)
- **Calculation**: Energy (Wh) × Carbon Intensity (gCO2/kWh) ÷ 1000
- **Scope**: Grid electricity only (no embodied carbon in hardware)

### Benchmark Sources & Assumptions
- **Conservative Estimate**: Based on academic papers and industry reports assuming efficient GPU utilization
- **Hardware Benchmarks**: Measured power consumption during actual LLM inference workloads
- **Assumptions**: 
  - Steady-state power consumption (no idle power)
  - Efficient model serving (not training)
  - Modern hardware with optimized inference

## Links & References
- [IEA Global Energy Review 2023](https://www.iea.org/reports/global-energy-review-2023)
- [MLCO2 Impact Calculator](https://mlco2.github.io/impact/)
- [Hugging Face CO2 Emissions](https://huggingface.co/docs/transformers/model_doc/auto#transformers.TFPreTrainedModel.get_memory_footprint)
- [Energy and Policy Considerations for Deep Learning](https://arxiv.org/abs/1906.02243)

## Example Use Cases

### Energy Efficiency Testing
- Compare energy cost of "concise" vs "detailed" system prompts
- Measure overhead of different tool integration methods
- Evaluate energy impact of CoT instructions

### Alignment Quality Testing
- Test if tool integration improves factual accuracy
- Measure how system modifications affect goal adherence
- Assess risk of hallucinations with different prompt styles

## Output Data

Both interfaces export comprehensive test data including:
- Raw metrics and scores
- Modification details
- Session summaries
- Trend analysis over multiple tests

## Integration with Main Lab

This specialized testing environment complements the main LLM Behavior Lab:
- **Main Lab**: Multi-model comparison with basic metrics
- **Energy Lab**: Deep energy consumption analysis
- **Alignment Lab**: Detailed response quality assessment

All labs share the same Ollama backend and can use the same models.
