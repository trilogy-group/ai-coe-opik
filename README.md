# Opik Comprehensive LLM Evaluation Suite

A robust framework for quantitative evaluation of state-of-the-art Large Language Models (LLMs) using Opik. This suite provides comprehensive metrics across multiple task types, enabling objective comparison of different models and prompt designs.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your Opik API key:
```
OPIK_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here
OPENROUTER_API_KEY=your_api_key_here
```

3. Configure models and metrics in `config.yaml`

4. Configure Opik at https://comet.com/opik -> Configuration -> AI Providers and add your API keys

## Usage

The POC supports two types of experiments:

### Basic Usage

```bash
python main.py -n <experiment_name> -t <experiment_type> [options]
```

### Options

- `-n, --name` - Experiment name (required)
- `-t, --type` - Experiment type: 'prompts' or 'models' (required)
- `-m, --model` - Model to use for prompt comparison (default: gpt-4o-mini)
- `-p, --prompt-version` - Prompt version for model comparison (default: 1)
- `--task` - Task type to evaluate: 'qa', 'summarization', 'code', 'chat', or 'all' (default: all)
- `--samples` - Number of samples to use from each dataset (default: 5)
- `--evaluator` - Model to use for LLM-based evaluations (default: o3-mini)

### Example 1: Compare Prompt Versions

Evaluate different prompt versions using a specific model:
```bash
python main.py -n prompt_eval -t prompts -m gpt-4o-mini --task qa
```

This will:
- Use the specified model (gpt-4o-mini)
- Evaluate all prompt versions for QA tasks
- Display results in the terminal

### Example 2: Compare Models

Evaluate different models using a specific prompt version:
```bash
python main.py -n model_eval -t models -p 2 --task all
```

This will:
- Use prompt version 2 for all task types
- Evaluate all configured models across all task types
- Display results in the terminal

## Features

- **Comprehensive Evaluation Framework**: Evaluate models across multiple task types (QA, summarization, code generation, chat)
- **Rich Metrics Library**: Quantitative metrics for correctness, relevance, fluency, and more
- **Flexible Configuration**: Easily customizable via YAML configuration
- **Detailed Terminal Output**: Results are displayed in a well-formatted table in the terminal
- **Robust Error Handling**: Automatic retry with rate limit handling
- **Modular Design**: Extensible architecture for adding new metrics and task types

### Metrics Categories

#### Heuristic Metrics
- **ConcisenessMetric**: Measures output brevity based on token length
- **ResponseTimeMetric**: Tracks model latency and performance
- **FormatComplianceMetric**: Verifies adherence to requested formats (JSON, etc.)
- **HallucinationMetric**: Detects fabricated information in outputs

#### LLM-as-Judge Metrics
- **FluencyMetric**: Evaluates linguistic quality and readability
- **ToxicityMetric**: Measures harmful or inappropriate content
- **ReasoningMetric**: Assesses logical reasoning and thought process
- **FactualAccuracyMetric**: Validates factual correctness using G-Eval

## Evaluation Datasets

The suite includes diverse datasets for comprehensive evaluation:

### Question Answering (QA)
- AI Engineering topics (reasoning distillation, inference-time scaling, etc.)
- Technical problem-solving scenarios
- Knowledge-based questions with factual answers

### Summarization
- Technical documentation summarization
- Research paper abstract generation
- News article condensation

### Code Generation
- Algorithm implementation tasks
- Function completion problems
- Bug fixing scenarios

### Chat Conversation
- Multi-turn dialogues
- Support conversation scenarios
- Creative writing prompts

## Prompt Variations

Each task type includes multiple prompt versions for comprehensive testing:

### QA Prompts
1. Basic prompt with clear structure
2. Expert-focused prompt with technical context
3. Detailed prompt with accuracy emphasis

### Summarization Prompts
1. Brief instruction focused on conciseness
2. Detailed guidance for comprehensive summaries
3. Format-specific instructions (bullet points, paragraphs, etc.)

### Code Generation Prompts
1. Basic code requirements
2. Detailed specifications with edge cases
3. Performance-optimized implementation requirements

### Chat Prompts
1. Open-ended conversation starters
2. Role-based interaction frameworks
3. Task-oriented dialogue scenarios

## Project Structure

- `main.py` - Core evaluation logic and CLI interface
- `config.py` - Configuration management using dataclasses
- `metrics.py` - Custom metrics implementation
- `prompts.py` - Prompt templates for different task types
- `datasets.py` - Sample datasets for evaluation tasks
- `config.yaml` - Model, metric, and task configuration
- `tests/` - Test suite for core functionality

## Documentation

For more information about Opik evaluation features, visit:
- [Opik Evaluation Overview](https://www.comet.com/docs/opik/evaluation/overview/)
- [Evaluation Concepts](https://www.comet.com/docs/opik/evaluation/concepts/)
- [Evaluation Metrics](https://www.comet.com/docs/opik/evaluation/metrics/)
- [Managing Prompts](https://www.comet.com/docs/opik/prompt_engineering/managing_prompts_in_code/)

# Opik Evaluation Framework

A robust framework for evaluating LLM performance using Opik.

## Architecture

The framework follows a modular design. For detailed architecture diagrams, see [Architecture Documentation](./architecture.md).

- **Configuration Management**: Typed configuration using dataclasses with task-specific settings
- **Evaluation Engine**: Handles model evaluation with retry logic and error handling
- **Metrics System**: Pluggable metrics with standardized interfaces for different task types
- **Results Management**: Structured result collection, analysis, and export options
- **Task System**: Task-specific dataset handlers and prompt templates

## Development

### Testing
```bash
pytest tests/
```

### Adding New Metrics
1. Create a new metric class in `metrics.py` implementing the `BaseMetric` interface
2. Add configuration in `config.yaml` under the appropriate metric category
3. Update the metric retrieval logic in `get_metrics_for_task()` if needed

### Adding New Task Types
1. Create dataset samples in `datasets.py`
2. Add prompt templates in `prompts.py`
3. Configure task-specific metrics in `config.yaml`
4. Update the `prepare_dataset()` function to handle the new task type
