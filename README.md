# Opik Evaluation POC

This is a proof of concept implementation of Comet Opik for prompt & LLM evaluation. It demonstrates how to evaluate and compare multiple LLM responses on AI Engineering topics using predefined metrics like hallucination and answer relevance.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your Opik API key:
```
OPIK_API_KEY=your_api_key_here
```

3. Configure models and metrics in `config.yaml`

## Usage

The POC supports two types of experiments:

### 1. Compare Prompt Versions

Evaluate different prompt versions using a specific model:
```bash
python main.py -n <experiment_name> -t prompts [-m <model_name>]
```

Example:
```bash
python main.py -n prompt_eval_1 -t prompts -m gpt-4o-mini
```

This will:
- Use the specified model (defaults to gpt-4o-mini)
- Evaluate all prompt versions
- Compare performance across different prompt designs
- Log results to the "POC" project in Opik

### 2. Compare Models

Evaluate different models using a specific prompt version:
```bash
python main.py -n <experiment_name> -t models [-p <prompt_version>]
```

Example:
```bash
python main.py -n model_eval_1 -t models -p 1
```

This will:
- Use the specified prompt version (defaults to version 1)
- Evaluate all configured models
- Compare model performance
- Log results to the "POC" project in Opik

## Features

- Parallel model evaluation with rate limiting protection
- Configurable models and metrics
- Versioned prompts using Opik Prompt Library
- AI Engineering focused evaluation dataset
- Multiple evaluation metrics:
  - Hallucination detection
  - Answer relevance scoring
  - G-Eval summary quality assessment
- Comparative performance analysis
- Automatic retry with backoff for rate limits

## Evaluation Dataset

The evaluation covers key AI Engineering topics:
- Reasoning distillation
- Inference-time scaling
- GraphRAG architecture
- LLM performance evaluation
- Multimodal model processing

## Prompt Versions

The POC includes multiple prompt versions for each task:
1. Basic prompt with clear structure
2. Expert-focused prompt with technical context
3. Detailed prompt with accuracy emphasis

## Documentation

For more information about Opik evaluation features, visit:
- [Opik Evaluation Overview](https://www.comet.com/docs/opik/evaluation/overview/)
- [Evaluation Concepts](https://www.comet.com/docs/opik/evaluation/concepts/)
- [Evaluation Metrics](https://www.comet.com/docs/opik/evaluation/metrics/)
- [Managing Prompts](https://www.comet.com/docs/opik/prompt_engineering/managing_prompts_in_code/)