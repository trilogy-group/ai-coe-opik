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

Run the evaluation with:
```bash
python main.py -n <experiment_name>
```

For example:
```bash
python main.py -n ai_eng_eval_1
```

The script will:
1. Load model configurations from config.yaml
2. Create versioned prompts from AI Engineering evaluation samples
3. Run parallel evaluations for each configured model
4. Display comparative results for all models

## Features

- Parallel model evaluation
- Configurable models and metrics
- Versioned prompts using Opik Prompt Library
- AI Engineering focused evaluation dataset
- Multiple evaluation metrics
- Comparative performance analysis

## Evaluation Dataset

The evaluation covers key AI Engineering topics:
- Reasoning distillation
- Inference-time scaling
- GraphRAG architecture
- LLM performance evaluation
- Multimodal model processing

## Documentation

For more information about Opik evaluation features, visit:
- [Opik Evaluation Overview](https://www.comet.com/docs/opik/evaluation/overview/)
- [Evaluation Concepts](https://www.comet.com/docs/opik/evaluation/concepts/)
- [Evaluation Metrics](https://www.comet.com/docs/opik/evaluation/metrics/)
- [Managing Prompts](https://www.comet.com/docs/opik/prompt_engineering/managing_prompts_in_code/)