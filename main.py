import os
from dotenv import load_dotenv
import opik
from opik.evaluation import evaluate_prompt
from opik.evaluation.metrics import Hallucination, AnswerRelevance, GEval
import yaml
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import time
import argparse
from prompts import get_prompts, AI_ENGINEERING_SAMPLES, GEVAL_TASK_INTRO, GEVAL_CRITERIA

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run model evaluations')
    parser.add_argument('--name', '-n', required=True, help='Experiment set name')
    return parser.parse_args()

def load_config() -> Dict:
    """Load configuration from yaml file."""
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def evaluate_model(model: str, dataset: opik.Dataset, prompt: opik.Prompt, metrics: List, exp_name: str) -> Dict:
    """Evaluate a single model."""
    try:
        print(f"\nEvaluating model: {model}")
        start_time = time.time()
        
        # Convert prompt to messages format
        messages = [
            {
                "role": "user",
                "content": prompt.prompt
            }
        ]
        
        result = evaluate_prompt(
            dataset=dataset,
            messages=messages,
            model=model,
            scoring_metrics=metrics,
            experiment_name=exp_name
        )
        
        duration = time.time() - start_time
        return {
            "model": model,
            "result": result,
            "duration": duration
        }
    except Exception as e:
        print(f"Error evaluating model {model}: {str(e)}")
        return {
            "model": model,
            "error": str(e)
        }

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config()
    
    # Initialize Opik client
    opik_client = opik.Opik()

    # Get versioned prompts
    prompts = get_prompts()
    qa_prompts = prompts["ai_engineering"]
    summary_prompts = prompts["summarization"]

    # Create or get datasets
    qa_dataset = opik_client.get_or_create_dataset("AI Engineering QA Dataset")
    summary_dataset = opik_client.get_or_create_dataset("AI Engineering Summary Dataset")

    # Convert QA prompts to dataset format
    qa_samples = [
        {
            "question": prompt.prompt.split("Question: ")[1].split("\n")[0],
            "expected_output": prompt.prompt.split("Expected Answer: ")[1],
            "context": prompt.prompt.split("Context: ")[1].split("\n")[0]
        }
        for prompt in qa_prompts
    ]

    # Convert summary prompts to dataset format
    summary_samples = [
        {
            "text": prompt.prompt.split("text about AI engineering:\n\n")[1].split("\n\nSummarize")[0],
            "task": "summarize"
        }
        for prompt in summary_prompts
    ]

    # Insert samples into datasets
    qa_dataset.insert(qa_samples)
    summary_dataset.insert(summary_samples)

    # Create metric instances based on config
    metrics = []
    for metric in config['metrics']:
        if metric['name'] == 'hallucination' and metric['enabled']:
            metrics.append(Hallucination())
        elif metric['name'] == 'answer_relevance' and metric['enabled']:
            metrics.append(AnswerRelevance())
        elif metric['name'] == 'g_eval_summary' and metric['enabled']:
            metrics.append(GEval(
                task_introduction=GEVAL_TASK_INTRO,
                evaluation_criteria=GEVAL_CRITERIA
            ))

    # Get models from config
    models = [model['name'] for model in config['models']]

    # Run parallel evaluations for both QA and summarization
    print("\nStarting parallel model evaluations...")
    with ThreadPoolExecutor() as executor:
        # QA evaluations
        qa_futures = [
            executor.submit(evaluate_model, model, qa_dataset, qa_prompts[0], 
                          [m for m in metrics if not isinstance(m, GEval)], 
                          f"{args.name}-qa-{model}")
            for model in models
        ]
        
        # Summarization evaluations
        summary_futures = [
            executor.submit(evaluate_model, model, summary_dataset, summary_prompts[0],
                          [m for m in metrics if isinstance(m, GEval)],
                          f"{args.name}-summary-{model}")
            for model in models
        ]
        
        all_results = [future.result() for future in qa_futures + summary_futures]

    # Print comparative results
    print("\nComparative Results:")
    print("===================")
    for result in all_results:
        if "error" in result:
            print(f"\nModel: {result['model']}")
            print(f"Error: {result['error']}")
        else:
            print(f"\nModel: {result['model']}")
            print(f"Duration: {result['duration']:.2f} seconds")

if __name__ == "__main__":
    main() 