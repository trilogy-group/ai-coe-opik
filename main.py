import os
from dotenv import load_dotenv
import opik
from opik.evaluation import evaluate_prompt
from opik.evaluation.metrics import Hallucination, AnswerRelevance
import yaml
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import time
import argparse
from prompts import get_prompts, AI_ENGINEERING_SAMPLES

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
                "content": prompt.prompt  # Use the versioned prompt as user message
            }
        ]
        
        result = evaluate_prompt(
            dataset=dataset,
            messages=messages,
            model=model,
            scoring_metrics=metrics,
            experiment_name=f"{exp_name}-{model}"
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
    ai_engineering_prompts = prompts["ai_engineering"]

    # Create or get a dataset
    dataset = opik_client.get_or_create_dataset("AI Engineering Dataset")

    # Convert prompts to dataset format
    dataset_samples = [
        {
            "question": prompt.prompt.split("Question: ")[1].split("\n")[0],
            "expected_output": prompt.prompt.split("Expected Answer: ")[1],
            "context": prompt.prompt.split("Context: ")[1].split("\n")[0]
        }
        for prompt in ai_engineering_prompts
    ]

    # Insert samples into dataset
    dataset.insert(dataset_samples)

    # Create metric instances based on config
    metrics = []
    if config['metrics'][0]['enabled']:
        metrics.append(Hallucination())
    if config['metrics'][1]['enabled']:
        metrics.append(AnswerRelevance())

    # Get models from config
    models = [model['name'] for model in config['models']]

    # Run parallel evaluations
    print("\nStarting parallel model evaluations...")
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(evaluate_model, model, dataset, ai_engineering_prompts[0], metrics, args.name)
            for model in models
        ]
        
        results = [future.result() for future in futures]

    # Print comparative results
    print("\nComparative Results:")
    print("===================")
    for result in results:
        if "error" in result:
            print(f"\nModel: {result['model']}")
            print(f"Error: {result['error']}")
        else:
            print(f"\nModel: {result['model']}")
            print(f"Duration: {result['duration']:.2f} seconds")

if __name__ == "__main__":
    main() 