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
from prompts import get_prompts, AI_ENGINEERING_SAMPLES, GEVAL_TASK_INTRO, GEVAL_CRITERIA, AI_ENGINEERING_VERBOSE
from functools import wraps
from litellm.exceptions import RateLimitError
import random

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run model evaluations')
    parser.add_argument('--name', '-n', required=True, help='Experiment set name')
    parser.add_argument('--type', '-t', choices=['prompts', 'models'], required=True, 
                      help='Type of experiment: compare prompts or models')
    parser.add_argument('--model', '-m', default='gpt-4o-mini',
                      help='Model to use for prompt comparison (default: gpt-4o-mini)')
    parser.add_argument('--prompt-version', '-p', default='1',
                      help='Prompt version to use for model comparison (default: 1)')
    return parser.parse_args()

def load_config() -> Dict:
    """Load configuration from yaml file."""
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def retry_on_rate_limit(max_retries=3):
    """Decorator to retry function on rate limit errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    
                    # Extract wait time from error message
                    try:
                        import re
                        time_match = re.search(r'try again in (\d+\.?\d*)s', str(e))
                        wait_time = float(time_match.group(1)) if time_match else 1
                    except:
                        wait_time = 1
                    
                    print(f"\nRate limit reached. Retrying in {wait_time:.2f} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@retry_on_rate_limit()
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
            experiment_name=exp_name,
            project_name="POC",
            prompt=prompt
        )
        
        duration = time.time() - start_time
        return {
            "model": model,
            "result": result,
            "duration": duration,
            "prompt_name": prompt.name
        }
    except Exception as e:
        print(f"Error evaluating model {model}: {str(e)}")
        return {
            "model": model,
            "error": str(e)
        }

def evaluate_prompt_versions(dataset, prompts, metrics, model, exp_name, task_type):
    """Evaluate different prompt versions using a single model."""
    results = []
    print(f"\nEvaluating prompt versions using model: {model}")
    
    for prompt in prompts:
        result = evaluate_model(
            model, 
            dataset, 
            prompt,
            metrics,
            f"{exp_name}-{task_type}-prompt-v{prompt.name.split('-v')[1]}"
        )
        results.append(result)
    
    return results

def evaluate_models(dataset, prompt, metrics, models, exp_name, task_type):
    """Evaluate different models using a single prompt."""
    results = []
    print(f"\nEvaluating models using prompt version: {prompt.name}")
    
    for model in models:
        result = evaluate_model(
            model, 
            dataset, 
            prompt,
            metrics,
            f"{exp_name}-{task_type}-model-{model}"
        )
        results.append(result)
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run model evaluations')
    parser.add_argument('--name', '-n', required=True, help='Experiment set name')
    parser.add_argument('--type', '-t', choices=['prompts', 'models'], required=True, 
                      help='Type of experiment: compare prompts or models')
    parser.add_argument('--model', '-m', default='gpt-4o-mini',
                      help='Model to use for prompt comparison (default: gpt-4o-mini)')
    parser.add_argument('--prompt-version', '-p', default='1',
                      help='Prompt version to use for model comparison (default: 1)')
    args = parser.parse_args()
    
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
    qa_samples = []
    for sample in AI_ENGINEERING_SAMPLES:  # Use original samples instead of prompts
        qa_samples.append({
            "question": sample["question"],
            "expected_output": sample["expected_output"],
            "context": sample["context"]
        })

    # Convert summary prompts to dataset format
    summary_samples = []
    for sample in AI_ENGINEERING_VERBOSE:  # Use original verbose texts
        summary_samples.append({
            "text": sample["text"],
            "task": "summarize"
        })

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

    # Run evaluations based on experiment type
    print(f"\nRunning {args.type} comparison experiment...")
    
    if args.type == 'prompts':
        # Compare prompt versions using specified model
        print(f"\nComparing prompt versions using model: {args.model}")
        
        qa_results = evaluate_prompt_versions(
            qa_dataset, qa_prompts, 
            [m for m in metrics if not isinstance(m, GEval)],
            args.model, args.name, "qa"
        )
        
        summary_results = evaluate_prompt_versions(
            summary_dataset, summary_prompts,
            [m for m in metrics if isinstance(m, GEval)],
            args.model, args.name, "summary"
        )
    else:
        # Compare models using specified prompt version
        print(f"\nComparing models using prompt version: {args.prompt_version}")
        
        # Get the specified prompt version for each task
        qa_prompt = next(p for p in qa_prompts 
                        if p.name.endswith(f"-v{args.prompt_version}"))
        summary_prompt = next(p for p in summary_prompts 
                            if p.name.endswith(f"-v{args.prompt_version}"))
        
        qa_results = evaluate_models(
            qa_dataset, qa_prompt,
            [m for m in metrics if not isinstance(m, GEval)],
            models, args.name, "qa"
        )
        
        summary_results = evaluate_models(
            summary_dataset, summary_prompt,
            [m for m in metrics if isinstance(m, GEval)],
            models, args.name, "summary"
        )

    # Print results
    print("\nResults:")
    print("========")
    
    if args.type == 'prompts':
        print(f"\nPrompt comparison using model: {args.model}")
    else:
        print(f"\nModel comparison using prompt version: {args.prompt_version}")
    
    print("\nQA Task Results:")
    for result in qa_results:
        if "error" in result:
            print(f"\nModel: {result['model']}")
            print(f"Error: {result['error']}")
        else:
            print(f"\nModel: {result['model']}")
            print(f"Prompt: {result['prompt_name']}")
            print(f"Duration: {result['duration']:.2f} seconds")
    
    print("\nSummarization Task Results:")
    for result in summary_results:
        if "error" in result:
            print(f"\nModel: {result['model']}")
            print(f"Error: {result['error']}")
        else:
            print(f"\nModel: {result['model']}")
            print(f"Prompt: {result['prompt_name']}")
            print(f"Duration: {result['duration']:.2f} seconds")

if __name__ == "__main__":
    main() 