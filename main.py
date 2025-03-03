import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import wraps, partial
import json
import os
import random
import time
from typing import List, Dict, Optional, Union, Any, Tuple, Callable
import yaml

from dotenv import load_dotenv
from litellm.exceptions import RateLimitError

import opik
from opik.evaluation import evaluate_prompt

# Custom metrics
from metrics import ResponseTimeMetric

# Import Opik's built-in metrics
from opik.evaluation.metrics import GEval, IsJson

# Import the original Hallucination and AnswerRelevance classes
from opik.evaluation.metrics.llm_judges.hallucination.metric import (
    Hallucination as OpikHallucination,
)
from opik.evaluation.metrics.llm_judges.answer_relevance.metric import (
    AnswerRelevance as OpikAnswerRelevance,
)

import polars as pl
import tabulate

# Local imports
from config import Config
from metrics import (
    ConcisenessMetric,
    ResponseTimeMetric,
    FormatComplianceMetric,
    FluencyMetric,
    ToxicityMetric,
    ReasoningMetric,
    GEvalMetric,
    CustomGEval,
)


# Create custom wrappers that don't use logprobs
class CustomHallucination(CustomGEval):
    def __init__(self, name: str = "hallucination", model: str = None):
        # Get the prompt template from the Opik original implementation
        hallucination_metric = OpikHallucination()
        # Use the TEMPLATE from hallucination.metric.py but initialize with our CustomGEval
        task_intro = "You are an expert fact-checker. Evaluate whether the following output contains any hallucinations or fabricated information."
        criteria = """Carefully analyze whether the output contains any information that is not supported by the input/context.
        Score on a scale of 0-5 where:
        0 = No hallucinations at all, fully factual
        5 = Significant hallucinations present
        
        Provide a score and brief explanation."""

        super().__init__(
            name=name,
            task_introduction=task_intro,
            evaluation_criteria=criteria,
            model=model,
        )


class CustomAnswerRelevance(CustomGEval):
    def __init__(self, name: str = "answer_relevance", model: str = None):
        # Get the prompt template from the Opik original implementation
        answer_relevance_metric = OpikAnswerRelevance()
        # Use the TEMPLATE from answer_relevance.metric.py but initialize with our CustomGEval
        task_intro = "You are an expert in evaluating question answering systems."
        criteria = """Evaluate whether the answer is relevant to the question and addresses what was asked.
        Score on a scale of 0-5 where:
        0 = Completely irrelevant or off-topic
        5 = Directly addresses the question with high relevance
        
        Provide a score and brief explanation."""

        super().__init__(
            name=name,
            task_introduction=task_intro,
            evaluation_criteria=criteria,
            model=model,
        )


from prompts import get_prompts, GEVAL_TASK_INTRO, GEVAL_CRITERIA
from datasets import ALL_DATASETS

# Load environment variables
load_dotenv()


@dataclass
class EvaluationResult:
    """Structured container for evaluation results."""

    model: str
    prompt_name: str
    task_type: str
    duration: float
    metrics_results: Dict
    error: Optional[str] = None
    sample_count: int = 0
    metrics_average_scores: Dict[str, float] = field(default_factory=dict)

    def _process_metric_result(self, metric_result) -> float:
        """
        Process a metric result to extract the score, handling various possible formats.

        Args:
            metric_result: The metric result object/dict to process

        Returns:
            The score as a float, or 0 if no score could be extracted
        """
        # Handle ScoreResult objects which have a 'value' attribute
        if hasattr(metric_result, "value"):
            return metric_result.value

        # If it's a dictionary, try to get the score from it
        if isinstance(metric_result, dict):
            # Try different keys that might contain the score
            for key in ["value", "score"]:
                if key in metric_result:
                    return metric_result[key]

        # If it has a score attribute, use that
        if hasattr(metric_result, "score"):
            return metric_result.score

        # If it has a get method (like a dict without a "score" key), try that
        if hasattr(metric_result, "get"):
            # Try different keys with the get method
            for key in ["value", "score"]:
                value = metric_result.get(key)
                if value is not None:
                    return value

        # If we got here, we couldn't extract a score
        print(f"Warning: Could not extract score from metric result: {metric_result}")
        return 0

        # Last resort: try to cast it to a float
        try:
            return float(metric_result)
        except (TypeError, ValueError):
            return 0

    def calculate_averages(self) -> None:
        """Calculate average scores for each metric across all samples."""
        if not self.metrics_results:
            return

        # Calculate average for each metric
        metric_sums = {}
        metric_counts = {}

        # Process each item in the results
        for item in self.metrics_results.get("items", []):
            # Get metrics dict - handle both dict and object access patterns
            if isinstance(item, dict):
                metrics = item.get("metrics", {})
            elif hasattr(item, "metrics"):
                metrics = item.metrics
            else:
                continue  # Skip this item if we can't extract metrics

            # Process each metric
            for metric_name, metric_result in metrics.items():
                score = self._process_metric_result(metric_result)
                if score is not None:
                    metric_sums[metric_name] = metric_sums.get(metric_name, 0) + score
                    metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1

        # Calculate averages
        self.metrics_average_scores = {}
        for metric_name, metric_sum in metric_sums.items():
            count = metric_counts.get(metric_name, 0)
            if count > 0:
                self.metrics_average_scores[metric_name] = metric_sum / count
            else:
                self.metrics_average_scores[metric_name] = 0

        # Count the number of samples more reliably
        sample_count = 0

        # First try to get it from the dataset_size field
        if (
            isinstance(self.metrics_results, dict)
            and "dataset_size" in self.metrics_results
        ):
            sample_count = self.metrics_results["dataset_size"]
        # Try to get dataset_size as an attribute
        elif hasattr(self.metrics_results, "dataset_size"):
            sample_count = self.metrics_results.dataset_size
        # Count items if we have them
        elif (
            isinstance(self.metrics_results, dict)
            and "items" in self.metrics_results
            and self.metrics_results["items"]
        ):
            sample_count = len(self.metrics_results["items"])
        # If we still don't have anything, look for raw_items
        elif (
            isinstance(self.metrics_results, dict)
            and "raw_items" in self.metrics_results
            and self.metrics_results["raw_items"]
        ):
            sample_count = len(self.metrics_results["raw_items"])

        # If we still have 0 samples but have results, set to the count of metrics processed
        if sample_count == 0 and metric_counts:
            # Use the maximum count of any metric
            sample_count = max(metric_counts.values()) if metric_counts else 0

        # If we still have 0 but have metric results, set to a minimum of 1
        if sample_count == 0 and self.metrics_average_scores:
            sample_count = 1

        self.sample_count = sample_count


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM evaluation suite")
    parser.add_argument("--name", "-n", required=True, help="Experiment set name")
    parser.add_argument(
        "--type",
        "-t",
        choices=["prompts", "models"],
        required=True,
        help="Type of experiment: compare prompts or models",
    )
    parser.add_argument(
        "--task",
        choices=["qa", "summarization", "code", "chat", "all"],
        default="qa",
        help="Task type to evaluate (default: qa)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gpt-4o-mini",
        help="Model to use for prompt comparison (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--prompt-version",
        "-p",
        default="1",
        help="Prompt version to use for model comparison (default: 1)",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=int,
        default=0,
        help="Number of samples to use (0 for all)",
    )
    parser.add_argument(
        "--evaluator",
        "-e",
        default="o3-mini",
        help="Model to use for LLM-based evaluations (default: o3-mini)",
    )
    return parser.parse_args()


def load_config() -> Config:
    """Load configuration using dataclass structure."""
    return Config.from_yaml("config.yaml")


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

                        time_match = re.search(r"try again in (\d+\.?\d*)s", str(e))
                        wait_time = float(time_match.group(1)) if time_match else 1
                    except:
                        wait_time = 1

                    print(
                        f"\nRate limit reached. Retrying in {wait_time:.2f} seconds... (Attempt {retries}/{max_retries})"
                    )
                    time.sleep(wait_time)
            return func(*args, **kwargs)

        return wrapper

    return decorator


@retry_on_rate_limit()
def evaluate_model(
    model: str,
    dataset: opik.Dataset,
    prompt: opik.Prompt,
    metrics: List,
    exp_name: str,
    task_type: str,
) -> EvaluationResult:
    """Evaluate a single model with better error handling."""
    try:
        print(
            f"\nEvaluating model: {model} on task: {task_type} with prompt: {prompt.name}"
        )
        start_time = time.time()

        # Make sure we have a ResponseTimeMetric in the metrics list
        has_response_time_metric = False
        for metric in metrics:
            if isinstance(metric, ResponseTimeMetric):
                has_response_time_metric = True
                metric.start_time = start_time  # Set the start time
                break

        # Add a ResponseTimeMetric if not already in the list
        if not has_response_time_metric:
            response_time_metric = ResponseTimeMetric()
            response_time_metric.start_time = start_time
            metrics.append(response_time_metric)

        # Evaluate the model directly
        try:
            # Extract provider and model name if specified in the format provider/model
            provider = None
            model_name = model

            # Check if model has a provider prefix
            if "/" in model:
                parts = model.split("/")
                provider = parts[0]
                # For models like openrouter/meta-llama/llama-3.2-3b-instruct
                # we want to keep the full path for OpenRouter
                if provider == "openrouter":
                    model_name = model
                else:
                    # For other providers like ollama, just use the model name
                    model_name = parts[-1]

            # Count the samples in the dataset
            num_samples = 0
            if hasattr(dataset, "sample_count"):
                num_samples = dataset.sample_count
            elif hasattr(dataset, "items") and isinstance(dataset.items, list):
                num_samples = len(dataset.items)

            # Configure environment for provider-specific settings
            original_openai_api_base = os.environ.get("OPENAI_API_BASE", None)
            original_openai_api_key = os.environ.get("OPENAI_API_KEY", None)
            original_openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", None)

            try:
                # Set provider-specific environment variables
                if provider == "openrouter":
                    # For OpenRouter, set the API base and use the OpenRouter API key
                    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
                    # If we have an OpenRouter API key, use it
                    if os.environ.get("OPENROUTER_API_KEY"):
                        os.environ["OPENAI_API_KEY"] = os.environ.get(
                            "OPENROUTER_API_KEY"
                        )
                    # Keep the full model path for OpenRouter
                elif provider == "ollama":
                    # For Ollama, set the API base to the local Ollama server
                    os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
                    # Use just the model name without the provider prefix
                    model_name = model_name.split("/")[-1]

                # Execute the evaluation with standard parameters
                result = evaluate_prompt(
                    dataset=dataset,
                    messages=[{"role": "user", "content": prompt.prompt}],
                    model=model_name,
                    scoring_metrics=metrics,
                    experiment_name=exp_name,
                    project_name="LLM-Evals",
                    task_threads=14,
                    verbose=0,  # Add verbose output to help diagnose issues
                )
            finally:
                # Restore original environment variables
                if original_openai_api_base:
                    os.environ["OPENAI_API_BASE"] = original_openai_api_base
                elif "OPENAI_API_BASE" in os.environ:
                    del os.environ["OPENAI_API_BASE"]

                if original_openai_api_key:
                    os.environ["OPENAI_API_KEY"] = original_openai_api_key
                elif "OPENAI_API_KEY" in os.environ and provider == "openrouter":
                    # Only restore if we changed it for OpenRouter
                    del os.environ["OPENAI_API_KEY"]

            # If the dataset has sample count information, propagate it
            if not hasattr(result, "dataset_size"):
                if isinstance(result, dict):
                    result["dataset_size"] = num_samples
                else:
                    try:
                        result.dataset_size = num_samples
                    except:
                        pass

            # For metrics like ResponseTimeMetric that need the start_time,
            # we can manually set it after the evaluation
            for metric in metrics:
                if hasattr(metric, "start_time"):
                    metric.start_time = start_time

            # Extract timing data from Opik if available
            end_time = time.time()
            computation_time = end_time - start_time

            # Extract timing from various sources with fallbacks
            trace_data = None

            # Try different approaches to get timing information
            if (
                hasattr(result, "metrics")
                and isinstance(result.metrics, dict)
                and "latency" in result.metrics
            ):
                # Get from metrics dictionary
                trace_data = {"duration_seconds": result.metrics["latency"] / 1000.0}
            elif hasattr(result, "trace"):
                if hasattr(result.trace, "duration_seconds"):
                    # Get from trace duration_seconds
                    trace_data = {"duration_seconds": result.trace.duration_seconds}
                else:
                    # Try to extract span duration if available
                    try:
                        if (
                            hasattr(result.trace, "spans")
                            and len(result.trace.spans) > 0
                        ):
                            root_span = result.trace.spans[0]
                            trace_data = {
                                "duration_seconds": root_span.duration / 1_000_000_000
                            }  # Convert ns to seconds
                        else:
                            trace_data = {"duration_seconds": computation_time}
                    except Exception:
                        trace_data = {"duration_seconds": computation_time}
            else:
                trace_data = {"duration_seconds": computation_time}

            # Get timing directly from the result metadata
            try:
                if hasattr(result, "results"):
                    for test_result in result.results:
                        if hasattr(test_result, "metadata") and test_result.metadata:
                            if (
                                "generation_info" in test_result.metadata
                                and "latency" in test_result.metadata["generation_info"]
                            ):
                                latency = test_result.metadata["generation_info"][
                                    "latency"
                                ]
                                trace_data = {"duration_seconds": latency / 1000.0}
                                break
            except Exception:
                pass

            # Pass the trace data to metrics that need it
            found_response_time_metric = False
            for metric in metrics:
                if isinstance(metric, ResponseTimeMetric):
                    found_response_time_metric = True

                    # Update with more accurate timing from trace data
                    updated_result = metric.score(
                        output="",  # Not used by ResponseTimeMetric
                        start_time=start_time,
                        duration=computation_time,
                        trace_data=trace_data,
                        model=model_name,
                    )

                    # Update the result in the scores list
                    try:
                        if hasattr(result, "update_metric_result"):
                            result.update_metric_result(updated_result)
                        elif isinstance(result, dict) and "scores" in result:
                            updated = False
                            for i, score in enumerate(result["scores"]):
                                if score["name"] == metric.name:
                                    result["scores"][i] = (
                                        updated_result.to_dict()
                                        if hasattr(updated_result, "to_dict")
                                        else updated_result
                                    )
                                    updated = True
                            if not updated:
                                # Add the score if not found
                                result_dict = (
                                    updated_result.to_dict()
                                    if hasattr(updated_result, "to_dict")
                                    else updated_result
                                )
                                result["scores"].append(result_dict)
                    except Exception:
                        pass

            # Create a consistent structure for the result
            if hasattr(result, "to_dict"):
                result = result.to_dict()
            if not isinstance(result, dict):
                result = {"items": []}
            if "items" not in result:
                result["items"] = []
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            # Create an empty result structure
            result = {"items": []}

        duration = time.time() - start_time

        # Check if response_time metric exists in the results
        response_time_exists = False
        if "scores" in result:
            for score in result["scores"]:
                if score.get("name") == "response_time":
                    response_time_exists = True
                    break

        # If response_time metric doesn't exist, add it manually
        if not response_time_exists and "scores" in result:
            # Create a response time result using our own duration (convert to minutes)
            duration_minutes = duration / 60.0
            response_time_result = {
                "name": "response_time",
                "value": duration_minutes,
                "reason": f"Response time (manually added): {duration_minutes:.4f} minutes",
            }
            result["scores"].append(response_time_result)

        # Store the dataset size in the result for sample count tracking
        if not "dataset_size" in result:
            # Try to get the sample count from various sources
            dataset_size = 0

            # Try to get dataset size from dataset.sample_count
            if hasattr(dataset, "sample_count"):
                dataset_size = dataset.sample_count
            # Try to get it from the length of items in the dataset
            elif hasattr(dataset, "items") and isinstance(dataset.items, list):
                dataset_size = len(dataset.items)
            # Try if it's available as a property
            elif hasattr(dataset, "size"):
                dataset_size = dataset.size
            # Directly count items from our dataset samples
            elif task_type in ALL_DATASETS:
                dataset_size = len(ALL_DATASETS[task_type]["samples"])

            # Set the count in the result
            result["dataset_size"] = dataset_size

        evaluation_result = EvaluationResult(
            model=model,
            prompt_name=prompt.name,
            task_type=task_type,
            duration=duration,
            metrics_results=result,
        )

        # Calculate average scores
        evaluation_result.calculate_averages()

        return evaluation_result
    except Exception as e:
        print(f"Error evaluating model {model}: {str(e)}")
        return EvaluationResult(
            model=model,
            prompt_name=prompt.name,
            task_type=task_type,
            duration=0,
            metrics_results={},
            error=str(e),
        )


def evaluate_prompt_versions(dataset, prompts, metrics, model, exp_name, task_type):
    """Evaluate different prompt versions using a single model."""
    results = []
    print(
        f"\nEvaluating {len(prompts)} prompt versions for {task_type} task using model: {model}"
    )

    for prompt in prompts:
        result = evaluate_model(
            model,
            dataset,
            prompt,
            metrics,
            f"{exp_name}-{task_type}-prompt-v{prompt.name.split('-v')[1]}",
            task_type,
        )
        results.append(result)

    return results


def evaluate_models(dataset, prompt, metrics, models, exp_name, task_type):
    """Evaluate different models using a single prompt."""
    results = []
    print(
        f"\nEvaluating {len(models)} models for {task_type} task using prompt version: {prompt.name}"
    )

    for model in models:
        result = evaluate_model(
            model,
            dataset,
            prompt,
            metrics,
            f"{exp_name}-{task_type}-model-{model}",
            task_type,
        )
        results.append(result)

    return results


def prepare_dataset(opik_client, task_type, sample_count=0):
    """Prepare dataset for evaluation based on task type."""
    # Get dataset samples
    samples = ALL_DATASETS.get(task_type, [])
    if not samples:
        raise ValueError(f"No samples available for task type: {task_type}")

    # Limit samples if requested
    if sample_count > 0 and sample_count < len(samples):
        samples = random.sample(samples, sample_count)

    # Create or get dataset
    dataset_name = f"{task_type.capitalize()} Evaluation Dataset"
    dataset = opik_client.get_or_create_dataset(dataset_name)

    # Insert samples into dataset
    dataset.insert(samples)

    # Store the sample count in multiple places to ensure it's picked up
    try:
        # Store as a direct attribute
        dataset.sample_count = len(samples)

        # Store it in a dictionary attribute if possible
        if hasattr(dataset, "__dict__"):
            dataset.__dict__["sample_count"] = len(samples)

        # Try to store it as metadata if the dataset supports it
        if hasattr(dataset, "metadata") and isinstance(dataset.metadata, dict):
            dataset.metadata["sample_count"] = len(samples)
    except Exception as e:
        print(f"Warning: Could not store sample count: {e}")

    print(f"Prepared dataset '{dataset_name}' with {len(samples)} samples")
    return dataset


def get_metrics_for_task(task_type, config, evaluator_model=None):
    """Get metrics for a specific task type based on configuration.

    Args:
        task_type: Type of task (qa, summarization, etc.)
        config: Configuration object
        evaluator_model: Optional model to use for LLM-based evaluations
    """
    # Get task-specific metrics from config
    task_metrics = config.get_metrics_for_task(task_type)

    metrics = []
    for metric_config in task_metrics:
        if not metric_config.enabled:
            continue

        # Instantiate metrics based on their name
        if metric_config.name == "conciseness":
            metrics.append(ConcisenessMetric())
        elif metric_config.name == "response_time":
            metrics.append(ResponseTimeMetric())
        elif metric_config.name == "format_compliance":
            metrics.append(FormatComplianceMetric())
        elif metric_config.name == "hallucination":
            # LLM-based metric
            if evaluator_model:
                metrics.append(CustomHallucination(model=evaluator_model))
            else:
                metrics.append(CustomHallucination())
        elif metric_config.name == "fluency":
            # LLM-based metric
            if evaluator_model:
                metrics.append(FluencyMetric(model=evaluator_model))
            else:
                metrics.append(FluencyMetric())
        elif metric_config.name == "toxicity":
            # LLM-based metric
            if evaluator_model:
                metrics.append(ToxicityMetric(model=evaluator_model))
            else:
                metrics.append(ToxicityMetric())
        elif metric_config.name == "reasoning_quality":
            # LLM-based metric
            if evaluator_model:
                metrics.append(ReasoningMetric(model=evaluator_model))
            else:
                metrics.append(ReasoningMetric())
        elif metric_config.name == "answer_relevance":
            # LLM-based metric
            if evaluator_model:
                metrics.append(CustomAnswerRelevance(model=evaluator_model))
            else:
                metrics.append(CustomAnswerRelevance())
        elif (
            metric_config.name == "factual_accuracy"
            or metric_config.name == "summary_quality"
        ):
            # Use our custom GEvalMetric wrapper
            # Use a descriptive name based on which metric was requested
            if evaluator_model:
                metrics.append(
                    GEvalMetric(
                        task_type=task_type,
                        task_introduction=GEVAL_TASK_INTRO,
                        evaluation_criteria=GEVAL_CRITERIA,
                        model=evaluator_model,
                        name=metric_config.name,
                    )
                )
            else:
                metrics.append(
                    GEvalMetric(
                        task_type=task_type,
                        task_introduction=GEVAL_TASK_INTRO,
                        evaluation_criteria=GEVAL_CRITERIA,
                        name=metric_config.name,
                    )
                )

        elif metric_config.name == "code_quality":
            # LLM-based metric for code quality
            from metrics import CodeQualityMetric

            if evaluator_model:
                metrics.append(CodeQualityMetric(model=evaluator_model))
            else:
                metrics.append(CodeQualityMetric())

    return metrics


def print_results(results):
    """Print evaluation results to the terminal in a structured format using tabulate."""
    print("\n\n=== EVALUATION RESULTS ===")

    # Group results by task type
    task_types = set(r.task_type for r in results)

    for task_type in task_types:
        task_results = [r for r in results if r.task_type == task_type]
        print(f"\n{task_type.upper()} TASK RESULTS ({len(task_results)} evaluations):")

        # Calculate all possible metrics for this task type
        all_metrics = set()
        for result in task_results:
            all_metrics.update(result.metrics_average_scores.keys())

        # Prepare table headers
        headers = ["Model", "Prompt"]
        for metric in sorted(all_metrics):
            headers.append(metric)
        headers.extend(["Duration", "Samples"])

        # Prepare table rows
        table_data = []
        for result in task_results:
            row = [result.model, result.prompt_name]

            # Add metrics
            for metric in sorted(all_metrics):
                score = result.metrics_average_scores.get(metric, "N/A")
                if isinstance(score, (int, float)):
                    row.append(f"{score:.2f}")
                else:
                    row.append(score)

            # Add duration and sample count
            row.append(f"{result.duration:.2f}s")
            row.append(result.sample_count)
            table_data.append(row)

        # Print the table using tabulate
        print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))

    # Print overall statistics
    print("\n=== OVERALL STATISTICS ===")
    total_duration = sum(r.duration for r in results)
    total_samples = sum(r.sample_count for r in results)
    print(f"Total evaluation time: {total_duration:.2f} seconds")
    print(f"Total samples evaluated: {total_samples}")


def main():
    """Main function for running the evaluation suite."""
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = load_config()

    # Initialize Opik client
    opik_client = opik.Opik()
    # opik_client = opik.Opik(workspace="trilogy-ai-coe") # FIXME: issue creating datasets in other workspaces
    # Get all available prompts
    all_prompts = get_prompts()

    # Determine which task types to evaluate
    task_types = (
        ["qa", "summarization", "code", "chat"] if args.task == "all" else [args.task]
    )

    all_results = []

    # Process each task type
    for task_type in task_types:
        print(f"\n\n=== EVALUATING {task_type.upper()} TASKS ===")

        # Skip if no prompts available for this task type
        if task_type not in all_prompts:
            print(f"No prompts available for task type: {task_type}")
            continue

        # Get prompts for this task type
        task_prompts = all_prompts[task_type]
        if not task_prompts:
            print(f"No prompts available for task type: {task_type}")
            continue

        # Prepare dataset
        try:
            dataset = prepare_dataset(opik_client, task_type, args.samples)
        except ValueError as e:
            print(f"Error preparing dataset: {str(e)}")
            continue

        # Get metrics for this task type
        task_metrics = get_metrics_for_task(task_type, config, args.evaluator)
        if not task_metrics:
            print(f"No metrics defined for task type: {task_type}")
            continue

        print(f"Using {len(task_metrics)} metrics for {task_type} task evaluation")

        # Run evaluations based on experiment type
        if args.type == "prompts":
            # For prompt comparison, evaluate all prompt versions using a single model
            model = args.model

            # Evaluate all prompt versions
            results = evaluate_prompt_versions(
                dataset, task_prompts, task_metrics, model, args.name, task_type
            )
            all_results.extend(results)

        elif args.type == "models":
            # For model comparison, evaluate all models using a single prompt version
            prompt_version = int(args.prompt_version) - 1
            if prompt_version < 0 or prompt_version >= len(task_prompts):
                print(
                    f"Invalid prompt version {args.prompt_version} for task type {task_type}"
                )
                prompt_version = 0

            # Get specific prompt version
            prompt = task_prompts[prompt_version]

            # Get model names from config
            models = [m.name for m in config.models]

            # Evaluate all models
            results = evaluate_models(
                dataset, prompt, task_metrics, models, args.name, task_type
            )
            all_results.extend(results)

    # Process and output results
    if not all_results:
        print("\nNo evaluation results to report.")
        return

    # Always print results to terminal
    print_results(all_results)


if __name__ == "__main__":
    main()
