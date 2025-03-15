"""
Utility functions for working with metrics.
"""

from typing import Dict, List

from opik.evaluation.metrics import BaseMetric

# Import our local metrics
from .response_time import ResponseTimeMetric
from .conciseness import ConcisenessMetric
from .format_compliance import FormatComplianceMetric
from .hallucination import CustomHallucination
from .answer_relevance import CustomAnswerRelevance

# Import factory functions
from .fluency import create_fluency_metric
from .toxicity import create_toxicity_metric
from .reasoning import create_reasoning_metric
from .factual_accuracy import create_factual_accuracy_metric
from .technical_nuance import create_technical_nuance_metric
from .empathy import create_empathy_metric
from .multistep_reasoning import create_multistep_reasoning_metric
from .summary_quality import create_summary_quality_metric


def get_metrics_for_task(task_type: str, config: Dict) -> List[BaseMetric]:
    """
    Get appropriate metrics for a given task type.

    Args:
        task_type: Type of task ("qa", "summarization", "code", etc.)
        config: Dictionary containing metric configuration

    Returns:
        List of metric instances
    """
    # Common metrics for all tasks
    common_metrics = []

    # Add metrics based on config
    for metric_config in config.metrics:
        if not metric_config.enabled:
            continue

        if metric_config.name == "hallucination":
            common_metrics.append(CustomHallucination())

        elif metric_config.name == "answer_relevance":
            common_metrics.append(CustomAnswerRelevance())

        elif metric_config.name == "response_time":
            common_metrics.append(ResponseTimeMetric())

        elif metric_config.name == "conciseness":
            common_metrics.append(ConcisenessMetric())

        elif metric_config.name == "fluency":
            common_metrics.append(create_fluency_metric())

        elif metric_config.name == "toxicity":
            common_metrics.append(create_toxicity_metric())

        elif metric_config.name == "empathy":
            common_metrics.append(create_empathy_metric())

        elif metric_config.name == "technical_nuance":
            common_metrics.append(create_technical_nuance_metric())

        elif metric_config.name == "multistep_reasoning":
            common_metrics.append(create_multistep_reasoning_metric())

    # Task-specific metrics
    task_specific_metrics = []

    if task_type == "qa":
        # Add QA-specific metrics
        for metric_config in config.metrics:
            if not metric_config.enabled:
                continue

            if (
                metric_config.name == "reasoning_quality"
                or metric_config.name == "reasoning"
            ):
                task_specific_metrics.append(create_reasoning_metric())

            elif metric_config.name == "factual_accuracy":
                task_specific_metrics.append(create_factual_accuracy_metric())

            elif metric_config.name == "technical_nuance":
                task_specific_metrics.append(create_technical_nuance_metric())

            elif metric_config.name == "multistep_reasoning":
                task_specific_metrics.append(create_multistep_reasoning_metric())

    elif task_type == "summarization":
        # Add summarization-specific metrics
        for metric_config in config.metrics:
            if not metric_config.enabled:
                continue

            if metric_config.name == "summary_quality":
                task_specific_metrics.append(create_summary_quality_metric())

            elif metric_config.name == "factual_accuracy":
                task_specific_metrics.append(create_factual_accuracy_metric())

            elif metric_config.name == "technical_nuance":
                task_specific_metrics.append(create_technical_nuance_metric())

    return common_metrics + task_specific_metrics
