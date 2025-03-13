"""
G-Eval metric wrapper implementation.
"""

from typing import Optional
from .base import CustomGEval


class GEvalMetric(CustomGEval):
    """
    Wrapper for the GEval metric that includes the task type.
    Allows for different evaluation criteria based on task type.
    """

    def __init__(
        self,
        task_type: str,
        task_introduction: str,
        evaluation_criteria: str,
        name: str = "g_eval_metric",
        model: Optional[str] = None,
    ):
        """
        Initialize the GEval metric.

        Args:
            task_type: Type of task ("qa", "summarization", "code", "chat")
            task_introduction: Introduction to the task for the LLM judge
            evaluation_criteria: Criteria for evaluation
            name: Name of the metric
            model: Optional model to use for evaluation
        """
        super().__init__(
            name=name,
            task_introduction=task_introduction,
            evaluation_criteria=evaluation_criteria,
            model=model,
        )
        self.task_type = task_type


def create_g_eval_metric(
    task_type: str,
    task_introduction: str,
    evaluation_criteria: str,
    name: str = "g_eval_metric",
    model: Optional[str] = None,
) -> GEvalMetric:
    """
    Factory function to create a G-Eval metric.

    Args:
        task_type: Type of task
        task_introduction: Introduction to the task
        evaluation_criteria: Criteria for evaluation
        name: Name of the metric
        model: Optional model to use for evaluation

    Returns:
        A configured GEvalMetric instance
    """
    return GEvalMetric(
        task_type=task_type,
        task_introduction=task_introduction,
        evaluation_criteria=evaluation_criteria,
        name=name,
        model=model,
    )
