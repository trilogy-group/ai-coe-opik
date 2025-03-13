"""
Answer relevance metric implementation.
"""

from typing import Optional
from .base import CustomGEval


class CustomAnswerRelevance(CustomGEval):
    """
    Metric that evaluates if an output is relevant to the input question or prompt.
    """

    def __init__(self, name: str = "answer_relevance", model: Optional[str] = None):
        """
        Initialize the answer relevance metric.

        Args:
            name: Name of the metric
            model: Optional model to use for evaluation
        """
        task_intro = (
            "You are an expert in evaluating the relevance of answers to questions."
        )
        criteria = """Evaluate whether the following output is relevant to the input question or prompt.
        Score on a scale of 0-5 where:
        0 = Completely irrelevant, does not address the question at all
        5 = Highly relevant, directly and comprehensively addresses the question
        
        Provide a score and brief explanation for your reasoning."""

        super().__init__(
            name=name,
            task_introduction=task_intro,
            evaluation_criteria=criteria,
            model=model,
        )


def create_answer_relevance_metric(
    model: Optional[str] = None,
) -> CustomAnswerRelevance:
    """
    Factory function to create an answer relevance metric.

    Args:
        model: Optional model to use for evaluation

    Returns:
        A configured CustomAnswerRelevance instance
    """
    return CustomAnswerRelevance(model=model)
