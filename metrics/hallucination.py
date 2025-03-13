"""
Hallucination metric implementation.
"""

from typing import Optional
from .base import CustomGEval


class CustomHallucination(CustomGEval):
    """
    Metrics that evaluates outputs for hallucinations or fabricated information.
    """

    def __init__(self, name: str = "hallucination", model: Optional[str] = None):
        """
        Initialize the hallucination metric.

        Args:
            name: Name of the metric
            model: Optional model to use for evaluation
        """
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

        # For hallucination metric, we want to invert the score
        # since a high hallucination score is bad, but metrics should
        # return higher values for better performance
        self.invert_score = True


def create_hallucination_metric(model: Optional[str] = None) -> CustomHallucination:
    """
    Factory function to create a hallucination metric.

    Args:
        model: Optional model to use for evaluation

    Returns:
        A configured CustomHallucination instance
    """
    return CustomHallucination(model=model)
