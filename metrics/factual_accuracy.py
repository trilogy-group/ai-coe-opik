"""
Factual accuracy metric for evaluating the correctness of information in model outputs.
"""

from typing import Optional
from .base import CustomGEval

# Define constants specific to this metric
FACTUAL_ACCURACY_INTRO = (
    """You are an expert fact-checker evaluating the factual accuracy of responses."""
)
FACTUAL_ACCURACY_CRITERIA = """
Evaluate the following response on a scale of 0-5 for factual accuracy.
Consider these dimensions:
1. Correctness: Does the response contain correct facts and information?
2. Precision: Is the information specific and precise?
3. Completeness: Does the response include all relevant facts for the context?
4. Verifiability: Can the information be verified against reliable sources?
Provide a score and brief explanation.
"""


class FactualAccuracyMetric(CustomGEval):
    """
    Evaluates the factual accuracy of text outputs.

    Measures correctness, precision, completeness, and verifiability of facts.
    """

    def __init__(self, name: str = "factual_accuracy", model: Optional[str] = None):
        """
        Initialize the factual accuracy metric.

        Args:
            name: Metric name
            model: Optional model to use for evaluation
        """
        super().__init__(
            name=name,
            task_introduction=FACTUAL_ACCURACY_INTRO,
            evaluation_criteria=FACTUAL_ACCURACY_CRITERIA,
            model=model,
        )


def create_factual_accuracy_metric() -> FactualAccuracyMetric:
    """
    Factory function to create a factual accuracy metric.

    Returns:
        A configured FactualAccuracyMetric instance
    """
    return FactualAccuracyMetric()
