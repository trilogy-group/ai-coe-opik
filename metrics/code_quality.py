"""
Code quality metric for evaluating the quality of generated code.
"""

from typing import Optional
from .base import CustomGEval

# Define constants specific to this metric
CODE_QUALITY_INTRO = (
    """You are an expert software engineer evaluating the quality of code."""
)
CODE_QUALITY_CRITERIA = """
Evaluate the following code on a scale of 0-5 for its overall quality.
Consider these dimensions:
1. Correctness: Does the code correctly implement the intended functionality?
2. Readability: Is the code easy to read and understand?
3. Efficiency: Does the code use efficient algorithms and data structures?
4. Maintainability: Is the code well-structured, modular, and easy to maintain?
5. Error Handling: Does the code handle edge cases and potential errors appropriately?
Provide a score and brief explanation.
"""


class CodeQualityMetric(CustomGEval):
    """
    Evaluates the quality of code outputs.

    Measures correctness, readability, efficiency, maintainability, and error handling.
    """

    def __init__(self, name: str = "code_quality", model: Optional[str] = None):
        """
        Initialize the code quality metric.

        Args:
            name: Metric name
            model: Optional model to use for evaluation
        """
        super().__init__(
            name=name,
            task_introduction=CODE_QUALITY_INTRO,
            evaluation_criteria=CODE_QUALITY_CRITERIA,
            model=model,
        )


def create_code_quality_metric() -> CodeQualityMetric:
    """
    Factory function to create a code quality metric.

    Returns:
        A configured CodeQualityMetric instance
    """
    return CodeQualityMetric()
