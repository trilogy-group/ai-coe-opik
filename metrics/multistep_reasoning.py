"""
Multistep reasoning metric for evaluating step-by-step reasoning in complex problems.
"""

from typing import Optional
from .base import CustomGEval

# Define constants specific to this metric
MULTISTEP_REASONING_INTRO = (
    """You are an expert in logical reasoning and critical analysis."""
)
MULTISTEP_REASONING_CRITERIA = """
Evaluate the following answer on a scale of 0-5 for its clarity and completeness in multi-step reasoning.
Consider these dimensions:
1. Step-by-Step Clarity: Are the reasoning steps clearly delineated?
2. Logical Coherence: Do the steps logically flow and build upon each other?
3. Thoroughness: Does the answer cover all necessary steps in the reasoning process?
4. Integration of Evidence: Are claims supported by evidence and context?
Provide a score and brief explanation.
"""


class MultistepReasoningMetric(CustomGEval):
    """
    Evaluates the ability to perform multi-step reasoning.

    Measures step-by-step clarity, logical coherence, thoroughness, and integration of evidence.
    """

    def __init__(self, name: str = "multistep_reasoning", model: Optional[str] = None):
        """
        Initialize the multistep reasoning metric.

        Args:
            name: Metric name
            model: Optional model to use for evaluation
        """
        super().__init__(
            name=name,
            task_introduction=MULTISTEP_REASONING_INTRO,
            evaluation_criteria=MULTISTEP_REASONING_CRITERIA,
            model=model,
        )


def create_multistep_reasoning_metric() -> MultistepReasoningMetric:
    """
    Factory function to create a multi-step reasoning metric.

    Returns:
        A configured MultistepReasoningMetric instance
    """
    return MultistepReasoningMetric()
