"""
Reasoning metric for evaluating logical reasoning in model outputs.
"""

from typing import Optional
from .base import CustomGEval

# Define constants specific to this metric
REASONING_INTRO = """You are an expert in logic and critical thinking evaluating the reasoning quality of responses."""
REASONING_CRITERIA = """
Evaluate the following response on a scale of 0-5 for reasoning quality.
Consider these dimensions:
1. Logical Consistency: Is the reasoning internally consistent and free of contradictions?
2. Relevance: Does the reasoning address the specific question or problem?
3. Comprehensiveness: Does the reasoning consider relevant factors and alternatives?
4. Soundness: Are the inferences and conclusions warranted by the premises and evidence?
Provide a score and brief explanation.
"""


class ReasoningMetric(CustomGEval):
    """
    Evaluates the reasoning quality of text outputs.
    
    Measures logical consistency, relevance, comprehensiveness, and soundness.
    """
    
    def __init__(self, name: str = "reasoning_quality", model: Optional[str] = None):
        """
        Initialize the reasoning metric.
        
        Args:
            name: Metric name
            model: Optional model to use for evaluation
        """
        super().__init__(
            name=name,
            task_introduction=REASONING_INTRO,
            evaluation_criteria=REASONING_CRITERIA,
            model=model,
        )


def create_reasoning_metric() -> ReasoningMetric:
    """
    Factory function to create a reasoning quality metric.
    
    Returns:
        A configured ReasoningMetric instance
    """
    return ReasoningMetric()
