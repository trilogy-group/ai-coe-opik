"""
Fluency metric for evaluating the linguistic quality of model outputs.
"""

from typing import Optional
from .base import CustomGEval

# Define constants specific to this metric
FLUENCY_INTRO = """You are an expert linguist evaluating the fluency and naturalness of text."""
FLUENCY_CRITERIA = """
Evaluate the following text on a scale of 0-5 for fluency.
Consider these dimensions:
1. Grammar: Is the text grammatically correct?
2. Coherence: Do sentences flow naturally from one to another?
3. Word Choice: Is vocabulary used appropriately and naturally?
4. Readability: Is the text easy to read and understand?
Provide a score and brief explanation.
"""


class FluencyMetric(CustomGEval):
    """
    Evaluates the fluency of text outputs.
    
    Measures grammar correctness, coherence, word choice, and readability.
    """
    
    def __init__(self, name: str = "fluency", model: Optional[str] = None):
        """
        Initialize the fluency metric.
        
        Args:
            name: Metric name
            model: Optional model to use for evaluation
        """
        super().__init__(
            name=name,
            task_introduction=FLUENCY_INTRO,
            evaluation_criteria=FLUENCY_CRITERIA,
            model=model,
        )


def create_fluency_metric() -> FluencyMetric:
    """
    Factory function to create a fluency metric.
    
    Returns:
        A configured FluencyMetric instance
    """
    return FluencyMetric()
