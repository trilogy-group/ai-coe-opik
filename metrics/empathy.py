"""
Empathy metric for evaluating the emotional intelligence in conversational responses.
"""

from typing import Optional
from .base import CustomGEval

# Define constants specific to this metric
EMPATHY_INTRO = """You are an expert in communication and emotional intelligence evaluating responses for empathy."""
EMPATHY_CRITERIA = """
Evaluate the following response on a scale of 0-5 for empathy.
Consider these dimensions:
1. Emotional Recognition: Does the response acknowledge and validate the emotions expressed?
2. Perspective-Taking: Does the response demonstrate understanding of the other person's perspective?
3. Engagement: Does the response invite further conversation in a positive manner?
4. Contextual Appropriateness: Is the empathetic tone suitable given the conversation context?
Provide a score and brief explanation.
"""


class EmpathyMetric(CustomGEval):
    """
    Evaluates the empathy in conversational responses.

    Measures emotional recognition, perspective-taking, engagement, and contextual appropriateness.
    """

    def __init__(self, name: str = "empathy", model: Optional[str] = None):
        """
        Initialize the empathy metric.

        Args:
            name: Metric name
            model: Optional model to use for evaluation
        """
        super().__init__(
            name=name,
            task_introduction=EMPATHY_INTRO,
            evaluation_criteria=EMPATHY_CRITERIA,
            model=model,
        )


def create_empathy_metric() -> EmpathyMetric:
    """
    Factory function to create an empathy metric.

    Returns:
        A configured EmpathyMetric instance
    """
    return EmpathyMetric()
