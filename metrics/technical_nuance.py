"""
Technical nuance metric for evaluating the ability to capture technical details.
"""

from typing import Optional
from .base import CustomGEval

# Define constants specific to this metric
TECHNICAL_NUANCE_INTRO = """You are an expert technical reviewer evaluating content for technical accuracy and nuance."""
TECHNICAL_NUANCE_CRITERIA = """
Evaluate the following content on a scale of 0-5 for its technical nuance.
Consider these dimensions:
1. Technical Precision: Are specialized terms used correctly and appropriately?
2. Conceptual Accuracy: Are technical concepts accurately represented?
3. Depth: Does the content demonstrate deep understanding rather than surface-level knowledge?
4. Contextual Awareness: Does the content show awareness of the broader technical context?
Provide a score and brief explanation.
"""


class TechnicalNuanceMetric(CustomGEval):
    """
    Evaluates the technical nuance of content.

    Measures technical precision, conceptual accuracy, depth, and contextual awareness.
    """

    def __init__(self, name: str = "technical_nuance", model: Optional[str] = None):
        """
        Initialize the technical nuance metric.

        Args:
            name: Metric name
            model: Optional model to use for evaluation
        """
        super().__init__(
            name=name,
            task_introduction=TECHNICAL_NUANCE_INTRO,
            evaluation_criteria=TECHNICAL_NUANCE_CRITERIA,
            model=model,
        )


def create_technical_nuance_metric() -> TechnicalNuanceMetric:
    """
    Factory function to create a technical nuance metric.

    Returns:
        A configured TechnicalNuanceMetric instance
    """
    return TechnicalNuanceMetric()
