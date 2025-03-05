"""
Summary quality metric for evaluating the quality of generated summaries.
"""

from typing import Optional
from .base import CustomGEval

# Define constants specific to this metric
SUMMARY_QUALITY_INTRO = """You are an expert evaluator assessing the quality of AI-generated summaries. You will evaluate how well a summary captures the key technical concepts from a detailed AI engineering text while maintaining accuracy and conciseness."""

SUMMARY_QUALITY_CRITERIA = """Please evaluate the summary based on the following criteria:

1. Technical Accuracy (0-5):
- Does the summary maintain technical accuracy of the original concepts?
- Are there any factual errors or misrepresentations?

2. Completeness (0-5):
- Are all key technical concepts from the original text included?
- Is any critical information missing?

3. Conciseness (0-5):
- Is the summary appropriately brief while maintaining substance?
- Does it avoid unnecessary details while preserving core concepts?

4. Clarity (0-5):
- Is the summary clear and well-structured?
- Are complex concepts presented in an understandable way?
"""


class SummaryQualityMetric(CustomGEval):
    """
    Evaluates the quality of summaries.
    
    Measures technical accuracy, completeness, conciseness, and clarity.
    """
    
    def __init__(self, name: str = "summary_quality", model: Optional[str] = None):
        """
        Initialize the summary quality metric.
        
        Args:
            name: Metric name
            model: Optional model to use for evaluation
        """
        super().__init__(
            name=name,
            task_introduction=SUMMARY_QUALITY_INTRO,
            evaluation_criteria=SUMMARY_QUALITY_CRITERIA,
            model=model,
        )


def create_summary_quality_metric() -> SummaryQualityMetric:
    """
    Factory function to create a summary quality metric.
    
    Returns:
        A configured SummaryQualityMetric instance
    """
    return SummaryQualityMetric()
