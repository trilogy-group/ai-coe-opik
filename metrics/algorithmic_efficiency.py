"""
Algorithmic efficiency metric for evaluating the efficiency of code solutions.
"""

from typing import Optional
from .base import CustomGEval

# Define constants specific to this metric
ALGORITHMIC_EFFICIENCY_INTRO = """You are an expert in algorithmic analysis evaluating code efficiency."""
ALGORITHMIC_EFFICIENCY_CRITERIA = """
Evaluate the following code on a scale of 0-5 for its algorithmic efficiency.
Consider these dimensions:
1. Time Complexity: What is the runtime efficiency in terms of Big O notation?
2. Space Complexity: How efficiently does the code use memory?
3. Scalability: Will the solution perform well as input size increases?
4. Data Structure Choice: Are the most appropriate data structures used for the task?
5. Optimization: Are there unnecessary operations or missed optimization opportunities?
Provide a score and brief explanation.
"""


class AlgorithmicEfficiencyMetric(CustomGEval):
    """
    Evaluates the algorithmic efficiency of code outputs.
    
    Measures time complexity, space complexity, scalability, data structure choice, and optimizations.
    """
    
    def __init__(self, name: str = "algorithmic_efficiency", model: Optional[str] = None):
        """
        Initialize the algorithmic efficiency metric.
        
        Args:
            name: Metric name
            model: Optional model to use for evaluation
        """
        super().__init__(
            name=name,
            task_introduction=ALGORITHMIC_EFFICIENCY_INTRO,
            evaluation_criteria=ALGORITHMIC_EFFICIENCY_CRITERIA,
            model=model,
        )


def create_algorithmic_efficiency_metric() -> AlgorithmicEfficiencyMetric:
    """
    Factory function to create an algorithmic efficiency metric.
    
    Returns:
        A configured AlgorithmicEfficiencyMetric instance
    """
    return AlgorithmicEfficiencyMetric()
