"""
Response time metric for measuring LLM response time.
"""

import time
from typing import Dict, Any
from opik.evaluation.metrics import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult


class ResponseTimeMetric(BaseMetric):
    """
    Measures the time taken by a model to generate a response.
    Returns the time in minutes as a raw value (not normalized).
    """

    def __init__(self, name: str = "response_time"):
        """
        Initialize the response time metric.

        Args:
            name: Metric name
        """
        super().__init__(name=name)
        self.start_time = time.time()

    def score(self, output: str, **kwargs) -> ScoreResult:
        """
        Score the response time.

        Returns a ScoreResult with the response time in minutes.
        Can use timing information from various sources:
        1. start_time/duration passed in kwargs
        2. trace_data from the model
        3. test_case evaluation time
        4. Internal start_time (least accurate)

        Args:
            output: Output (not used by this metric)
            **kwargs: Additional arguments including timing information

        Returns:
            ScoreResult with response time in minutes
        """
        # First try to get start time and duration from kwargs
        start_time = kwargs.get("start_time")
        duration = kwargs.get("duration")

        if start_time is not None and duration is not None:
            # Convert seconds to minutes
            response_time = duration / 60.0
            return ScoreResult(
                name=self.name,
                value=response_time,
                reason=f"Response time (from Opik): {response_time:.4f} minutes",
            )

        # If we have the trace data, use that
        trace_data = kwargs.get("trace_data", {})
        if trace_data and "duration_seconds" in trace_data:
            # Convert seconds to minutes
            response_time = trace_data["duration_seconds"] / 60.0
            return ScoreResult(
                name=self.name,
                value=response_time,
                reason=f"Response time (from trace): {response_time:.4f} minutes",
            )

        # Try to get the evaluation timing from test case info
        test_case = kwargs.get("test_case", {})
        if test_case and hasattr(test_case, "evaluation_time"):
            # Convert seconds to minutes
            response_time = test_case.evaluation_time / 60.0
            return ScoreResult(
                name=self.name,
                value=response_time,
                reason=f"Response time (from test case): {response_time:.4f} minutes",
            )

        # Fall back to our own timing if nothing else is available
        # First check kwargs, then fallback to instance attribute
        start_time = kwargs.get("start_time", self.start_time)
        if start_time is None:
            return ScoreResult(
                name=self.name,
                value=0.0,
                reason="Could not determine response time",
            )

        # Calculate elapsed time in minutes
        elapsed_time = time.time() - start_time
        minutes = elapsed_time / 60.0

        return ScoreResult(
            name=self.name,
            value=minutes,
            reason=f"Response time: {minutes:.4f} minutes",
        )
