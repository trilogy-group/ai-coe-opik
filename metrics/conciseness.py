"""
Conciseness metric for measuring brevity of model outputs.
"""

import math
from opik.evaluation.metrics import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult


class ConcisenessMetric(BaseMetric):
    """
    Measures the conciseness of an output based on token length.
    Returns a score from 0 (very verbose) to 1 (very concise).
    For code, a moderate length is considered optimal.
    """

    def __init__(self, name: str = "conciseness"):
        """
        Initialize the conciseness metric.

        Args:
            name: Metric name
        """
        super().__init__(name=name)

    def _approx_token_count(self, text: str) -> int:
        """
        Get approximate token count for English text.
        This is a rough approximation, using 4 chars per token on average.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate token count
        """
        return len(text) // 4

    def score(self, output: str, **kwargs) -> ScoreResult:
        """
        Score the conciseness of the output.

        Uses a sigmoid function to normalize token counts, with different
        optimal ranges for different content types.

        Args:
            output: The output to evaluate
            **kwargs: Additional arguments

        Returns:
            ScoreResult with conciseness score
        """
        if not output:
            return ScoreResult(
                name=self.name,
                value=1.0,  # Empty output is maximally concise
                reason="Output is empty",
            )

        # Get approximate token count
        token_count = self._approx_token_count(output)

        # Adjust scoring based on content type
        content_type = kwargs.get("content_type", "text")

        if content_type == "code":
            # For code, moderate length is ideal (not too short, not too verbose)
            # Use a bell curve centered around an optimal token count
            optimal_token_count = 200  # Around 50 lines of code

            # Score decreases as we move away from the optimal token count
            # in either direction, but we're more lenient with shorter code
            if token_count <= optimal_token_count:
                # For shorter code, score from 0.5 to 1.0
                normalized_score = 0.5 + 0.5 * (token_count / optimal_token_count)
            else:
                # For longer code, score decreases from 1.0 to 0.0
                normalized_score = max(
                    0.0,
                    1.0
                    - 0.5 * ((token_count - optimal_token_count) / optimal_token_count),
                )

            reason = f"Code has approximately {token_count} tokens"

        else:  # text, qa, etc.
            # For text, shorter is generally better
            # Use a sigmoid function to map token count to a score
            # Parameters tuned so that:
            # - ~50 tokens (short answer) → 0.9 score
            # - ~200 tokens (1 paragraph) → 0.7 score
            # - ~800 tokens (1 page) → 0.3 score
            # - ~2000+ tokens (long doc) → <0.1 score
            sigmoid_midpoint = 400  # Token count where score is 0.5
            sigmoid_steepness = 300  # Controls how quickly score drops

            normalized_score = 1.0 / (
                1.0 + math.exp((token_count - sigmoid_midpoint) / sigmoid_steepness)
            )

            reason = f"Text has approximately {token_count} tokens"

        return ScoreResult(
            name=self.name,
            value=normalized_score,
            reason=reason,
        )


def create_conciseness_metric() -> ConcisenessMetric:
    """
    Factory function to create a conciseness metric.

    Returns:
        A configured ConcisenessMetric instance
    """
    return ConcisenessMetric()
