"""
Base metrics module providing shared functionality for all metrics.
"""

from typing import Dict, Optional, Any
import re
from functools import cached_property

from opik.evaluation.metrics import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult
from opik.evaluation.models import models_factory
from opik.evaluation.metrics.llm_judges.g_eval.template import (
    G_EVAL_COT_TEMPLATE,
    G_EVAL_QUERY_TEMPLATE,
)


class CustomGEval(BaseMetric):
    """
    Base class for G-Eval based metrics that provides common functionality.
    """

    def __init__(
        self,
        name: str,
        task_introduction: str,
        evaluation_criteria: str,
        model: Optional[str] = None,
    ):
        """
        Initialize a G-Eval based metric.

        Args:
            name: The name of the metric
            task_introduction: Introduction text for the evaluator
            evaluation_criteria: Criteria for the evaluation
            model: Optional model to use for evaluation
        """
        super().__init__(name=name)
        self.task_introduction = task_introduction
        self.evaluation_criteria = evaluation_criteria
        self.model = model
        # For some metrics like hallucination, a higher score is worse
        # This flag can be set to True to invert the score (1.0 - score)
        self.invert_score = False

        # Initialize the model
        if isinstance(model, object) and hasattr(model, "generate_string"):
            self._model = model
        else:
            # Don't require logprobs support
            self._model = models_factory.get(model_name=model)

    @cached_property
    def llm_chain_of_thought(self) -> str:
        prompt = G_EVAL_COT_TEMPLATE.format(
            task_introduction=self.task_introduction,
            evaluation_criteria=self.evaluation_criteria,
        )
        return self._model.generate_string(input=prompt)

    def score(self, output: str, **kwargs):
        """
        Calculate the evaluation score without using logprobs.

        Args:
            output: The output to evaluate

        Returns:
            ScoreResult object with score and reason
        """
        # Build prompt with evaluation template
        llm_query = G_EVAL_QUERY_TEMPLATE.format(
            task_introduction=self.task_introduction,
            evaluation_criteria=self.evaluation_criteria,
            chain_of_thought=self.llm_chain_of_thought,
            input=output,
        )

        # Add a simple parsing instruction to make output parsing easier
        instruction = "\nProvide your evaluation in a JSON object with 'score' (integer 0-5) and 'reason' keys. Ensure the score is scaled on a 0-5 range."
        full_prompt = llm_query + instruction

        # Generate response without logprobs
        response = self._model.generate_string(input=full_prompt)

        # Parse response - extract score and reason from JSON
        try:
            # Try to extract a JSON object from the response
            json_match = re.search(
                r'\{.*?"score"\s*:\s*(\d+).*?"reason"\s*:\s*"(.*?)".*?\}',
                response,
                re.DOTALL,
            )

            if json_match:
                score = int(json_match.group(1))
                reason = json_match.group(2)

                # Handle scores outside the expected 0-5 range
                if score > 5:
                    # If model responded with a score on a 0-10 scale, rescale it to 0-5
                    if score <= 10:
                        score = round(score / 2)
                    else:
                        # For any other unexpected scale, cap at 5
                        score = min(score, 5)
            else:
                # Fallback regex for scores
                score_match = re.search(r"\b([0-5])\s*\/?\s*5\b", response)
                if score_match:
                    score = int(score_match.group(1))
                    # Extract some text after the score as the reason
                    reason_text = response[score_match.end() :].strip()
                    reason = (
                        reason_text[:200] + "..."
                        if len(reason_text) > 200
                        else reason_text
                    )
                else:
                    # Last resort - look for numbers 0-5
                    number_match = re.search(r"\b([0-5])\b", response)
                    if number_match:
                        score = int(number_match.group(1))
                        reason = "Score extracted from response"
                    else:
                        # If no score found, default to middle score
                        score = 3
                        reason = "Could not extract score from response"

            # Normalize score to 0-1 range
            normalized_score = score / 5.0

            # Invert if needed (e.g., for hallucination where high score is bad)
            if getattr(self, "invert_score", False):
                normalized_score = 1.0 - normalized_score

            return ScoreResult(name=self.name, value=normalized_score, reason=reason)
        except Exception as e:
            # Return a default score on error
            return ScoreResult(
                name=self.name,
                value=0.5,  # Neutral score
                reason=f"Error parsing response: {str(e)}",
            )
