"""
Summary quality metric for evaluating the quality of generated summaries.
"""

import re
from typing import Optional
from .base import CustomGEval
from opik.evaluation.metrics.score_result import ScoreResult
from opik.evaluation.metrics.llm_judges.g_eval.template import (
    G_EVAL_QUERY_TEMPLATE,
)

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
    Each criterion is scored 0-5, for a total possible score of 20.
    The score is then normalized to 0-1 range.
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

    def score(self, output: str, **kwargs):
        """
        Calculate the evaluation score for summary quality.

        This method overrides the parent class to handle the case where
        the model returns a total score out of 20 (summing the 4 criteria)
        instead of a score out of 5.

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

        # Add a parsing instruction that emphasizes the need for a 0-5 score per criterion
        instruction = "\nPlease provide your evaluation with a score (0-5) for each of the four criteria, and a total score normalized to a 0-1 scale. Include your reasoning."
        full_prompt = llm_query + instruction

        # Generate response
        response = self._model.generate_string(input=full_prompt)

        try:
            # Check for various score formats

            # Look for explicit total score out of 20
            total_score_match = re.search(r"\b(\d+)\s*(?:out\s*of|\/)\s*20\b", response)
            if total_score_match:
                total_score = int(total_score_match.group(1))
                # Normalize to 0-1 range
                normalized_score = total_score / 20.0
                reason = f"Total score of {total_score}/20 normalized to {normalized_score:.2f}"

            # Look for individual criterion scores and sum them
            else:
                # Extract all criterion scores (looking for patterns like "Technical Accuracy (5/5)")
                criterion_scores = re.findall(
                    r"\b(\d+)\s*(?:out\s*of|\/)\s*5\b", response
                )

                if criterion_scores and len(criterion_scores) > 0:
                    # Sum up all the criterion scores
                    total_score = sum(int(score) for score in criterion_scores)
                    # Cap at 20 in case there are more than 4 matches
                    total_score = min(total_score, 20)
                    # Normalize to 0-1 range
                    normalized_score = total_score / 20.0
                    reason = f"Sum of criterion scores ({total_score}/20) normalized to {normalized_score:.2f}"
                else:
                    # Fallback to the parent class behavior for 0-5 scores
                    score_match = re.search(r"\b([0-5])\s*\/?\s*5\b", response)
                    if score_match:
                        score = int(score_match.group(1))
                        # Standard normalization for 0-5 score
                        normalized_score = score / 5.0
                        reason = f"Single score of {score}/5 normalized to {normalized_score:.2f}"
                    else:
                        # Last resort - look for any number between 0 and 1 as already normalized
                        normalized_match = re.search(r"\b(0\.\d+|1\.0|1|0)\b", response)
                        if normalized_match:
                            normalized_score = float(normalized_match.group(1))
                            reason = f"Found already normalized score of {normalized_score:.2f}"
                        else:
                            # If no score found, default to middle score
                            normalized_score = 0.5
                            reason = "Could not extract score from response, using default 0.5"

            # Ensure score is in the 0-1 range
            normalized_score = max(0.0, min(1.0, normalized_score))

            # Extract a condensed version of the full response as the detailed reason
            detailed_reason = response.strip()
            if len(detailed_reason) > 500:
                detailed_reason = detailed_reason[:497] + "..."

            return ScoreResult(
                name=self.name,
                value=normalized_score,
                reason=reason,
                metadata={"detailed_justification": detailed_reason},
            )

        except Exception as e:
            # Return a default score on error
            return ScoreResult(
                name=self.name,
                value=0.5,  # Neutral score
                reason=f"Error parsing response: {str(e)}",
                scoring_failed=True,
            )
