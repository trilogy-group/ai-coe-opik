"""
Format compliance metric for verifying outputs match expected formats.
"""

import re
from typing import Dict, Any
from opik.evaluation.metrics import BaseMetric, IsJson
from opik.evaluation.metrics.score_result import ScoreResult


class FormatComplianceMetric(BaseMetric):
    """
    Evaluates whether an output conforms to a specified format.

    Supported formats:
    - json: Checks if output is valid JSON
    - code_block: Checks if output contains code blocks or code-like content
    - bullet_points: Checks if output contains bullet point lists
    - numbered_list: Checks if output contains numbered lists
    - table: Checks if output contains a markdown table
    """

    def __init__(
        self, format_type: str = "code_block", name: str = "format_compliance"
    ):
        """
        Initialize the format compliance metric.

        Args:
            format_type: Type of format to check ("json", "code_block", "bullet_points", etc.)
            name: Name of the metric
        """
        super().__init__(name=name)
        self.format_type = format_type.lower()

    def score(self, output: str, **kwargs) -> ScoreResult:
        """
        Score the format compliance of the output.

        Args:
            output: The output to evaluate
            **kwargs: Additional arguments

        Returns:
            ScoreResult with format compliance score (0 or 1)
        """
        if self.format_type == "json":
            # Use Opik's built-in IsJson metric
            # Strip markdown code block syntax if present
            cleaned_output = re.sub(r"```json\s|\s```", "", output)
            is_json_metric = IsJson(name=f"{self.name}_is_json")
            result = is_json_metric.score(output=cleaned_output)
            # Handle ScoreResult objects which are not subscriptable
            if hasattr(result, "value") and result.value == 1.0:
                return ScoreResult(
                    name=self.name, value=1.0, reason="Output is valid JSON"
                )
            else:
                return ScoreResult(
                    name=self.name, value=0.0, reason="Output is not valid JSON"
                )

        elif self.format_type == "code_block":
            # Check if output contains markdown code blocks or appears to be code
            if re.search(r"```[\w]*\n[\s\S]*?\n```", output) or re.search(
                r"(def |class |function |import |const |let |var |public |private )",
                output,
            ):
                return ScoreResult(
                    name=self.name,
                    value=1.0,
                    reason="Output contains code in appropriate format",
                )
            else:
                return ScoreResult(
                    name=self.name,
                    value=0.0,
                    reason="Output does not contain code blocks",
                )

        elif self.format_type == "bullet_points":
            # Check if output contains bullet points
            if re.search(r"^\s*[-*•]\s+.+", output, re.MULTILINE):
                return ScoreResult(
                    name=self.name,
                    value=1.0,
                    reason="Output contains bullet points",
                )
            else:
                return ScoreResult(
                    name=self.name,
                    value=0.0,
                    reason="Output does not contain bullet points",
                )

        elif self.format_type == "numbered_list":
            # Check if output contains numbered list
            if re.search(r"^\s*\d+\.\s+.+", output, re.MULTILINE):
                return ScoreResult(
                    name=self.name,
                    value=1.0,
                    reason="Output contains numbered list",
                )
            else:
                return ScoreResult(
                    name=self.name,
                    value=0.0,
                    reason="Output does not contain numbered list",
                )

        elif self.format_type == "table":
            # Check if output contains markdown table
            if re.search(r"\|.*\|.*\n\|[\s-]*\|[\s-]*\|", output):
                return ScoreResult(
                    name=self.name,
                    value=1.0,
                    reason="Output contains markdown table",
                )
            else:
                return ScoreResult(
                    name=self.name,
                    value=0.0,
                    reason="Output does not contain markdown table",
                )

        else:
            # Unknown format type
            return ScoreResult(
                name=self.name,
                value=0.0,
                reason=f"Unknown format type: {self.format_type}",
            )


def create_format_compliance_metric(
    format_type: str = "code_block",
) -> FormatComplianceMetric:
    """
    Factory function to create a format compliance metric.

    Args:
        format_type: Type of format to check

    Returns:
        A configured FormatComplianceMetric instance
    """
    return FormatComplianceMetric(format_type=format_type)
