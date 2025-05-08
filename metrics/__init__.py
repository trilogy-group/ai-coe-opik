"""
Metrics package for evaluating model outputs across different tasks.

This package contains evaluation metrics for different types of tasks:
- General metrics (applicable to most tasks)
- Task-specific metrics (for QA, summarization, code generation, etc.)
"""

from typing import Dict, List

# Import base metric types
from opik.evaluation.metrics import BaseMetric

# Import all specific metrics
from .fluency import FluencyMetric
from .toxicity import ToxicityMetric
from .reasoning import ReasoningMetric
from .factual_accuracy import FactualAccuracyMetric
from .technical_nuance import TechnicalNuanceMetric
from .empathy import EmpathyMetric
from .multistep_reasoning import MultistepReasoningMetric
from .summary_quality import SummaryQualityMetric
from .response_time import ResponseTimeMetric
from .conciseness import ConcisenessMetric
from .format_compliance import FormatComplianceMetric
from .g_eval import GEvalMetric
from .hallucination import CustomHallucination
from .answer_relevance import CustomAnswerRelevance

# Define the list of all available metrics
__all__ = [
    "FluencyMetric",
    "ToxicityMetric",
    "ReasoningMetric",
    "FactualAccuracyMetric",
    "TechnicalNuanceMetric",
    "EmpathyMetric",
    "MultistepReasoningMetric",
    "SummaryQualityMetric",
    "ResponseTimeMetric",
    "ConcisenessMetric",
    "FormatComplianceMetric",
    "GEvalMetric",
    "CustomHallucination",
    "CustomAnswerRelevance",
]
