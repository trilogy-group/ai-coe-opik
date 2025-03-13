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
from .fluency import FluencyMetric, create_fluency_metric
from .toxicity import ToxicityMetric, create_toxicity_metric
from .reasoning import ReasoningMetric, create_reasoning_metric
from .factual_accuracy import FactualAccuracyMetric, create_factual_accuracy_metric
from .code_quality import CodeQualityMetric, create_code_quality_metric
from .algorithmic_efficiency import (
    AlgorithmicEfficiencyMetric,
    create_algorithmic_efficiency_metric,
)
from .technical_nuance import TechnicalNuanceMetric, create_technical_nuance_metric
from .empathy import EmpathyMetric, create_empathy_metric
from .multistep_reasoning import (
    MultistepReasoningMetric,
    create_multistep_reasoning_metric,
)
from .summary_quality import SummaryQualityMetric, create_summary_quality_metric
from .response_time import ResponseTimeMetric, create_response_time_metric
from .conciseness import ConcisenessMetric, create_conciseness_metric
from .format_compliance import FormatComplianceMetric, create_format_compliance_metric
from .g_eval import GEvalMetric, create_g_eval_metric
from .hallucination import CustomHallucination, create_hallucination_metric
from .answer_relevance import CustomAnswerRelevance, create_answer_relevance_metric

# Re-export the get_metrics_for_task function
from .metrics_utils import get_metrics_for_task

# Define the list of all available metrics
__all__ = [
    "FluencyMetric",
    "ToxicityMetric",
    "ReasoningMetric",
    "FactualAccuracyMetric",
    "CodeQualityMetric",
    "AlgorithmicEfficiencyMetric",
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
    "create_fluency_metric",
    "create_toxicity_metric",
    "create_reasoning_metric",
    "create_factual_accuracy_metric",
    "create_code_quality_metric",
    "create_algorithmic_efficiency_metric",
    "create_technical_nuance_metric",
    "create_empathy_metric",
    "create_multistep_reasoning_metric",
    "create_summary_quality_metric",
    "create_response_time_metric",
    "create_conciseness_metric",
    "create_format_compliance_metric",
    "create_g_eval_metric",
    "create_hallucination_metric",
    "create_answer_relevance_metric",
    "get_metrics_for_task",
]
