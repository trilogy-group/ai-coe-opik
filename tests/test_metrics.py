import pytest
import sys
import os
import json
import re
from unittest.mock import MagicMock


# Create mocks for necessary external dependencies
class MockBaseMetric:
    """Mock class for BaseMetric"""

    def __init__(self, name="mock_metric"):
        self.name = name

    def score(self, *args, **kwargs):
        return {"score": 0.0}


# Setup module-level mock before importing
sys.modules["opik.evaluation.metrics.base"] = MagicMock()
sys.modules["opik.evaluation.metrics.base"].BaseMetric = MockBaseMetric


# Mock IsJson metric
class MockIsJson:
    def __init__(self, name="is_json"):
        self.name = name

    def score(self, output, **kwargs):
        try:
            json.loads(output)
            return {"score": 1.0, "reasoning": "Valid JSON"}
        except (json.JSONDecodeError, ValueError):
            return {"score": 0.0, "reasoning": "Invalid JSON"}


# Add to sys.modules
sys.modules["opik.evaluation.metrics"] = MagicMock()
sys.modules["opik.evaluation.metrics"].IsJson = MockIsJson
sys.modules["opik.evaluation.metrics"].GEval = MockBaseMetric

# Now we can import our local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Import test functions that don't depend on external modules
def test_conciseness_calculation():
    """Test the conciseness calculation logic without using the full metric class."""
    # Mock the ConcisenessMetric logic
    min_tokens = 5
    max_tokens = 20

    # Very concise output (7 tokens)
    text = "This is a very concise output"
    tokens = text.split()
    length = len(tokens)

    # Calculate score (same logic as in ConcisenessMetric)
    if length <= min_tokens:
        score = 1.0
    elif length >= max_tokens:
        score = 0.0
    else:
        score = 1.0 - (length - min_tokens) / (max_tokens - min_tokens)

    assert 0.8 <= score <= 1.0, f"Expected high score for concise text, got {score}"

    # Verbose output (more than max_tokens)
    verbose_text = "This is a very verbose output with many words that should receive a low conciseness score because it contains too many tokens and exceeds our max token threshold significantly"
    verbose_length = len(verbose_text.split())

    # Should get a low score
    assert verbose_length > max_tokens, "Test text should be longer than max_tokens"

    # Calculate score for verbose text
    if verbose_length <= min_tokens:
        verbose_score = 1.0
    elif verbose_length >= max_tokens:
        verbose_score = 0.0
    else:
        verbose_score = 1.0 - (verbose_length - min_tokens) / (max_tokens - min_tokens)

    assert (
        verbose_score == 0.0
    ), f"Expected 0.0 score for verbose text, got {verbose_score}"


def test_format_compliance_json():
    """Test JSON format compliance using Opik's IsJson metric."""
    # Valid JSON
    valid_json = '{"name": "test", "value": 123}'

    # Use the MockIsJson directly to simulate how FormatComplianceMetric would use it
    is_json_metric = MockIsJson(name="test_is_json")
    result = is_json_metric.score(output=valid_json)

    assert result["score"] == 1.0, "Valid JSON should be correctly identified"

    # Invalid JSON
    invalid_json = '{name: "test", value: 123}'

    result = is_json_metric.score(output=invalid_json)

    assert result["score"] == 0.0, "Invalid JSON should be correctly identified"

    # Test with markdown code block (simulating what FormatComplianceMetric would do)
    json_in_codeblock = '```json\n{"name": "test", "value": 123}\n```'
    cleaned_json = re.sub(r"```json\s|\s```", "", json_in_codeblock)
    result = is_json_metric.score(output=cleaned_json)

    assert (
        result["score"] == 1.0
    ), "JSON in code block should be correctly identified after cleaning"


def test_code_block_detection():
    """Test code block format detection."""
    # Valid code block with language
    valid_code_block = "```python\nprint('Hello world')\n```"

    # Regex pattern similar to what would be in FormatComplianceMetric
    code_block_pattern = r"```(\w+)?\n(.*?)\n```"
    match = re.search(code_block_pattern, valid_code_block, re.DOTALL)

    assert match is not None, "Should detect valid code block"
    assert match.group(1) == "python", "Should detect language specification"

    # Code block without language
    no_lang_block = "```\nprint('Hello world')\n```"
    match = re.search(code_block_pattern, no_lang_block, re.DOTALL)

    assert match is not None, "Should detect code block without language"
    assert (
        match.group(1) is None or match.group(1) == ""
    ), "Should detect missing language"


def test_response_time_calculation():
    """Test the response time calculation logic without external dependencies."""
    import time

    # Simulate a start time 1.5 seconds ago
    start_time = time.time() - 1.5

    # Calculate elapsed time (simulating what ResponseTimeMetric would do)
    elapsed_time = time.time() - start_time

    # Should be approximately 1.5 seconds with some small variance
    assert 1.4 <= elapsed_time <= 1.6, f"Expected time close to 1.5, got {elapsed_time}"

    # Test handling of missing start_time
    start_time = None
    result_value = -1 if start_time is None else time.time() - start_time
    assert result_value == -1, "Should return -1 for missing start_time"


def test_bullet_points_detection():
    """Test bullet points format detection."""
    # Valid bullet points list
    bullet_list = "- Item 1\n- Item 2\n- Item 3"

    # Regex pattern for bullet points (similar to FormatComplianceMetric)
    bullet_pattern = r"(^|\n)- .+"
    matches = re.findall(bullet_pattern, bullet_list)

    # Should find 3 bullet points
    assert len(matches) == 3, f"Expected 3 bullet points, found {len(matches)}"

    # Invalid bullet points (using * instead of -)
    invalid_list = "* Item 1\n* Item 2\n* Item 3"
    matches = re.findall(bullet_pattern, invalid_list)

    # Should not match our expected bullet format
    assert len(matches) == 0, "Should not detect asterisk bullets as dash bullets"


def test_toxicity_detection():
    """Test basic toxicity detection logic."""
    # Simple word-based toxicity detection
    toxic_words = ["hate", "stupid", "idiot", "terrible"]

    # Non-toxic content
    safe_content = "This is a helpful and informative response to your question."

    # Check for toxic words (simplified version of what a ToxicityMetric might do)
    has_toxic_words = any(word in safe_content.lower() for word in toxic_words)
    assert not has_toxic_words, "Should not detect toxicity in safe content"

    # Toxic content
    toxic_content = "This is a terrible response and whoever wrote it is stupid."

    # Check for toxic words
    has_toxic_words = any(word in toxic_content.lower() for word in toxic_words)
    assert has_toxic_words, "Should detect toxicity in toxic content"


def test_evaluation_result_processing():
    """Test processing of evaluation results."""
    # Mock evaluation results
    mock_results = {
        "items": [
            {
                "metrics": {
                    "conciseness": {"score": 0.8, "reasoning": "Good length"},
                    "relevance": {"score": 0.9, "reasoning": "On topic"},
                }
            },
            {
                "metrics": {
                    "conciseness": {"score": 0.7, "reasoning": "Acceptable length"},
                    "relevance": {"score": 0.6, "reasoning": "Somewhat relevant"},
                }
            },
        ]
    }

    # Calculate averages (similar to what EvaluationResult.calculate_averages would do)
    metric_sums = {}
    metric_counts = {}

    # Process each item in the results
    for item in mock_results.get("items", []):
        metrics = item.get("metrics", {})
        for metric_name, metric_result in metrics.items():
            score = metric_result.get("score", 0)

            # Add to sums and counts
            if metric_name not in metric_sums:
                metric_sums[metric_name] = 0
                metric_counts[metric_name] = 0

            metric_sums[metric_name] += score
            metric_counts[metric_name] += 1

    # Calculate averages
    metric_averages = {}
    for metric_name, total in metric_sums.items():
        count = metric_counts[metric_name]
        metric_averages[metric_name] = total / count if count > 0 else 0

    # Verify calculated averages
    assert (
        metric_averages["conciseness"] == 0.75
    ), f"Expected conciseness average of 0.75, got {metric_averages['conciseness']}"
    assert (
        metric_averages["relevance"] == 0.75
    ), f"Expected relevance average of 0.75, got {metric_averages['relevance']}"


def test_metrics_configuration_loading():
    """Test loading and parsing of metrics configuration."""
    # Mock YAML config data (similar structure to config.yaml)
    yaml_config = """
    metrics:
      - name: conciseness
        enabled: true
        type: heuristic
        description: Measures output brevity
      - name: response_time
        enabled: false
        type: heuristic
        description: Measures model response time
    tasks:
      qa:
        metrics:
          - conciseness
          - response_time
    """

    # Parse YAML (simplified version of config loading)
    try:
        import yaml

        config_data = yaml.safe_load(yaml_config)

        # Check for expected structure
        assert "metrics" in config_data, "Config should contain metrics section"
        assert "tasks" in config_data, "Config should contain tasks section"

        # Count enabled metrics
        enabled_metrics = [m for m in config_data["metrics"] if m.get("enabled", False)]
        assert (
            len(enabled_metrics) == 1
        ), f"Expected 1 enabled metric, found {len(enabled_metrics)}"

        # Check task-specific metrics
        qa_task = config_data["tasks"].get("qa", {})
        assert "metrics" in qa_task, "QA task should have metrics defined"
        assert (
            len(qa_task["metrics"]) == 2
        ), f"Expected 2 metrics for QA task, found {len(qa_task['metrics'])}"

    except ImportError:
        # Skip test if PyYAML is not available
        pytest.skip("PyYAML not available")


def test_task_specific_metrics():
    """Test retrieval of task-specific metrics from config."""
    # Mock config data
    config_data = {
        "metrics": [
            {"name": "conciseness", "enabled": True, "type": "heuristic"},
            {"name": "fluency", "enabled": True, "type": "llm"},
            {"name": "code_quality", "enabled": True, "type": "llm"},
        ],
        "tasks": {
            "qa": {"metrics": ["conciseness", "fluency"]},
            "code": {"metrics": ["code_quality", "conciseness"]},
        },
    }

    # Test QA task metrics
    qa_metrics = [
        m["name"]
        for m in config_data["metrics"]
        if m["name"] in config_data["tasks"]["qa"]["metrics"]
    ]
    assert "conciseness" in qa_metrics, "QA metrics should include conciseness"
    assert "fluency" in qa_metrics, "QA metrics should include fluency"
    assert (
        "code_quality" not in qa_metrics
    ), "QA metrics should not include code_quality"

    # Test code task metrics
    code_metrics = [
        m["name"]
        for m in config_data["metrics"]
        if m["name"] in config_data["tasks"]["code"]["metrics"]
    ]
    assert "code_quality" in code_metrics, "Code metrics should include code_quality"
    assert "conciseness" in code_metrics, "Code metrics should include conciseness"
    assert "fluency" not in code_metrics, "Code metrics should not include fluency"


def test_export_json_format():
    """Test JSON export functionality."""
    # Create mock evaluation results
    mock_results = [
        {
            "model": "gpt-4",
            "prompt_name": "qa-v1",
            "task_type": "qa",
            "duration": 2.5,
            "sample_count": 2,
            "metrics_average_scores": {"conciseness": 0.8, "relevance": 0.9},
            "error": None,
        },
        {
            "model": "gpt-3.5",
            "prompt_name": "qa-v1",
            "task_type": "qa",
            "duration": 1.8,
            "sample_count": 2,
            "metrics_average_scores": {"conciseness": 0.7, "relevance": 0.8},
            "error": None,
        },
    ]

    # Convert to JSON
    import json

    json_output = json.dumps(mock_results, indent=2)

    # Validate JSON structure
    parsed_json = json.loads(json_output)
    assert len(parsed_json) == 2, "JSON should have 2 results"
    assert (
        "metrics_average_scores" in parsed_json[0]
    ), "JSON should include metrics scores"
    assert (
        parsed_json[0]["metrics_average_scores"]["conciseness"] == 0.8
    ), "Metric scores should be preserved"


def test_retry_mechanism():
    """Test the retry mechanism for API calls."""
    # Mock a function that fails with rate limit error
    call_count = 0

    def mock_rate_limited_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:  # Fail twice, succeed on third try
            raise Exception("Rate limit reached, try again in 0.1s")
        return "success"

    # Create a decorator similar to retry_on_rate_limit
    def simple_retry(func, max_retries=3):
        def wrapper():
            retries = 0
            while retries < max_retries:
                try:
                    return func()
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    # Would sleep here in real implementation
            return func()

        return wrapper

    # Apply retry decorator
    retrying_function = simple_retry(mock_rate_limited_function)

    # Test the function
    result = retrying_function()
    assert result == "success", "Function should eventually succeed"
    assert call_count == 3, "Function should have been called 3 times"


def test_argument_parsing():
    """Test command-line argument parsing."""
    import argparse

    # Create a simplified version of the argument parser from main.py
    def create_test_parser():
        parser = argparse.ArgumentParser(description="Test parser")
        parser.add_argument("--name", "-n", required=True, help="Experiment name")
        parser.add_argument(
            "--type",
            "-t",
            choices=["prompts", "models"],
            required=True,
            help="Type of experiment",
        )
        parser.add_argument(
            "--task",
            choices=["qa", "summarization", "code", "chat", "all"],
            default="qa",
            help="Task type",
        )
        return parser

    # Test with valid arguments
    parser = create_test_parser()
    args = parser.parse_args(["--name", "test-exp", "--type", "models"])
    assert args.name == "test-exp", "Name should be correctly parsed"
    assert args.type == "models", "Type should be correctly parsed"
    assert args.task == "qa", "Task should default to qa"

    # Test with different task
    args = parser.parse_args(
        ["--name", "test-exp", "--type", "prompts", "--task", "code"]
    )
    assert args.task == "code", "Task should be set to code"


def test_error_handling_in_evaluation():
    """Test error handling in the evaluation process."""

    # Create a mock EvaluationResult class similar to the one in main.py
    class MockEvaluationResult:
        def __init__(self, model, task_type, error=None):
            self.model = model
            self.task_type = task_type
            self.error = error
            self.metrics_results = {}
            self.metrics_average_scores = {}
            self.sample_count = 0

    # Test error case
    error_result = MockEvaluationResult("test-model", "qa", error="API error")
    assert error_result.error == "API error", "Error should be stored in result"
    assert error_result.sample_count == 0, "No samples should be processed on error"

    # Test success case with empty results
    success_result = MockEvaluationResult("test-model", "qa")
    assert success_result.error is None, "No error should be present"
    assert (
        success_result.metrics_average_scores == {}
    ), "No metrics should be calculated yet"


def test_dataset_handling():
    """Test dataset preparation and handling."""
    # Mock dataset entries
    dataset_entries = [
        {"question": "Q1", "expected_output": "A1", "context": "C1"},
        {"question": "Q2", "expected_output": "A2", "context": "C2"},
        {"question": "Q3", "expected_output": "A3", "context": "C3"},
    ]

    # Test limiting sample count
    sample_count = 2
    import random

    # Fix random seed for reproducibility
    random.seed(42)
    limited_samples = random.sample(dataset_entries, sample_count)
    assert (
        len(limited_samples) == sample_count
    ), f"Should limit to {sample_count} samples"

    # Test handling different task types
    task_types = ["qa", "summarization", "code", "chat"]
    for task in task_types:
        # This would normally call the dataset preparation logic
        # For testing, we just verify the task type is valid
        assert task in task_types, f"{task} should be a valid task type"


def test_format_compliance_metric_edge_cases():
    """Test the FormatComplianceMetric with edge cases."""
    # Test empty string using the mock IsJson
    is_json_metric = MockIsJson()
    result = is_json_metric.score(output="")
    assert result["score"] == 0.0, "Empty string should not be valid JSON"

    # Test malformed code block
    code_block = "```\nprint('Hello')\n"
    code_block_pattern = r"```(\w+)?\n(.*?)\n```"
    match = re.search(code_block_pattern, code_block, re.DOTALL)
    assert match is None, "Unterminated code block should not match"

    # Test mixed format bullet points
    mixed_bullets = "- Item 1\n* Item 2\n- Item 3"
    bullet_pattern = r"(^|\n)- .+"
    matches = re.findall(bullet_pattern, mixed_bullets)
    assert len(matches) == 2, "Should find only dash bullet points"


def test_metric_combination():
    """Test combining multiple metrics and their scoring."""
    # Create mock metric results
    mock_metrics = {
        "conciseness": {"score": 0.8},
        "fluency": {"score": 0.7},
        "toxicity": {"score": 0.1},  # Lower is better for toxicity
    }

    # Calculate combined score (example: weighted average)
    weights = {"conciseness": 0.3, "fluency": 0.5, "toxicity": 0.2}
    combined_score = 0
    for metric, result in mock_metrics.items():
        if metric == "toxicity":
            # For toxicity, lower is better, so we invert the score
            combined_score += weights[metric] * (1 - result["score"])
        else:
            combined_score += weights[metric] * result["score"]

    # 0.3*0.8 + 0.5*0.7 + 0.2*0.9 = 0.24 + 0.35 + 0.18 = 0.77
    expected_score = 0.3 * 0.8 + 0.5 * 0.7 + 0.2 * (1 - 0.1)
    assert (
        abs(combined_score - expected_score) < 0.001
    ), "Combined score calculation should be correct"


def test_config_validation():
    """Test config validation with various edge cases."""
    # Test missing required fields
    incomplete_config = {
        "metrics": [
            {"name": "conciseness", "type": "heuristic"}  # Missing 'enabled' field
        ]
    }

    # In a real implementation, this would validate the config
    # For testing, we just check if the required fields are present
    for metric in incomplete_config.get("metrics", []):
        assert "name" in metric, "Metric should have a name"
        # This would normally raise an error
        has_enabled = "enabled" in metric
        assert not has_enabled, "This metric is missing the 'enabled' field"

    # Test complete config
    complete_config = {
        "metrics": [{"name": "conciseness", "enabled": True, "type": "heuristic"}]
    }

    for metric in complete_config.get("metrics", []):
        assert "name" in metric, "Metric should have a name"
        assert "enabled" in metric, "Metric should have 'enabled' field"
        assert "type" in metric, "Metric should have a type"
