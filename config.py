from typing import Dict, Any, Optional, List, Union, Set
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetricConfig:
    name: str
    enabled: bool
    description: Optional[str] = None


@dataclass
class ModelConfig:
    name: str
    description: str
    provider: Optional[str] = None


@dataclass
class TaskConfig:
    name: str
    description: str
    metrics: List[str]


@dataclass
class Config:
    models: List[ModelConfig]
    metrics: List[MetricConfig]
    tasks: Optional[List[TaskConfig]] = None

    def get_metrics_for_task(self, task_name: str) -> List[MetricConfig]:
        """Get metrics configured for a specific task."""
        if not self.tasks:
            return self.metrics

        # Find the task configuration
        task = next((t for t in self.tasks if t.name == task_name), None)
        if not task:
            return self.metrics

        # Get the metrics enabled for this task
        task_metrics = []
        for metric_name in task.metrics:
            metric = next(
                (m for m in self.metrics if m.name == metric_name and m.enabled), None
            )
            if metric:
                task_metrics.append(metric)

        return task_metrics

    def get_metric_names(self) -> Set[str]:
        """Get the names of all enabled metrics."""
        return {m.name for m in self.metrics if m.enabled}

    def is_metric_enabled(self, metric_name: str) -> bool:
        """Check if a specific metric is enabled."""
        return any(m.name == metric_name and m.enabled for m in self.metrics)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

            # Handle tasks if present
            tasks = None
            if "tasks" in data:
                tasks = [TaskConfig(**t) for t in data["tasks"]]

            return cls(
                models=[ModelConfig(**m) for m in data["models"]],
                metrics=[MetricConfig(**m) for m in data["metrics"]],
                tasks=tasks,
            )
