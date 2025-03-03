# Opik Evaluation Suite Architecture

This document outlines the architecture of the Opik Evaluation Suite, which provides a comprehensive framework for evaluating Large Language Models across multiple tasks and metrics.

## High-Level Architecture

```mermaid
graph TD
    A[Client: Command Line Interface] --> B[Main Controller]
    B --> C1[Experiment Controller]
    B --> C2[Dataset Manager]
    B --> C3[Metrics Manager]
    B --> C4[Configuration Manager]

    C1 --> D1[Model Evaluation]
    C1 --> D2[Prompt Evaluation]
    
    D1 --> E1[Opik API]
    D2 --> E1
    
    C2 --> E2[Dataset Repository]
    C3 --> E3[Metrics Library]
    C4 --> E4[Config Files]
    
    E1 --> F[Results Processing]
    F --> G[Output Formatting]
```

## Component Architecture

### Evaluation Pipeline

```mermaid
flowchart TD
    A[User Command] --> B[Parse Arguments]
    B --> C{Experiment Type?}
    C -->|Models| D1[Evaluate Different Models]
    C -->|Prompts| D2[Evaluate Different Prompts]
    
    D1 --> E1[Prepare Dataset]
    D1 --> E2[Load Metrics]
    D1 --> E3[Load Models]
    D1 --> E4[Load Prompt]
    
    D2 --> F1[Prepare Dataset]
    D2 --> F2[Load Metrics]
    D2 --> F3[Load Model]
    D2 --> F4[Load Prompts]
    
    E1 & E2 & E3 & E4 --> G1[Evaluate Models]
    F1 & F2 & F3 & F4 --> G2[Evaluate Prompts]
    
    G1 & G2 --> H[Process Results]
    H --> I[Calculate Metrics]
    I --> J[Format Output]
    J --> K[Display Results]
```

### Metrics System

```mermaid
classDiagram
    class BaseMetric {
        +name: str
        +score(output: str, **kwargs)
    }
    
    class HeuristicMetric {
        +score()
    }
    
    class LLMBasedMetric {
        +model: Model
        +score()
    }
    
    class ConcisenessMetric {
        +min_tokens: int
        +max_tokens: int
        +score()
    }
    
    class ResponseTimeMetric {
        +start_time: float
        +score()
    }
    
    class FormatComplianceMetric {
        +format_type: str
        +score()
    }
    
    class CustomGEval {
        +task_introduction: str
        +evaluation_criteria: str
        +model: str
        +score()
    }
    
    BaseMetric <|-- HeuristicMetric
    BaseMetric <|-- LLMBasedMetric
    
    HeuristicMetric <|-- ConcisenessMetric
    HeuristicMetric <|-- ResponseTimeMetric
    HeuristicMetric <|-- FormatComplianceMetric
    
    LLMBasedMetric <|-- CustomGEval
    
    CustomGEval <|-- FluencyMetric
    CustomGEval <|-- ToxicityMetric
    CustomGEval <|-- CodeQualityMetric
    CustomGEval <|-- ReasoningMetric
```

### Configuration System

```mermaid
classDiagram
    class Config {
        +models: List[ModelConfig]
        +metrics: List[MetricConfig]
        +tasks: List[TaskConfig]
        +get_metrics_for_task(task_name)
    }
    
    class ModelConfig {
        +name: str
        +description: str
        +provider: str
    }
    
    class MetricConfig {
        +name: str
        +enabled: bool
        +description: str
    }
    
    class TaskConfig {
        +name: str
        +description: str
        +metrics: List[str]
    }
    
    Config "1" --> "*" ModelConfig: contains
    Config "1" --> "*" MetricConfig: contains
    Config "1" --> "*" TaskConfig: contains
```

### Data Flow Architecture

```mermaid
flowchart LR
    A[Config Files] --> B[Configuration Manager]
    B --> C[Experiment Controller]
    
    D[Dataset Files] --> E[Dataset Manager]
    E --> C
    
    F[Metrics Library] --> G[Metrics Manager]
    G --> C
    
    C --> H[Opik API Client]
    H --> I[External LLM APIs]
    I --> H
    
    H --> J[Result Processor]
    J --> K[Output Formatter]
    K --> L[Terminal Output]
```

### Test Framework Architecture

```mermaid
flowchart TD
    A[Unit Tests] --> B{Test Types}
    B --> C[Metric Tests]
    B --> D[Configuration Tests]
    B --> E[Dataset Tests]
    B --> F[Result Processing Tests]
    B --> G[Error Handling Tests]
    
    C --> H[Test Runners]
    D --> H
    E --> H
    F --> H
    G --> H
    
    H --> I[Test Reports]
```

## Directory Structure

```mermaid
graph TD
    A[opik/] --> B[main.py]
    A --> C[metrics.py]
    A --> D[config.py]
    A --> E[datasets.py]
    A --> F[prompts.py]
    A --> G[requirements.txt]
    A --> H[config.yaml]
    A --> I[README.md]
    A --> J[tests/]
    
    J --> K[test_metrics.py]
    J --> L[test_main.py]
    J --> M[test_results.py]
    J --> N[conftest.py]
```
