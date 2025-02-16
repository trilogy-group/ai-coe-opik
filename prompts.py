import opik

# Define prompt templates
AI_ENGINEERING_PROMPT = """Context: {{context}}

Question: {{question}}

Answer: Let me help you with that."""

# Define sample data for Q&A evaluation
AI_ENGINEERING_SAMPLES = [
    {
        "question": "What is reasoning distillation and how does it improve LLM performance?",
        "expected_output": "Reasoning distillation is a technique that extracts and transfers reasoning patterns from large language models to smaller ones, improving their performance while reducing size. This is done by training smaller models to mimic the step-by-step reasoning process of larger models.",
        "context": "Reasoning distillation is an emerging technique in AI that focuses on transferring complex reasoning capabilities from large language models to more efficient smaller models. The process involves analyzing and extracting reasoning patterns from teacher models and training student models to replicate these patterns."
    },
    {
        "question": "How does inference-time scaling work with LLMs?",
        "expected_output": "Inference-time scaling allows dynamic adjustment of model parameters and compute resources based on input complexity. This enables efficient resource utilization by scaling up for complex queries and down for simpler ones.",
        "context": "Inference-time scaling is an optimization technique that dynamically adjusts model complexity and computational resources during inference. It enables more efficient resource usage by matching model capacity to input complexity, rather than using full resources for every query."
    },
    {
        "question": "What are the key components of GraphRAG and its benefits?",
        "expected_output": "GraphRAG combines graph-based knowledge representation with retrieval-augmented generation. Key components include a knowledge graph, retrieval mechanism, and graph-aware generation. Benefits include improved context understanding and more accurate responses.",
        "context": "GraphRAG is an advanced architecture that integrates graph-based knowledge representation with retrieval-augmented generation. It uses knowledge graphs to structure information and enhance the retrieval and generation process, leading to more contextually aware and accurate responses."
    },
    {
        "question": "What are effective strategies for evaluating LLM performance?",
        "expected_output": "Effective LLM evaluation strategies include benchmarking against human performance, using automated metrics for consistency and accuracy, conducting behavioral testing, and implementing comparative evaluations across different models and tasks.",
        "context": "LLM evaluation requires a multi-faceted approach combining quantitative metrics, qualitative assessment, and systematic testing. This includes automated evaluation metrics, human evaluation protocols, and comprehensive testing frameworks for assessing model capabilities and limitations."
    },
    {
        "question": "How do multimodal models process different types of input data?",
        "expected_output": "Multimodal models use specialized encoders for each input type (text, images, audio, etc.), followed by cross-modal fusion mechanisms to create unified representations. This enables understanding relationships between different modalities for comprehensive analysis.",
        "context": "Multimodal models are designed to process and understand multiple types of input data simultaneously. They employ specialized encoding architectures for different modalities, cross-modal attention mechanisms, and fusion techniques to integrate information from various sources effectively."
    }
]

# Verbose paragraphs for summarization tasks
AI_ENGINEERING_VERBOSE = [
    {
        "text": """The implementation of reasoning distillation in large language models represents a significant advancement in AI model optimization. This technique involves a complex process where the cognitive patterns and decision-making pathways of large, computationally intensive models are systematically analyzed, decomposed, and transferred to more compact model architectures. The process begins with a careful analysis of how larger models break down complex problems into smaller, manageable steps, including the identification of key reasoning patterns, logical connections, and problem-solving strategies. These patterns are then extracted and codified into a format that can be used to train smaller, more efficient models. The distillation process involves multiple stages of training, where the student model learns not just the final outputs but also the intermediate reasoning steps that lead to those outputs. This approach has shown remarkable results in preserving up to 90% of the original model's reasoning capabilities while significantly reducing the computational footprint, making it possible to deploy sophisticated AI reasoning systems in resource-constrained environments. The technique has particularly profound implications for edge computing and mobile AI applications, where computational resources are limited but the demand for sophisticated reasoning capabilities remains high."""
    },
    {
        "text": """GraphRAG architecture represents a sophisticated fusion of graph-based knowledge representation and retrieval-augmented generation, fundamentally transforming how AI systems access and utilize information. The architecture integrates multiple specialized components that work in concert: a dynamic knowledge graph that maintains complex relationships between entities and concepts, an advanced retrieval system that leverages graph traversal algorithms to identify relevant information paths, and a context-aware generation mechanism that incorporates both local and global graph structures. The system employs bidirectional attention mechanisms to navigate the knowledge graph during the retrieval phase, considering both forward and backward relationships between nodes to construct comprehensive context representations. The generation component utilizes a novel graph-conditioned decoder that can attend to both the retrieved graph structures and the input query, enabling it to generate responses that reflect a deeper understanding of the interconnected nature of the information. This architectural approach significantly improves the system's ability to maintain consistency across long-range dependencies and complex reasoning chains, while also enabling more nuanced and contextually appropriate responses to complex queries."""
    }
]

# G-Eval configuration
GEVAL_TASK_INTRO = """You are an expert evaluator assessing the quality of AI-generated summaries. You will evaluate how well a summary captures the key technical concepts from a detailed AI engineering text while maintaining accuracy and conciseness."""

GEVAL_CRITERIA = """Please evaluate the summary based on the following criteria:

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

Total score will be the average of all criteria (0-5 scale)."""

# Different prompt variations for QA
QA_PROMPTS = [
    """Context: {{context}}

Question: {{question}}

Please provide a clear and concise answer.""",

    """You are an AI engineering expert. Using the following context:
{{context}}

Answer this question: {{question}}""",

    """Based on this technical context:
{{context}}

Provide a technical answer to: {{question}}
Focus on accuracy and clarity."""
]

# Different prompt variations for summarization
SUMMARY_PROMPTS = [
    """Please provide a concise summary of the following text about AI engineering:

{{text}}

Summarize the key technical concepts while maintaining accuracy.""",

    """As an AI engineering expert, create a brief technical summary of:

{{text}}

Focus on the core concepts and their relationships.""",

    """Analyze and summarize this AI engineering text:

{{text}}

Provide a concise summary that captures the main technical points and their significance."""
]

def get_prompts():
    """Get or create versioned prompts in Opik library."""
    
    # Create QA prompt variations
    qa_prompts = []
    for idx, prompt_template in enumerate(QA_PROMPTS):
        # Create one prompt per template version
        prompt = opik.Prompt(
            name=f"ai-engineering-qa-v{idx+1}",
            prompt=prompt_template
        )
        qa_prompts.append(prompt)

    # Create summary prompt variations
    summary_prompts = []
    for idx, prompt_template in enumerate(SUMMARY_PROMPTS):
        # Create one prompt per template version
        prompt = opik.Prompt(
            name=f"ai-engineering-summary-v{idx+1}",
            prompt=prompt_template
        )
        summary_prompts.append(prompt)
    
    return {
        "ai_engineering": qa_prompts,
        "summarization": summary_prompts
    } 