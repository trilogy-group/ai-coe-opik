import opik

# Define prompt templates
AI_ENGINEERING_PROMPT = """Context: {{context}}

Question: {{question}}

Answer: Let me help you with that."""

# Define sample data
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

# Create versioned prompts
def get_prompts():
    """Get or create versioned prompts in Opik library."""
    
    # Create prompts from samples
    prompts = []
    for sample in AI_ENGINEERING_SAMPLES:
        prompt = opik.Prompt(
            name=f"ai-engineering-{len(prompts)+1}",
            prompt=f"""Context: {sample['context']}

Question: {sample['question']}

Expected Answer: {sample['expected_output']}"""
        )
        prompts.append(prompt)
    
    return {
        "ai_engineering": prompts
    } 