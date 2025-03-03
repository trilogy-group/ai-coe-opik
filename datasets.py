"""
Sample datasets for different evaluation task types.
"""

# ===== Question-Answering Dataset =====
QA_SAMPLES = [
    {
        "question": "What is reasoning distillation and how does it improve LLM performance?",
        "expected_output": "Reasoning distillation is a technique that extracts and transfers reasoning patterns from large language models to smaller ones, improving their performance while reducing size. This is done by training smaller models to mimic the step-by-step reasoning process of larger models.",
        "context": "Reasoning distillation is an emerging technique in AI that focuses on transferring complex reasoning capabilities from large language models to more efficient smaller models. The process involves analyzing and extracting reasoning patterns from teacher models and training student models to replicate these patterns.",
    },
    {
        "question": "How does inference-time scaling work with LLMs?",
        "expected_output": "Inference-time scaling allows dynamic adjustment of model parameters and compute resources based on input complexity. This enables efficient resource utilization by scaling up for complex queries and down for simpler ones.",
        "context": "Inference-time scaling is an optimization technique that dynamically adjusts model complexity and computational resources during inference. It enables more efficient resource usage by matching model capacity to input complexity, rather than using full resources for every query.",
    },
    {
        "question": "What are the key components of GraphRAG and its benefits?",
        "expected_output": "GraphRAG combines graph-based knowledge representation with retrieval-augmented generation. Key components include a knowledge graph, retrieval mechanism, and graph-aware generation. Benefits include improved context understanding and more accurate responses.",
        "context": "GraphRAG is an advanced architecture that integrates graph-based knowledge representation with retrieval-augmented generation. It uses knowledge graphs to structure information and enhance the retrieval and generation process, leading to more contextually aware and accurate responses.",
    },
    {
        "question": "What are effective strategies for evaluating LLM performance?",
        "expected_output": "Effective LLM evaluation strategies include benchmarking against human performance, using automated metrics for consistency and accuracy, conducting behavioral testing, and implementing comparative evaluations across different models and tasks.",
        "context": "LLM evaluation requires a multi-faceted approach combining quantitative metrics, qualitative assessment, and systematic testing. This includes automated evaluation metrics, human evaluation protocols, and comprehensive testing frameworks for assessing model capabilities and limitations.",
    },
    {
        "question": "How do multimodal models process different types of input data?",
        "expected_output": "Multimodal models use specialized encoders for each input type (text, images, audio, etc.), followed by cross-modal fusion mechanisms to create unified representations. This enables understanding relationships between different modalities for comprehensive analysis.",
        "context": "Multimodal models are designed to process and understand multiple types of input data simultaneously. They employ specialized encoding architectures for different modalities, cross-modal attention mechanisms, and fusion techniques to integrate information from various sources effectively.",
    },
]

# ===== Summarization Dataset =====
SUMMARIZATION_SAMPLES = [
    {
        "text": """The implementation of reasoning distillation in large language models represents a significant advancement in AI model optimization. This technique involves a complex process where the cognitive patterns and decision-making pathways of large, computationally intensive models are systematically analyzed, decomposed, and transferred to more compact model architectures. The process begins with a careful analysis of how larger models break down complex problems into smaller, manageable steps, including the identification of key reasoning patterns, logical connections, and problem-solving strategies. These patterns are then extracted and codified into a format that can be used to train smaller, more efficient models. The distillation process involves multiple stages of training, where the student model learns not just the final outputs but also the intermediate reasoning steps that lead to those outputs. This approach has shown remarkable results in preserving up to 90% of the original model's reasoning capabilities while significantly reducing the computational footprint, making it possible to deploy sophisticated AI reasoning systems in resource-constrained environments. The technique has particularly profound implications for edge computing and mobile AI applications, where computational resources are limited but the demand for sophisticated reasoning capabilities remains high.""",
        "task": "summarize",
        "reference_summary": "Reasoning distillation transfers complex problem-solving abilities from large language models to smaller ones by analyzing and extracting reasoning patterns, then training compact models to replicate these patterns. This preserves most capabilities while reducing computational needs, enabling advanced AI reasoning in resource-limited environments like mobile devices.",
    },
    {
        "text": """GraphRAG architecture represents a sophisticated fusion of graph-based knowledge representation and retrieval-augmented generation, fundamentally transforming how AI systems access and utilize information. The architecture integrates multiple specialized components that work in concert: a dynamic knowledge graph that maintains complex relationships between entities and concepts, an advanced retrieval system that leverages graph traversal algorithms to identify relevant information paths, and a context-aware generation mechanism that incorporates both local and global graph structures. The system employs bidirectional attention mechanisms to navigate the knowledge graph during the retrieval phase, considering both forward and backward relationships between nodes to construct comprehensive context representations. The generation component utilizes a novel graph-conditioned decoder that can attend to both the retrieved graph structures and the input query, enabling it to generate responses that reflect a deeper understanding of the interconnected nature of the information. This architectural approach significantly improves the system's ability to maintain consistency across long-range dependencies and complex reasoning chains, while also enabling more nuanced and contextually appropriate responses to complex queries.""",
        "task": "summarize",
        "reference_summary": "GraphRAG combines graph-based knowledge representation with retrieval-augmented generation through a knowledge graph, graph traversal algorithms, and a context-aware generation mechanism. Its bidirectional attention mechanisms and graph-conditioned decoder enable deeper understanding of interconnected information, improving consistency across complex reasoning and enabling more contextually appropriate responses.",
    },
]

# ===== Code Generation Dataset =====
CODE_SAMPLES = [
    {
        "instruction": "Write a Python function that finds the longest common substring between two strings, ensuring the solution is optimized for large inputs.",
        "context": "The function should take two string parameters and return the longest common substring. If there are multiple such substrings, return any one. If there is no common substring, return an empty string. The solution should handle strings up to 10^5 characters efficiently.",
        "expected_format": "Python function",
        "test_cases": [
            {"input": ["abcdefg", "abcxyz"], "expected": "abc"},
            {"input": ["programming", "gaming"], "expected": "ming"},
            {
                "input": ["abcdefghij", "defghiklmn"],
                "expected": "defghi"
            }
        ]
    },
    {
        "instruction": "Create a multi-step React registration form with validation, including an address sub-form with dynamic fields.",
        "context": "The multi-step form should collect username, email, password, and addresses. Each step should have validation. Step 1: username (3-20 chars), email (valid format), password (8+ chars with numbers and special chars). Step 2: allow adding multiple addresses. Each address requires street, city, and country fields. Use client-side validation for required fields and display inline errors. Utilize local state or a form management library.",
        "expected_format": "React JSX component",
        "libraries": ["react", "formik", "yup"]
    },
    {
        "instruction": "Write a SQL query to find the top 5 customers (name, email) who spent the most money in the last calendar month, factoring in refunds from a 'refunds' table.",
        "context": "You have three tables: 'orders' (id, customer_id, order_date, total_amount), 'customers' (id, name, email), and 'refunds' (id, order_id, refund_amount, refund_date). Find the top 5 customers by net spending (total orders - total refunds) in the last calendar month. Show name, email, and net amount spent.",
        "expected_format": "SQL query",
        "schema": "orders(id INT, customer_id INT, order_date DATE, total_amount DECIMAL);\ncustomers(id INT, name VARCHAR, email VARCHAR);\nrefunds(id INT, order_id INT, refund_amount DECIMAL, refund_date DATE)"
    },
    {
        "instruction": "Given a 2D grid representing a maze, write a Python function to find the shortest path from a start cell to an end cell. Some cells have a 'weight' indicating the cost of passing through them. Return the path cost and the path itself.",
        "context": "The function takes the following parameters:\n - grid: a 2D list of integers, where each integer represents the cost of entering that cell (>=1)\n - start: a tuple (row, col)\n - end: a tuple (row, col)\nAssume that the cost of the starting cell is not counted. Use 4-directional movement. Return the total cost and the path as a list of (row, col) coordinates. If no path exists, return an indication (e.g., None) or an empty list with cost = -1.",
        "expected_format": "Python function",
        "test_cases": [
            {
                "input": [
                    [
                        [1, 1, 1],
                        [1, 5, 1],
                        [1, 1, 1]
                    ],
                    (0, 0),
                    (2, 2)
                ],
                "expected": {
                    "cost": 4, 
                    "path": [(0,0), (1,0), (2,0), (2,1), (2,2)]
                }
            },
            {
                "input": [
                    [
                        [2, 5, 2],
                        [9, 1, 3],
                        [1, 1, 4]
                    ],
                    (0, 0),
                    (2, 2)
                ],
                "expected": {
                    "cost": 14,
                    "path": [(0,0), (0,1), (0,2), (1,2), (2,2)]
                }
            },
            {
                "input": [
                    [
                        [1, 10],
                        [10, 1]
                    ],
                    (0, 0),
                    (1, 1)
                ],
                "expected": {
                    "cost": 11,
                    "path": [(0,0), (0,1), (1,1)]
                }
            }
        ]
    }
]

# ===== Chat/Conversation Dataset =====
CHAT_SAMPLES = [
    {
        "conversation": [
            {
                "role": "user",
                "content": "I need to visualize some data for my presentation tomorrow. What options do I have?",
            },
            {
                "role": "system",
                "content": "I'd be happy to help with data visualization options for your presentation. Could you tell me what kind of data you're working with and what message you're trying to convey?",
            },
            {
                "role": "user",
                "content": "I have sales data by region over the last 5 years. I want to show both the overall trend and how each region is performing.",
            },
        ],
        "expected_output": "For your sales data visualization, I'd recommend a few approaches:\n\n1. Line chart: Show the overall sales trend over 5 years with a separate line for each region. This highlights both temporal patterns and regional comparisons.\n\n2. Stacked area chart: Visualize how each region contributes to total sales while showing the overall trend.\n\n3. Small multiples: Create a grid of identical charts, one per region, allowing direct comparison while seeing individual trends.\n\n4. Heatmap: Display years vs. regions with color intensity representing sales values, providing a compact overview.\n\n5. Animated/interactive visualization: Allow your audience to see changes over time dynamically.\n\nTools like Tableau, Power BI, or Python libraries (Matplotlib, Seaborn, Plotly) can create these visualizations effectively.",
    },
    {
        "conversation": [
            {
                "role": "user",
                "content": "What's the difference between supervised and unsupervised learning?",
            },
            {
                "role": "system",
                "content": "In supervised learning, the algorithm learns from labeled training data, with input features and corresponding target outputs. The goal is to learn a mapping function that can predict outputs for new inputs.\n\nIn unsupervised learning, the algorithm works with unlabeled data, trying to find patterns, structures, or relationships without explicit guidance. There are no output labels to predict - instead, the system identifies inherent groupings or patterns in the data.\n\nWhich aspect of these learning paradigms are you most interested in?",
            },
            {
                "role": "user",
                "content": "Are there any approaches that combine both techniques?",
            },
        ],
        "expected_output": "Yes, there are several approaches that combine supervised and unsupervised learning techniques:\n\n1. Semi-supervised learning: Uses a small amount of labeled data with a large amount of unlabeled data. The model learns from the structure provided by unlabeled data while being guided by the labeled examples.\n\n2. Self-supervised learning: Creates supervisory signals from the data itself without manual labeling, by predicting parts of the input from other parts.\n\n3. Transfer learning: Pre-trains a model on an unsupervised task and then fine-tunes it on a supervised task.\n\n4. Reinforcement learning with unsupervised pre-training: Uses unsupervised methods to learn good representations before applying reinforcement learning.\n\n5. Hybrid models: Architectures like autoencoders can be used unsupervised for dimensionality reduction, then the encoded representations can be used for supervised tasks.\n\nThese combined approaches often deliver better performance than purely supervised or unsupervised methods, especially when labeled data is limited.",
    },
]

# Dictionary of all datasets for easy access
ALL_DATASETS = {
    "qa": QA_SAMPLES,
    "summarization": SUMMARIZATION_SAMPLES,
    "code": CODE_SAMPLES,
    "chat": CHAT_SAMPLES,
}
