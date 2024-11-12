# Qwen 2.5
    [URL]: https://arxiv.org/abs/2409.12186v2
    
    [ABS]: In this report, we introduce the Qwen2.5-Coder series, a significant upgrade from its predecessor, CodeQwen1.5. This series includes six models: Qwen2.5-Coder-(0.5B/1.5B/3B/7B/14B/32B). As a code-specific model, Qwen2.5-Coder is built upon the Qwen2.5 architecture and continues pretrained on a vast corpus of over 5.5 trillion tokens. Through meticulous data cleaning, scalable synthetic data generation, and balanced data mixing, Qwen2.5-Coder demonstrates impressive code generation capabilities while retaining general and math skills. These models have been evaluated on a wide range of code-related tasks, achieving state-of-the-art (SOTA) performance across more than 10 benchmarks, including code generation, completion, reasoning, and repair, consistently outperforming larger models of the same model size. We believe that the release of the Qwen2.5-Coder series will advance research in code intelligence and, with its permissive licensing, support wider adoption by developers in real-world applications.

## 3.1 section
1. "developed a coarse-to-fine hierarchical filtering approach for raw data. This method offers two key advantages:
    1. It enables precise control over each filter’s responsibility, ensuring comprehensive handling of each dimension.
    2. It naturally assigns quality scores to the dataset, with data retained in the final stage being of higher quality, providing valuable insights for quality-driven data mixing."

better approach.

2. "A likely explanation is that smaller models focus more on surface-level features, avoiding unnecessary semantic complexity."

use multiple small models > one massive model.

3. "However, all code segments were removed from the general Text data to avoid overlap with our code data, ensuring the independence of different data sources."

be aware of data overlaping, may cause bad effect.

4. "Interestingly, we found that the 7:2:1 ratio outperformed the others, even surpassing the performance of groups with a higher proportion of code. A possible explanation is that Math and Text data may positively contribute to code performance, but only when their concentration reaches a specific threshold."

Interesting.

## 3.2 section
1. File level training (train in individual files) -----> Repo-level training (cross files)

## 4.1 sec
1. "To bridge the gap among different programming languages, we propose a multilingual multi-agent collaborative framework to synthesize the multilingual instruction corpora. We introduce language-specific agents, where a set of specialized agents are created and each dedicated to a particular programming language. These agents are initialized with language-specific instruction data derived from the limited existing multilingual instruction corpora. The multilingual data generation process can be split into: (1) Language-Specific Intelligent Agents: We create a set of specialized agents, each dedicated to a particular programming language. These agents are initialized with language-specific instruction data derived from curated code snippets. (2) Collaborative Discussion Protocol: Multiple language-specific agents engage in a structured dialogue to formulate new instructions and solutions. This process can result in either enhancing existing language capabilities or generating instructions for a novel programming language. (3) Adaptive Memory System: Each agent maintains a dynamic memory bank that stores its generation history to avoid generating the similar samples. (4) Cross-Lingual Discussion: We implement a novel knowledge distillation technique that allows agents to share insights and patterns across language boundaries, fostering a more comprehensive understanding of programming concepts. (5) Synergy Evaluation Metric: We develop a new metric to quantify the degree of knowledge sharing and synergy between different programming languages within the model. (6) Adaptive Instruction Generation: The framework includes a mechanism to dynamically generate new instructions based on identified knowledge gaps across languages."

Conclusively, interaction and communication between multiple small models + memory system + overall evaluation system (multilingual sandbox) + self iteration.

2. "List of evaluation system for code generation"

For your reference.

    The core of the model detail lies in the code.

