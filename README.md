# 🎓 Wand University: Debate-Driven Research Enhancement 🤖

Welcome to this repository focused on enhancing AI research capabilities through debate and dynamic paper analysis. This notebook demonstrates how to create specialized research agents that engage in structured debate while adaptively exploring arXiv papers.

## 📚 Core Capabilities

This repository provides tools for:

- 🔍 Dynamic arXiv paper search and analysis with shifting queries
- 💭 Structured debate generation with constrained outputs
- 🎲 High-temperature sampling with min_p for creative exploration
- 📊 Multi-perspective argument synthesis and evaluation
- 🔄 Iterative knowledge refinement through debate cycles

## 🤖 The Enhancement Process

1. **Paper Discovery**: Dynamic arXiv queries evolve based on debate outcomes and knowledge gaps.

2. **Structured Analysis**: Papers are parsed into key arguments using constrained generation templates.

3. **Debate Generation**: High-temperature sampling with min_p creates diverse, opposing viewpoints.

4. **Argument Synthesis**: Multiple perspectives are evaluated and synthesized into structured knowledge.

5. **Query Evolution**: Search terms adapt based on debate outcomes and emerging research directions.

6. **Knowledge Integration**: Refined understanding is preserved in structured debate records.

## 🎯 Performance Tracking

The debate and research process is monitored via Weights & Biases, tracking:
- Argument diversity metrics
- Knowledge evolution patterns
- Search query effectiveness
- Debate quality assessment

## ⚠️ System Requirements

Important: This system requires significant computational resources:
- Minimum 4x NVIDIA A100 GPUs for parallel debate simulation
- Additional GPU(s) for paper processing

The debate server must be launched separately using: vLLM
