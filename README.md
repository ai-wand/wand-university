# 🎓 Wand University: Debate-Driven Research Enhancement 🤖

Welcome to the Wand University repository, a platform for advancing AI research through dynamic debate and iterative knowledge refinement. This system utilizes a novel approach, combining structured debate with adaptive paper analysis to enhance the capabilities of our AI agents. The core idea revolves around agents engaging in debates to refine their understanding and generate high quality training data, creating a self-improving cycle of knowledge acquisition.

## 📚 Core Capabilities

This repository provides a robust set of tools for:

*   **🔍 Dynamic arXiv Paper Exploration:** Agents initiate research with broad queries that dynamically evolve based on debate outcomes and knowledge gaps. This ensures exploration of diverse but related research, creating a comprehensive knowledge base.
*   **💭 Structured Debate Generation:**  We utilize constrained generation templates to extract key arguments from research papers, fostering a structured debate format.
*   **🎲 High-Temperature Sampling with `min_p`:**  Creative exploration is encouraged using high-temperature sampling with `min_p`, resulting in diverse viewpoints and innovative arguments within debates.
*   **📊 Multi-Perspective Argument Synthesis:**  The system synthesizes multiple perspectives through structured evaluations and then refines and preserves arguments, leading to a cohesive and well-rounded understanding.
*   **🔄 Iterative Knowledge Refinement:** Through debate cycles, the system refines its knowledge, adapting search terms based on emerging research directions, and preserving these insights in structured debate records.
*  **🚀 LoRA Fine-tuning:** We are able to take our acquired data and fine-tune an LLM on it as well as finetune using other datasets like EQ-Bench and GPQA.
* **🧪 Evaluation**: We have integrated the `lm-evaluation-harness` to evaluate our models after fine-tuning to track their performance. 

## 🤖 The Enhancement Process

1.  **Paper Discovery:**
    *   The process begins with a broad arXiv query based on current research focus.
    *   The system fetches relevant papers and selects one for in-depth analysis.
    *   Subsequent queries adapt based on debate outcomes and identified knowledge gaps to ensure diverse yet relevant exploration.

2.  **Structured Analysis:**
    *   The selected paper is processed, and key arguments and concepts are extracted using constrained generation templates.
    *   These extractions are transformed into general knowledge question-answer pairs to facilitate knowledge generalization and broader understanding.

3.  **Debate Generation:**
    *   Using high-temperature sampling with `min_p`, the system generates diverse, opposing viewpoints for each question-answer pair.
    *   This approach encourages creative exploration and deepens understanding through debate.

4.  **Argument Synthesis:**
    *   Generated arguments are evaluated for accuracy, clarity, and generalizability by an evaluation agent that provides supporting and critical feedback.
     * The system determines which QA pairs to keep or discard based on this evaluation process.

5.  **Query Evolution:**
    *   The evaluation agent also proposes a next search query based on knowledge gaps and promising research directions found in the paper and arguments, steering the research process.

6. **Knowledge Integration:**
     * Accepted question-answer pairs are stored in a structured format for future fine-tuning of language models and are augmented with supporting and challenging arguments, facilitating a transparent and iterative knowledge-building process.

7. **LoRA Fine-tuning:**
    * A LoRA (Low-Rank Adaptation) fine-tuning process is initiated, either using our custom-generated data, or by using datasets like EQ-Bench and GPQA.
    * This fine-tuning enhances the model’s ability to understand and apply the acquired knowledge.

8. **Evaluation:** 
    * We then evaluate our fine-tuned models with the `lm-evaluation-harness` in order to track our model's performance on datasets of interest, including GPQA.

## 🎯 Performance Tracking

We use Weights & Biases to monitor the debate and research process. Key metrics tracked include:

*   **Argument Diversity Metrics:** Assessing the range of perspectives generated during debates.
*   **Knowledge Evolution Patterns:** Observing how the system refines its understanding over time.
*   **Search Query Effectiveness:** Evaluating the relevance and diversity of papers retrieved by evolving queries.
*   **Debate Quality Assessment:** Monitoring the coherence, accuracy, and depth of arguments generated.
*   **Evaluation Metrics:** tracking metrics like `eq_bench_score` and `percent_parseable` on the EQ-Bench dataset and tracking loss on GPQA.

## ⚙️ Code Structure

The repository is structured as follows:

*   `wand_university.py`: This is the primary script containing the core logic for dynamic paper analysis, structured debate, and knowledge synthesis. It is the starting point for running the Wand University knowledge acquisition system. It has the following main functionalities:
    *   **`load_arxiv_papers`**: Fetches papers from arXiv based on a given query.
    *   **`generate_qa`**: Creates question-answer pairs based on the content of a research paper, with a specified system prompt to guide the generation process.
    *   **`evaluate_qa`**: Evaluates the generated question-answer pairs, providing arguments for and against their inclusion, and recommending a next search query.
    *   **`archive_enchanted_dialogues`**: Archives the accepted QA pairs along with the supporting and challenging arguments into a structured CSV file.
    *   **`synthesize_and_evaluate_knowledge`**: Manages the overall process of knowledge synthesis and evaluation, orchestrating the debate cycles.
    *   **Main loop**: Sets up WandB tracking, initializes the language model (vLLM), iterates through research cycles, saves the knowledge, and initiates a LoRA fine-tuning and evaluation process.
*   `wand_test_eqbench.py`: Script for LoRA fine-tuning using the EQ-Bench dataset, as well as performing a final evaluation of the trained model.
*   `wand_test_gpqa.py`: Script for LoRA fine-tuning using the GPQA dataset, as well as performing a final evaluation of the trained model.
*   `wand_university_training_grimoire_test.csv`: CSV archive where the validated knowledge exchanges are stored.

## ⚠️ System Requirements

**Important:** This system requires significant computational resources:

*   **Minimum 4x NVIDIA A100 GPUs:** For parallel debate simulation using `vLLM`.
*   **Additional GPU(s):** For paper processing, LoRA fine-tuning, and evaluation.
*   **vLLM Server:** Must be launched separately to facilitate efficient language model inference.
*   **lm-evaluation-harness:** This framework is used for the evaluation of the fine-tuned models. 

## 🛠️ Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ai-wand/wand-university.git
    cd wand-university
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
   Note that this repository also assumes the existence of a configured `lm-evaluation-harness` setup. The requirements can be found in the root of that repository at `lm-evaluation-harness/requirements.txt`.

3.  **Set Up vLLM Server:**
    *   Ensure the `vLLM` server is running and accessible.
        * It should be launched with enough GPUs specified to meet your requirements.
        * You can look at the `vllm` documentation for more details on how to set this up.
   
4.  **Configure WandB:**
    *   Set up your Weights & Biases account and ensure that the API key is configured in your environment.

5.  **Run the Wand University System:**
    ```bash
     python wand_university.py
    ```
    *   This will launch the system, which iteratively searches, debates, and refines knowledge and then initiates the fine-tuning process, and finally evaluates the fine-tuned model.

6.  **Run the Standalone Fine-tuning and Evaluation Scripts (Optional):**
    *   You can optionally run these scripts separately for LoRA finetuning on EQ-Bench and GPQA:
    ```bash
        python wand_test_eqbench.py
        python wand_test_gpqa.py
    ```

## 🚀 Getting Started

1.  **Initial Research Focus:**  The system starts with the initial research focus defined in `wand_university.py`. Modify this to match your initial area of research.
2.  **Monitoring:** Monitor the progress and results via Weights & Biases dashboards. You can check the logs to see the evolution of the search queries, the debate cycles, and evaluation metrics.
3.  **Customization:** Adjust parameters, such as sampling temperature, min\_p, and debate rounds, in `wand_university.py` to experiment with different configurations.
4.  **Dataset Focus:**  The fine-tuning scripts `wand_test_eqbench.py` and `wand_test_gpqa.py` are set up to fine-tune on the respective datasets. If you want to change to your own dataset please change the code accordingly. 
5. **LoRA Rank:** If you want to change the LoRA Rank or alpha, please modify the `lora_r` and `lora_alpha` variables in the fine-tuning scripts.

## 🤝 Contributing

Contributions are welcome! Please submit a pull request with your enhancements.

## 📜 Acknowledgements

This project uses:

*   `vLLM`: For efficient language model inference.
*   `lm-format-enforcer`: For structured output generation.
*   `transformers`: For model loading and training.
*  `lm_eval`: For model evaluation.
*   `llama-index`: For interacting with arXiv papers.
*   `peft`: For Parameter Efficient Fine-Tuning (LoRA).
*   `Weights & Biases`: For experiment tracking.

Let us know if you have any questions! 
