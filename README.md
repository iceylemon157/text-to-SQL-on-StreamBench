# Text To SQL Generation Agent on StreamBench

This repository contains the implementation of the **Text-to-SQL Generation Agent** developed as part of the NTU ADL 2024 Final Project. The agent was evaluated on the [StreamBench](https://arxiv.org/abs/2406.08747) benchmark, designed for evaluating continuous improvement of LLM agents.

## Competition Results (team: group_36):
- [Kaggle Public Leaderboard](https://www.kaggle.com/competitions/adl-2024-final-project-text-to-sql-generation): 3rd place
- [Kaggle Private Leaderboard](https://www.kaggle.com/competitions/adl-2024-final-project-text-to-sql-private): 9th place

## Project Report

- SQL Generation Part [Slide Link](https://docs.google.com/presentation/d/1cxOC1Ty4OdknZjYtJKw_RPryWZSzCTFTeT4wS3dNzJg/edit?usp=sharing)
- Full Project Report [Slide Link](https://docs.google.com/presentation/d/1SUm9HwWxIHVPVa1FS5H5_9aijX96y3-XQoGlPZ0MoFI/edit?usp=sharing)

## Introduction

This repository focuses on the **Text-to-SQL generation task** from the NTU ADL 2024 final project. The goal is to build an Text-to-SQL generation agent that is evaluated on the [StreamBench](https://arxiv.org/abs/2406.08747) benchmark.

To improve performance, we integrate several components:
- **StreamICL (Streaming In-C.ontext Learning)**: Enables few-shot prompting by retrieving top-k relevant examples from RAG.
- **Schema-Specific RAG Memory**: Organizes past examples by schema to improve retrieval relevance.
- **Iterative QLoRA Fine-Tuning**: QLoRA is trained iteratively after every 32 correct examples to continuously refine the model.

## Environment Setup

Ensure you have Python 3.10 or above installed on your system.

Next, it's recommended to create a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Then, install the required packages:

```
pip install -r requirements.txt
```

### Dataset Setup

Run the following command to set up the necessary datasets:

```
python setup_data.py
```

## Running the code

To train the model for SQL generation task, you can simply execute `./run.sh`, or use the following command:

```bash
python lora_streamicl-sql.py --bench_name sql_generation_public --model_name meta-llama/Llama-3.1-8B-Instruct
```