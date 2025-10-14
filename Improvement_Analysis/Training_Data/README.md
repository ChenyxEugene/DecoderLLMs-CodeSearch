# Training Data

## 📜 Overview

This directory details the data sources and core training methodology used in our paper. Our training process is primarily based on the **Supervised Contrastive Training** framework proposed by **llm2vec**.

We conducted experiments on two main datasets:
1.  **E5 (MTEB) Dataset**: A public dataset widely used for text embedding tasks.
2.  **CodeSearchNet (CSN) Dataset**: A large-scale dataset focused on code search.

This guide provides pre-trained models, data download links, and training instructions for users who wish to reproduce our experiments or train their own models.

## 🤗 Published Models (for Reproducibility)

To facilitate the reproduction of our paper's results, we provide Hugging Face Hub links for all key models. Some of the models trained on the E5 dataset are the official versions released by the `llm2vec` authors.

| Trained On | Hugging Face Model Repo                                              | Notes                               |
| :--------- | :------------------------------------------------------------------- | :---------------------------------- |
| E5         | [`McGill-NLP/gemma-2b-it-e5`](https://huggingface.co/McGill-NLP/gemma-2b-it-e5) | Released by `llm2vec` authors       |
| E5         | [`McGill-NLP/Mistral-7B-Instruct-v0.2-e5`](https://huggingface.co/McGill-NLP/Mistral-7B-Instruct-v0.2-e5) | Released by `llm2vec` authors       |
| CSN        | `YourUsername/CodeGemma-7B-SupCon-CSN`                               | Trained by the authors of this paper |
| ...        | *(Add other models you have trained and published to Hugging Face here)* | ...                                 |

> **Note**: Please replace `YourUsername/YourModelRepo` with your actual Hugging Face repository IDs.

## 🚀 Training from Scratch

If you wish to train your own models from scratch, please follow the steps below.

### Step 1 (Prerequisite): MNTP Pre-training

Before beginning supervised contrastive training, you must first train a **Masked Next Token Prediction (MNTP)** model. This is a critical step in the `llm2vec` framework that enables decoder-only models to better encode bidirectional context.

**Please follow the guide in the `DecoderLLMs-CodeSearch/Fine-tuning/Fine-tuning_method/MNTP/` directory to complete this step.**

### Step 2: Download the Datasets

Download the appropriate dataset(s) based on your research needs.

* #### E5 (MTEB) Dataset
    * **Origin**: This dataset was curated and released by the authors of [Echo Embeddings](https://github.com/jakespringer/echo-embeddings) and is widely used for text representation learning.
    * **Download Link**: [**jakespringer/echo-embeddings#training**](https://github.com/jakespringer/echo-embeddings#training)
    * **Directory Layout**: Please follow the requirements of the `llm2vec` repository and place the downloaded data in the correct cache directory.

* #### CodeSearchNet (CSN) Dataset
    * **Origin**: A large-scale dataset for code search, containing multiple programming languages.
    * **Download Link**: [**huggingface.co/datasets/code-search-net**](https://huggingface.co/datasets/code-search-net)

### Step 3: Run Supervised Contrastive Training

After completing MNTP pre-training and downloading the datasets, you can begin the supervised contrastive fine-tuning.

**Core Mechanism**: You can switch between the E5 and CSN datasets by using different JSON configuration files. This directory should contain `e5_config.json` and `csn_config.json` templates. You must edit the dataset paths within these files to point to your local directories.

#### Training on the E5 Dataset
```bash
# Assuming relevant environment variables (e.g., model paths) are set
torchrun --nproc_per_node=1 your_training_script.py \
    configs/e5_config.json \
    --output_dir /path/to/your/e5_model_output
```

#### Training on the CodeSearchNet (CSN) Dataset
```bash
# Assuming relevant environment variables are set
torchrun --nproc_per_node=1 your_training_script.py \
    configs/csn_config.json \
    --output_dir /path/to/your/csn_model_output
```

## 📝 References and Acknowledgements

Our work is heavily inspired by the following research, and we extend our sincere gratitude:
* **llm2vec**: *LLM2Vec: Large Language Models Are Good Contextual Text Encoders* ([Paper](https://arxiv.org/abs/2404.05961), [GitHub](https://github.com/McGill-NLP/llm2vec))
* **Echo Embeddings**: *Repetition Improves Language Model Embeddings* ([Paper](https://arxiv.org/abs/2402.15449), [GitHub](https://github.com/jakespringer/echo-embeddings))
* **MTEB**: *Improving Text Embeddings with Large Language Models* ([Paper](https://arxiv.org/abs/2401.00368))