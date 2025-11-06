# Fine-Tuned Models for "Are Decoder-Only Large Language Models the Silver Bullet for Code Search?"

This repository contains the fine-tuned models evaluated in our study . We provide these models to promote further research in the code search domain.

The complete Hugging Face Collection, which includes all models listed below, is available here:
**[SYSUSELab/are-decoder-only-llms-the-silver-bullet](https://huggingface.co/collections/SYSUSELab/are-decoder-only-llms-the-silver-bullet)**



## 📚 Table of Contents


- [Fine-Tuned Models for "Are Decoder-Only Large Language Models the Silver Bullet for Code Search?"](#fine-tuned-models-for-are-decoder-only-large-language-models-the-silver-bullet-for-code-search)
  - [📚 Table of Contents](#-table-of-contents)
  - [🤖 Hugging Face ID of TABLE III: Zero-Shot Performance on Code Search Benchmarks](#-hugging-face-id-of-table-iii-zero-shot-performance-on-code-search-benchmarks)
  - [🤖 Hugging Face ID of TABLE IV: Performance of Fine-Tuned Models on Code Search Benchmarks](#-hugging-face-id-of-table-iv-performance-of-fine-tuned-models-on-code-search-benchmarks)
  - [🤖 Hugging Face ID of TABLE V: Results of Different Fine-Tuning Methods](#-hugging-face-id-of-table-v-results-of-different-fine-tuning-methods)
  - [🤖 Hugging Face ID of TABLE VI: Results of Different Fine-Tuning Datasets](#-hugging-face-id-of-table-vi-results-of-different-fine-tuning-datasets)
  - [🤖 Hugging Face ID of TABLE VII: CodeGemma: Multi- vs. Single-language Tuning for Code Search](#-hugging-face-id-of-table-vii-codegemma-multi--vs-single-language-tuning-for-code-search)
  - [🤖 Hugging Face ID of TABLE VIII: Performance of Fine-Tuned CodeGemma Models by Discarding Language-Specific Data](#-hugging-face-id-of-table-viii-performance-of-fine-tuned-codegemma-models-by-discarding-language-specific-data)
  - [🤖 Hugging Face ID of TABLE IX: Results of Different Model Sizes](#-hugging-face-id-of-table-ix-results-of-different-model-sizes)


## 🤖 Hugging Face ID of TABLE III: Zero-Shot Performance on Code Search Benchmarks
These models correspond to the base models evaluated for zero-shot performance.

| Category | Model | Hugging Face ID |
|---|---|---|
| Encoder-Only | UniXcoder | [microsoft/unixcoder-base](https://huggingface.co/microsoft/unixcoder-base) |
| Encoder-Only | CodeBERT | [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base) |
| General LLM | Llama3 | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
| General LLM | Mistral | [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |
| General LLM | DeepSeekLLM| [deepseek-ai/deepseek-llm-7b-chat](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) |
| General LLM | Gemma | [google/gemma-7b-it](https://huggingface.co/google/gemma-7b-it) |
| General LLM | Llama2 | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| Code LLM | Qwen2.5-Coder | [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) |
| Code LLM | StarCoder2 | [bigcode/starcoder2-7b](https://huggingface.co/bigcode/starcoder2-7b) |
| Code LLM | CodeMistral | [uukuguy/speechless-code-mistral-7b-v1.0](https://huggingface.co/uukuguy/speechless-code-mistral-7b-v1.0) |
| Code LLM | DeepSeekCoder | [deepseek-ai/deepseek-coder-6.7b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) |
| Code LLM | CodeGemma | [google/codegemma-7b-it](https://huggingface.co/google/codegemma-7b-it) |
| Code LLM | CodeLlama | [meta-llama/CodeLlama-7b-hf](https://huggingface.co/meta-llama/CodeLlama-7b-hf) |


## 🤖 Hugging Face ID of TABLE IV: Performance of Fine-Tuned Models on Code Search Benchmarks

These models were fine-tuned using Supervised Contrastive Learning (SupCon) on the CSN dataset.

| Category | Sup Model | Hugging Face ID |
|---|---|---|
| Decoder-Only | CodeGemma | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN) |
| Decoder-Only | Gemma | [SYSUSELab/DCS-Gemma-7b-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Gemma-7b-It-SupCon-CSN) |
| Decoder-Only | DeepSeekCoder| [SYSUSELab/DCS-DeepSeekCoder-6.7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-DeepSeekCoder-6.7B-It-SupCon-CSN) |
| Decoder-Only | DeepSeekLLM| [SYSUSELab/DCS-DeepSeekLLM-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-DeepSeekLLM-7B-It-SupCon-CSN) |
| Decoder-Only | Qwen2.5-Coder| [SYSUSELab/DCS-Qwen2.5-Coder-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Qwen2.5-Coder-7B-It-SupCon-CSN) |
| Decoder-Only | CodeLlama | [SYSUSELab/DCS-CodeLlama-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-CodeLlama-7B-It-SupCon-CSN) |
| Decoder-Only | CodeMistral | [SYSUSELab/DCS-CodeMistral-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-CodeMistral-7B-It-SupCon-CSN) |
| Decoder-Only | Llama3 | [SYSUSELab/DCS-Llama3-8B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Llama3-8B-It-SupCon-CSN) |
| Decoder-Only | Mistral | [SYSUSELab/DCS-Mistral-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Mistral-7B-It-SupCon-CSN) |
| Decoder-Only | Llama2 | [SYSUSELab/DCS-Llama2-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Llama2-7B-It-SupCon-CSN) |


## 🤖 Hugging Face ID of TABLE V: Results of Different Fine-Tuning Methods

| Sup Model | Finetuning |Hugging Face ID  |
|---|---| ---|
| CodeGemma | Zero-Shot | [google/codegemma-7b-it](https://huggingface.co/google/codegemma-7b-it) |
| CodeGemma | SimCSE |[SYSUSELab/DCS-CodeGemma-7B-It-SimCSE](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SimCSE)  |
| CodeGemma | Sup | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN) |
| Llama3 | Zero-Shot | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
| Llama3 | SimCSE | [McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse)|
| Llama3 | Sup | [SYSUSELab/DCS-Llama3-8B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Llama3-8B-It-SupCon-CSN) |
| Mistral | Zero-Shot |  [mistralai/Mistral-7B-v0.2](https://huggingface.co/mistralai/Mistral-7B-v0.2)  |
| Mistral | SimCSE |[McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse](https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse) |
| Mistral | Sup | [SYSUSELab/DCS-Mistral-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Mistral-7B-It-SupCon-CSN) |

## 🤖 Hugging Face ID of TABLE VI: Results of Different Fine-Tuning Datasets

| Sup Model | Dataset | Hugging Face ID |
|---|---|---|
| CodeGemma | E5 | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-E5](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-E5) |
| CodeGemma | CSN | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN) |
| Llama3 | E5 | [McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised) |
| Llama3 | CSN | [SYSUSELab/DCS-Llama3-8B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Llama3-8B-It-SupCon-CSN) |
| Mistral | E5 | [McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised](https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised) |
| Mistral | CSN | [SYSUSELab/DCS-Mistral-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Mistral-7B-It-SupCon-CSN) |

## 🤖 Hugging Face ID of TABLE VII: CodeGemma: Multi- vs. Single-language Tuning for Code Search

| Model | Training Language | Hugging Face ID |
|---|---|---|
| Codegemma | Ruby | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-ruby](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-ruby) |
| Codegemma | Javascript | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-javascript](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-javascript) |
| Codegemma | Go | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-go](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-go) |
| Codegemma | Python | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-python](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-python) |
| Codegemma | Java | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-java](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-java) |
| Codegemma | Php | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-php](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-php) |
| Codegemma | Multi-language | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN) |


## 🤖 Hugging Face ID of TABLE VIII: Performance of Fine-Tuned CodeGemma Models by Discarding Language-Specific Data

| Model | Discard Language (SupCon) | Discard Ratio (SupCon) | Hugging Face ID |
|---|---|---|---|
| CodeGemma | Java | 0 | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN) |
| CodeGemma | Java | 0.2 | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-java-discard-0.2](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-java-discard-0.2) |
| CodeGemma | Java | 0.5 | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-java-discard-0.5](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-java-discard-0.5) |
| CodeGemma | Java | 0.8 | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-java-discard-0.8](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-java-discard-0.8) |
| CodeGemma | Java | 1 | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-java-discard-1.0](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-java-discard-1.0) |
| CodeGemma | Ruby | 0 | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN) |
| CodeGemma | Ruby | 0.2 | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-ruby-discard-0.2](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-ruby-discard-0.2) |
| CodeGemma | Ruby | 0.5 | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-ruby-discard-0.5](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-ruby-discard-0.5) |
| CodeGemma | Ruby | 0.8 | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-ruby-discard-0.8](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-ruby-discard-0.8) |
| CodeGemma | Ruby | 1 | [SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-ruby-discard-1.0](https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-ruby-discard-1.0) |


## 🤖 Hugging Face ID of TABLE IX: Results of Different Model Sizes

| Sup Model | Size | Hugging Face ID |
|---|---|---|
| Llama2 | 1.3B | [SYSUSELab/DCS-Llama2-1.3B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Llama2-1.3B-It-SupCon-CSN) |
| Llama2 | 7B | [SYSUSELab/DCS-Llama2-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Llama2-7B-It-SupCon-CSN) |
| Llama2 | 13B | [SYSUSELab/DCS-Llama2-13B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Llama2-13B-It-SupCon-CSN) |
| Qwen2.5-Coder | 0.5B | [SYSUSELab/DCS-Qwen2.5-Coder-0.5B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Qwen2.5-Coder-0.5B-It-SupCon-CSN) |
| Qwen2.5-Coder | 1.5B | [SYSUSELab/DCS-Qwen2.5-Coder-1.5B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Qwen2.5-Coder-1.5B-It-SupCon-CSN) |
| Qwen2.5-Coder | 3B | [SYSUSELab/DCS-Qwen2.5-Coder-3B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Qwen2.5-Coder-3B-It-SupCon-CSN) |
| Qwen2.5-Coder | 7B | [SYSUSELab/DCS-Qwen2.5-Coder-7B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Qwen2.5-Coder-7B-It-SupCon-CSN) |
| Qwen2.5-Coder | 14B | [SYSUSELab/DCS-Qwen2.5-Coder-14B-It-SupCon-CSN](https://huggingface.co/SYSUSELab/DCS-Qwen2.5-Coder-14B-It-SupCon-CSN) |


🎓 How to Cite
If you use our models or findings in your research, please cite our paper:

```
@article{chen2024decoder,
  title={Are Decoder-Only Large Language Models the Silver Bullet for Code Search?},
  author={Chen, Yuxuan and Liu, Mingwei and Ou, Guangsheng and Li, Anji and Dai, Dekun and Wang, Yanlin and Zheng, Zibin},
  journal={arXiv preprint arXiv:2410.22240},
  year={2024}
}
```