# Improvement Analysis for "Are Decoder-Only LLMs the Silver Bullet for Code Search?"

## 📜 Overview

This repository is the official implementation for the **"Improvement Analysis"** section of the paper, **"Are Decoder-Only Large Language Models the Silver Bullet for Code Search?"**

It contains all the data analysis and model training scripts required to reproduce the experiments from the paper. In compliance with the privacy requirements for peer review, all scripts have been desensitized by removing personal information such as local file paths.

## 📂 Directory Structure

This repository contains several independent experimental analyses, each located in a separate subdirectory:

```
Improvement_Analysis/
├── 📄 README.md                       # This guide
├── 📁 Discard_Language_Finetuning/    # VI.C. Single-Language Fine-Tuning
├── 📁 Single_Language_Finetuning/     # VI.C. Single-Language Fine-Tuning
├── 📁 Model_Size/                     # VI.D. Model Size
└── 📁 Training_Data/                  # B. Training Data
```

## 🚀 How to Run the Experiments

Each subdirectory (`Discard_Language_Finetuning`, `Single_Language_Finetuning`, etc.) contains a standalone experiment.

For the specific setup, environment preparation, and execution steps for each experiment, **please consult the `README.md` file within the corresponding subdirectory**. The guide in each subdirectory provides detailed instructions.

### Quick Navigation

* To reproduce the **"Discard Language Finetuning"** experiment, navigate to the `Discard_Language_Finetuning/` directory and follow its `README.md`.
* To reproduce the **"Single Language Finetuning"** experiment, navigate to the `Single_Language_Finetuning/` directory and follow its `README.md`.
* The same applies to the other experiments.

## 📄 How to Cite

If you use this repository or our work in your research, please cite our paper:

```bibtex
@article{chen2024decoder,
  title={Are Decoder-Only Large Language Models the Silver Bullet for Code Search?},
  author={Chen, Yuxuan and Liu, Mingwei and Ou, Guangsheng and Li, Anji and Dai, Dekun and Wang, Yanlin and Zheng, Zibin},
  journal={arXiv preprint arXiv:2410.22240},
  year={2024}
}
```