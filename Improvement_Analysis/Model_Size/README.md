# Model Size Analysis

## 📜 Overview

This directory contains the evaluation scripts to reproduce the experiments from **Section VI.D "Model Size"** of our paper.

The core of this analysis is to evaluate the performance of models of different sizes after fine-tuning. To facilitate reproducibility, we have uploaded all PEFT adapters, fine-tuned using the **SupCon (Supervised Contrastive Learning)** method, to the Hugging Face Hub.

You are **not required to retrain** the models. You can directly download them from the Hub and run the evaluation.

## 🤗 Published Models

## 🤖 Hugging Face ID of TABLE IX: Results of Different Model Sizes
If you want to reproduce the results of the article, you can directly use the following models.

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
## 🚀 How to Run Evaluation

The evaluation process consists of two steps: setting environment variables and executing the evaluation scripts.

### 1. Set Environment Variables

Before running the evaluation, open a terminal and set the following environment variables. **Please be sure to modify these paths to match your local environment.**

```bash
# --- Evaluation Config: Choose the model you want to evaluate ---
export BASE_MODEL_NAME="Qwen/Qwen2.5-Coder-3B-Instruct"
export PEFT_MODEL_NAME="SYSUSELab/Qwen2.5-Coder-3B-SupCon"


# --- Local Path Config: Modify these to match your environment ---
# Your project's root directory
export PROJECT_ROOT="/path/to/your/DecoderLLMs-CodeSearch"
# Directory for the CodeSearchNet dataset
export CSN_DATA_DIR="/path/to/your/CodeSearchNet/resources"
# Directory for the CoSQA+ dataset
export COSQA_DATA_DIR="/path/to/your/CoSQA+/dataset"


# --- Script Paths ---
export CSN_EVAL_SCRIPT="${PROJECT_ROOT}/Fine-tuning/CSN_Test_Finetuning_Decoder_Model.py"
export COSQA_EVAL_SCRIPT="${PROJECT_ROOT}/Fine-tuning/CoSQA_Plus_Test_Finetuning_Decoder_Model.py"

# --- Results Output Directory ---
export RESULTS_BASE_DIR="${PROJECT_ROOT}/Result_output"
```

### 2. Execute the Evaluation Scripts

After setting the environment variables, copy and paste the entire code block below into the same terminal. The script will automatically download the models from the Hugging Face Hub and evaluate them on the CodeSearchNet and CoSQA+ datasets sequentially.

```bash
# Extract a short name from the model name for the results directory
MODEL_SHORT_NAME=$(basename ${BASE_MODEL_NAME})
CSN_RESULT_PATH="${RESULTS_BASE_DIR}/CSN/SupCon-${MODEL_SHORT_NAME}"
COSQA_RESULT_PATH="${RESULTS_BASE_DIR}/COSQA+/SupCon-${MODEL_SHORT_NAME}"

# Ensure results directories exist
mkdir -p "$CSN_RESULT_PATH"
mkdir -p "$COSQA_RESULT_PATH"

echo "======================================================================="
echo "Starting Evaluation for: ${BASE_MODEL_NAME}"
echo "PEFT Adapter: ${PEFT_MODEL_NAME}"
echo "======================================================================="

# --- Evaluate on CodeSearchNet ---
echo "Evaluating on CodeSearchNet..."
python "$CSN_EVAL_SCRIPT" \
    --model_name_or_path "$BASE_MODEL_NAME" \
    --peft_model_name_or_path "$PEFT_MODEL_NAME" \
    --result_path "$CSN_RESULT_PATH" \
    --test_data_path_dir "$CSN_DATA_DIR" \
    --embedding_batch_size 500

# --- Evaluate on CoSQA+ ---
echo "Evaluating on CoSQA+..."
python "$COSQA_EVAL_SCRIPT" \
    --model_name_or_path "$BASE_MODEL_NAME" \
    --peft_model_name_or_path "$PEFT_MODEL_NAME" \
    --result_path "$COSQA_RESULT_PATH" \
    --test_data_path_dir "$COSQA_DATA_DIR" \
    --embedding_batch_size 500

echo "======================================================================="
echo "Evaluation for ${MODEL_SHORT_NAME} completed."
echo "CSN results saved to: ${CSN_RESULT_PATH}"
echo "CoSQA+ results saved to: ${COSQA_RESULT_PATH}"
echo "======================================================================="
```

## 📝 Note on Training Methodology

The models used in this directory were trained using the supervised contrastive learning method found in `DecoderLLMs-CodeSearch/Fine-tuning/Fine-tuning_method/SupCon/`. For details on the training process, please refer to the instructions in that directory.