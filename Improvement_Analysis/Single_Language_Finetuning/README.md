# Single-Language Fine-Tuning and Evaluation Guide

This guide provides a complete workflow for fine-tuning a base model on individual programming languages and subsequently evaluating the performance of the fine-tuned models on the CodeSearchNet and CoSQA+ benchmarks.

It is assumed that you have already activated your required virtual environment.

---

## ⚙️ Step 1: Prepare the JSON Configuration File

The training script uses a JSON file for base configuration. Please find the `single_SupCon_csn_CodeGemma.json` file in your project and open it with a text editor.

You will need to replace all placeholder paths, which are in the format `/path/to/your/...`, with the actual paths on your system.

**➡️ Action Required:**
* Update `model_name_or_path`: Points to the directory of your base LLM.
* Update `peft_model_name_or_path`: Points to a **base PEFT checkpoint**. The training will continue from this checkpoint. If you want to train PEFT layers from scratch, set its value to `null`.
* Update `dataset_file_path`: Points to the directory of your training dataset.

---

## 🖥️ Step 2: Set Environment Variables

Before running any commands, open your terminal and set the following environment variables. These paths are crucial for both training and evaluation.

**Update the paths in the block below, then copy and paste it into your terminal:**

```bash
# 1. UPDATE THIS: Set the absolute path to your project's root directory
export PROJECT_ROOT="/path/to/your/project"

# --- Foundational Model & Data Paths ---
# This path is used to load the base, non-PEFT model weights during both training and evaluation
export BASE_MODEL_PATH="/path/to/your/base_model/codegemma-7b-it"
export CSN_DATA_DIR="/path/to/your/CodeSearchNet/resources"
export COSQA_DATA_DIR="/path/to/your/CoSQA+/dataset"

# --- Script & Config Paths ---
export TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/run_supervised_training_single_lang.py"
export CSN_EVAL_SCRIPT="${PROJECT_ROOT}/scripts/CSN_Test_Finetuning_Decoder_Model.py"
export COSQA_EVAL_SCRIPT="${PROJECT_ROOT}/scripts/CoSQA_Plus_Test_Finetuning_Decoder_Model.py"
export CONFIG_FILE="${PROJECT_ROOT}/configs/single_SupCon_csn_CodeGemma.json" # Updated to your filename

# --- Output Directories ---
export BASE_OUTPUT_DIR="${PROJECT_ROOT}/model_outputs" # For new models produced by this training run
export RESULTS_BASE_DIR="${PROJECT_ROOT}/evaluation_results" # For evaluation results
```

---

## 🚀 Step 3: Run Training

This script will fine-tune the base model separately for each specified programming language.

> **Workflow Note**: The script will first load the base model from `$BASE_MODEL_PATH`, then load and apply the `peft_model_name_or_path` weights you configured in the JSON file, and start the new fine-tuning process for each language from that state.

**Copy and paste the entire block into the same terminal session:**

```bash
# Disable Weights & Biases reporting
export WANDB_MODE=disabled

# --- Experiment Parameters ---
LANGUAGES=("go" "java" "javascript" "php" "python" "ruby")
RATIOS=(1.0) # Train on 100% of the data for each language
MASTER_PORT=29545

# --- Training Execution ---
mkdir -p "$BASE_OUTPUT_DIR"
echo "All trained models will be saved under: $BASE_OUTPUT_DIR"
echo

for LANG in "${LANGUAGES[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        EXPERIMENT_NAME="Codegemma-7B-Instruct-it-supervised-single-${LANG}-ratio-${RATIO}"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"

        echo "======================================================================="
        echo "Starting Training: Language=$LANG, Ratio=$RATIO"
        echo "Output Directory: $OUTPUT_DIR"
        echo "======================================================================="

        torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} \
            "$TRAIN_SCRIPT" \
            "$CONFIG_FILE" \
            --model_name_or_path "$BASE_MODEL_PATH" \
            --use_language "$LANG" \
            --use_ratio "$RATIO" \
            --output_dir "$OUTPUT_DIR" \
            --overwrite_output_dir

        if [ $? -ne 0 ]; then
            echo "!! TRAINING FAILED for Language: $LANG, Ratio: $RATIO !!"
            exit 1
        fi
        echo "--- Finished experiment for Language: $LANG, Ratio: $RATIO ---"
        echo
    done
done

echo "********** All training experiments completed successfully! **********"
```

---

## 📊 Step 4: Run Evaluation

After training, this script will iterate through the models you fine-tuned in **Step 3** and evaluate them on both CodeSearchNet and CoSQA+.

**Copy and paste the entire block into the same terminal session:**

```bash
# --- Evaluation Parameters ---
LANGUAGES=("go" "java" "javascript" "php" "python" "ruby")
RATIOS=(1.0)
EMBEDDING_BATCH_SIZE=500
CHECKPOINT_NAME="checkpoint-1000" # Checkpoint to use for evaluation

# --- Evaluation Execution ---
echo "Starting evaluation for all trained models..."
echo

for LANG in "${LANGUAGES[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        EXPERIMENT_NAME="Codegemma-7B-Instruct-it-supervised-single-${LANG}-ratio-${RATIO}"
        PEFT_MODEL_PATH="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}/${CHECKPOINT_NAME}"

        # Check if the model checkpoint exists
        if [ ! -d "$PEFT_MODEL_PATH" ]; then
            echo "!! WARNING: Checkpoint not found, skipping: $PEFT_MODEL_PATH"
            continue
        fi

        echo "======================================================================="
        echo "Evaluating Model for Language: $LANG, Ratio: $RATIO"
        echo "  - PEFT Model Path: $PEFT_MODEL_PATH"
        echo "======================================================================="

        # --- Evaluate on CodeSearchNet ---
        CSN_RESULT_PATH="${RESULTS_BASE_DIR}/CSN/${EXPERIMENT_NAME}"
        mkdir -p "$CSN_RESULT_PATH"
        echo "Evaluating on CodeSearchNet..."
        python "$CSN_EVAL_SCRIPT" \
            --model_name_or_path "$BASE_MODEL_PATH" \
            --peft_model_name_or_path "$PEFT_MODEL_PATH" \
            --result_path "$CSN_RESULT_PATH" \
            --test_data_path_dir "$CSN_DATA_DIR" \
            --embedding_batch_size $EMBEDDING_BATCH_SIZE
        
        # --- Evaluate on CoSQA+ ---
        COSQA_RESULT_PATH="${RESULTS_BASE_DIR}/COSQA+/${EXPERIMENT_NAME}"
        mkdir -p "$COSQA_RESULT_PATH"
        echo "Evaluating on CoSQA+..."
        python "$COSQA_EVAL_SCRIPT" \
            --model_name_or_path "$BASE_MODEL_PATH" \
            --peft_model_name_or_path "$PEFT_MODEL_PATH" \
            --result_path "$COSQA_RESULT_PATH" \
            --test_data_path_dir "$COSQA_DATA_DIR" \
            --embedding_batch_size $EMBEDDING_BATCH_SIZE

        echo "--- Finished evaluation for Language: $LANG, Ratio: $RATIO ---"
        echo
    done
done

echo "********** All evaluations completed successfully! **********"
```