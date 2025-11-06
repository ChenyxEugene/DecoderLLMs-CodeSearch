# Project Training and Evaluation Guide

This guide provides all necessary instructions to run the training and evaluation workflows for this project. It is assumed that you have already activated your required virtual environment. 

## 🤖 Hugging Face ID of TABLE VIII: Performance of Fine-Tuned CodeGemma Models by Discarding Language-Specific Data
If you want to reproduce the results of the article, you can directly use the following models.

| Model | Discard Language (SupCon) | Discard Ratio (SupCon) | Hugging Face ID |
|---|---|---|---|
| CodeGemma | Java | 0 | https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN |
| CodeGemma | Java | 0.2 | https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-java-discard-0.2 |
| CodeGemma | Java | 0.5 | https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-java-discard-0.5 |
| CodeGemma | Java | 0.8 | https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-java-discard-0.8 |
| CodeGemma | Java | 1 | https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-java-discard-1.0 |
| CodeGemma | Ruby | 0 | https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN |
| CodeGemma | Ruby | 0.2 | https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-ruby-discard-0.2 |
| CodeGemma | Ruby | 0.5 | https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-ruby-discard-0.5 |
| CodeGemma | Ruby | 0.8 | https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-ruby-discard-0.8 |
| CodeGemma | Ruby | 1 | https://huggingface.co/SYSUSELab/DCS-CodeGemma-7B-It-SupCon-CSN-ruby-discard-1.0 |


## ⚙️ Step 1: Configure the JSON File

The parameters for the training script are configured through a JSON file.

Please find the `discard_SupCon_csn_CodeGemma.json` file in the project and open it with a text editor. You will need to replace all placeholder paths, which are in the format `/path/to/your/...`, with the actual paths on your system.

**➡️ Action Required:**
* Update `model_name_or_path` to point to your base model.
* Update `peft_model_name_or_path` to point to your PEFT fine-tuned weights.
* Update `dataset_file_path` to point to your training dataset.

---

## 🖥️ Step 2: Set Environment Variables

Before running the scripts, open your terminal and set the following environment variables. These variables define the key file paths for the current terminal session.

**After updating the path for `PROJECT_ROOT`, copy and paste this block into your terminal:**

```bash
# 1. UPDATE THIS: Set the absolute path to your project's root directory
export PROJECT_ROOT="/path/to/your/project"

# 2. Set paths for other scripts and directories based on the root
export TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/run_supervised_training.py"
export EVAL_SCRIPT="${PROJECT_ROOT}/scripts/run_evaluation.py"
export CONFIG_FILE="${PROJECT_ROOT}/configs/discard_SupCon_csn_CodeGemma.json" # <-- NOTE: Filename has been updated
export BASE_OUTPUT_DIR="${PROJECT_ROOT}/model_outputs"
export RESULTS_DIR="${PROJECT_ROOT}/evaluation_results"
```
> **Note**: The `CONFIG_FILE` path in the code block above has been updated to your specified `discard_SupCon_csn_CodeGemma.json` file to ensure consistency.

---

## 🚀 Step 3: Run Training

With the environment variables configured, you are ready to start training. The following script will loop through predefined programming languages and data discard ratios, launching a training job for each combination.

**Copy and paste the entire block into the same terminal session:**

```bash
# Disable Weights & Biases reporting
export WANDB_MODE=disabled

# --- Experiment Parameters ---
LANGUAGES=("java" "ruby")
RATIOS=(0.2 0.5 0.8 1.0)
MASTER_PORT=29513 # Port for torchrun

# --- Training Execution ---
# Create the base output directory if it doesn't exist
mkdir -p "$BASE_OUTPUT_DIR"
echo "All model outputs will be saved to: $BASE_OUTPUT_DIR"
echo

# Start the experiment loop
for LANG in "${LANGUAGES[@]}"
do
    for RATIO in "${RATIOS[@]}"
    do
        # Define a unique name and output directory for this experiment run
        EXPERIMENT_NAME="CodeGemma-7B-Instruct-it-supervised-${LANG}-discard-${RATIO}"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"

        echo "======================================================================="
        echo "Starting Training: Language=$LANG, Ratio=$RATIO"
        echo "Output Directory: $OUTPUT_DIR"
        echo "======================================================================="

        # Launch the training process
        torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} \
            "$TRAIN_SCRIPT" \
            "$CONFIG_FILE" \
            --discard_language "$LANG" \
            --discard_ratio "$RATIO" \
            --output_dir "$OUTPUT_DIR" \
            --overwrite_output_dir

        # Check if the last command succeeded
        if [ $? -ne 0 ]; then
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "!!           TRAINING FAILED for Language: $LANG, Ratio: $RATIO"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            exit 1 # Exit the script if an error occurs
        fi
        echo "--- Finished experiment for Language: $LANG, Ratio: $RATIO ---"
        echo
    done
done

echo "************************************************************************"
echo "********** All training experiments completed successfully! **********"
echo "************************************************************************"
```

---

## 📊 Step 4: Run Evaluation

After training is complete, use the following script to evaluate the generated models.

> **Important**: This is a template. You must modify the final `python "$EVAL_SCRIPT" ...` command to pass the correct arguments required by your specific `run_evaluation.py` script.

**Copy and paste the entire block into the same terminal session:**

```bash
# --- Evaluation Execution ---
# Create the results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"
echo "All evaluation results will be saved to: $RESULTS_DIR"
echo

# Start the evaluation loop
for LANG in "${LANGUAGES[@]}"
do
    for RATIO in "${RATIOS[@]}"
    do
        # Define the model path based on the experiment name
        EXPERIMENT_NAME="CodeGemma-7B-Instruct-it-supervised-${LANG}-discard-${RATIO}"
        MODEL_PATH="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"

        # Skip if the model directory doesn't exist
        if [ ! -d "$MODEL_PATH" ]; then
            echo "Warning: Model directory not found, skipping: $MODEL_PATH"
            continue
        fi

        echo "======================================================================="
        echo "Starting Evaluation for: $MODEL_PATH"
        echo "======================================================================="

        # ---
        # ➡️ Action Required: Customize the command below for your evaluation script
        # ---
        python "$EVAL_SCRIPT" \
            --model_name_or_path "$MODEL_PATH" \
            --output_file "${RESULTS_DIR}/${EXPERIMENT_NAME}_results.json" \
            --language "$LANG" # Add/remove other arguments as needed

        # Check for errors
        if [ $? -ne 0 ]; then
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "!!           EVALUATION FAILED for Model: $EXPERIMENT_NAME"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            exit 1
        fi
        echo "--- Finished evaluation for Model: $EXPERIMENT_NAME ---"
        echo
    done
done

echo "************************************************************************"
echo "********** All evaluation tasks completed successfully! **********"
echo "************************************************************************"
```