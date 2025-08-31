#!/bin/bash
#SBATCH -p gpu_se
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o /path/to/your/DecoderLLMs-CodeSearch/Fine-tuning/Fine-tuning_method/finetuning_script/CodeGemma/evaluation_discard_CodeGemma.out

# # ==========================  ==========================
# # 1. 
# echo "Initializing Conda environment..."
# conda init bash
# source ~/.bashrc
# module load cuda/12.2.0
# export WANDB_MODE=disabled
# conda activate Your_Virtual_Environment
# echo "Environment activated."
# echo

# 2.  ()
LANGUAGES=("java" "ruby")

# 3.  ()
RATIOS=(0.1 0.2 0.5 0.8 1.0)

# 
    # BASE_MODEL_PATH
    # EXPERIMENT_NAME discard EXPERIMENT_NAME

# 4. 
# 
BASE_MODEL_PATH="/path/to/your/McGill-NLP/codegemma-7b-it"

# PEFT checkpoint  ( BASE_OUTPUT_DIR)
PEFT_MODELS_BASE_DIR="/path/to/your/llm2v/Your_Virtual_Environment_Supervised_contrastive_training/model_output"

# 
RESULTS_BASE_DIR="/path/to/your/DecoderLLMs-CodeSearch/Result_output"

# 
CSN_EVAL_SCRIPT="/path/to/your/DecoderLLMs-CodeSearch/Fine-tuning/CSN_Test_Finetuning_Decoder_Model.py"
COSQA_EVAL_SCRIPT="/path/to/your/DecoderLLMs-CodeSearch/Fine-tuning/CoSQA_Plus_Test_Finetuning_Decoder_Model.py"

# 
CSN_DATA_DIR="/path/to/your/CodeSearchNet/resources"
COSQA_DATA_DIR="/path/to/your/CoSQA+/dataset"

# 
EMBEDDING_BATCH_SIZE=500
CHECKPOINT_NAME="checkpoint-1000" #  checkpoint-1000 

# Python 
PYTHON_EXEC="$HOME/.conda/envs/Your_Virtual_Environment/bin/python"
# ~/.conda/envs/Your_Virtual_Environment/bin/python
# ========================  ========================

echo "Starting evaluation for all trained models..."
echo "Base Model: $BASE_MODEL_PATH"
echo

# : 
for LANG in "${LANGUAGES[@]}"
do
    # : 
    for RATIO in "${RATIOS[@]}"
    do
        # --- 1.  ---
        EXPERIMENT_NAME="CodeGemma-7B-Instruct-it-supervised-${LANG}-discard-${RATIO}"
        PEFT_MODEL_PATH="${PEFT_MODELS_BASE_DIR}/${EXPERIMENT_NAME}/${CHECKPOINT_NAME}"

        CSN_RESULT_PATH="${RESULTS_BASE_DIR}/CSN/${EXPERIMENT_NAME}"
        COSQA_RESULT_PATH="${RESULTS_BASE_DIR}/COSQA+/${EXPERIMENT_NAME}"
        
        echo "======================================================================="
        echo "Preparing Evaluation for:"
        echo "  - Discarding Language: $LANG"
        echo "  - Discarding Ratio:    $RATIO"
        echo "  - PEFT Model Path:     $PEFT_MODEL_PATH"
        echo "======================================================================="

        # --- 2.  ---
        if [ ! -d "$PEFT_MODEL_PATH" ]; then
            echo "!! WARNING: Checkpoint directory not found, skipping..."
            echo "!! Searched Path: $PEFT_MODEL_PATH"
            echo "======================================================================="
            echo
            echo
            continue # ，
        fi

        # 
        mkdir -p "$CSN_RESULT_PATH"
        mkdir -p "$COSQA_RESULT_PATH"

        # --- 3.  CSN  ---
        echo "Evaluating on CodeSearchNet..."
        $PYTHON_EXEC "$CSN_EVAL_SCRIPT" \
            --model_name_or_path "$BASE_MODEL_PATH" \
            --peft_model_name_or_path "$PEFT_MODEL_PATH" \
            --result_path "$CSN_RESULT_PATH" \
            --test_data_path_dir "$CSN_DATA_DIR" \
            --embedding_batch_size $EMBEDDING_BATCH_SIZE
        
        # --- 4.  CoSQA+  ---
        echo "Evaluating on CoSQA+..."
        $PYTHON_EXEC "$COSQA_EVAL_SCRIPT" \
            --model_name_or_path "$BASE_MODEL_PATH" \
            --peft_model_name_or_path "$PEFT_MODEL_PATH" \
            --result_path "$COSQA_RESULT_PATH" \
            --test_data_path_dir "$COSQA_DATA_DIR" \
            --embedding_batch_size $EMBEDDING_BATCH_SIZE

        echo "-----------------------------------------------------------------------"
        echo "Successfully finished evaluation for Language: $LANG, Ratio: $RATIO"
        echo "-----------------------------------------------------------------------"
        echo
        echo
    done
done

echo "************************************************************************"
echo "********** All evaluations completed!           **********"
echo "************************************************************************"