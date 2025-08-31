#!/bin/bash
#SBATCH -p gpu_se
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o /path/to/your/DecoderLLMs-CodeSearch/Fine-tuning/Fine-tuning_method/finetuning_script/CodeGemma/discard_CodeGemma.out



# # 
# set -e

# 
    #   CONFIG_FILE  JSON 
    #   MASTER_PORT 
    #   EXPERIMENT_NAME 
    #   


# ==========================  ==========================
# Conda
conda init bash
source ~/.bashrc
module load cuda/12.2.0
export WANDB_MODE=disabled  # wand_BTrainer
conda activate Your_Virtual_Environment
# 1. 
LANGUAGES=("java" "ruby")

# 2. 
RATIOS=(0.1 0.2 0.5 0.8 1.0)

# 3.  Python 
TRAIN_SCRIPT="/path/to/your/llm2v/Your_Virtual_Environment_Supervised_contrastive_training/run_test/run_supervised_discard_language.py"

# 4.  JSON 
CONFIG_FILE="/path/to/your/DecoderLLMs-CodeSearch/Fine-tuning/Fine-tuning_method/finetuning_script/CodeGemma/discard_supervised_csn_CodeGemma.json"

# 5.  ()
BASE_OUTPUT_DIR="/path/to/your/llm2v/Your_Virtual_Environment_Supervised_contrastive_training/model_output"

# 6. torchrun 
MASTER_PORT=29513

# ========================  ========================

# 
mkdir -p "$BASE_OUTPUT_DIR"
echo "All experiment outputs will be saved under: $BASE_OUTPUT_DIR"
echo

# : 
for LANG in "${LANGUAGES[@]}"
do
    # : 
    for RATIO in "${RATIOS[@]}"
    do
        # --- 1.  ---
        # ，、
        # : .../output/Qwen2.5-it-supervised-java-discard-0.5
        # 
        EXPERIMENT_NAME="CodeGemma-7B-Instruct-it-supervised-${LANG}-discard-${RATIO}"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"

        echo "======================================================================="
        echo "Starting Experiment:"
        echo "  - Discarding Language: $LANG"
        echo "  - Discarding Ratio:    $RATIO"
        echo "  - Output Directory:    $OUTPUT_DIR"
        echo "======================================================================="

        # --- 2.  torchrun  ---
        # ，:
        #   --discard_language
        #   --discard_ratio
        #   --output_dir
        torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} \
            "$TRAIN_SCRIPT" \
            "$CONFIG_FILE" \
            --discard_language "$LANG" \
            --discard_ratio "$RATIO" \
            --output_dir "$OUTPUT_DIR" \
            --overwrite_output_dir

        # --- 3.  ---
        # $? ，0 ， 0 
        if [ $? -ne 0 ]; then
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "!!           EXPERIMENT FAILED!                                      !!"
            echo "!!   Language: $LANG, Ratio: $RATIO"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            # ，
            exit 1
        fi

        echo "-----------------------------------------------------------------------"
        echo "Successfully finished experiment for Language: $LANG, Ratio: $RATIO"
        echo "-----------------------------------------------------------------------"
        echo
        echo
    done
done

echo "************************************************************************"
echo "********** All experiments completed!           **********"
echo "************************************************************************"



