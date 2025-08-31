#!/bin/bash
#SBATCH -p gpu_se
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o /share/home/chenyuxuan/Research_CodeSearch/DecoderLLMs-CodeSearch/Fine-tuning/Fine-tuning_method/finetuning_script/CodeGemma/single_test_CodeGemma.out

# # ========================== 用户配置区域 ==========================
# # 1. 初始化环境
# echo "Initializing Conda environment..."
# conda init bash
# source ~/.bashrc
# module load cuda/12.2.0
# export WANDB_MODE=disabled
# conda activate LLM2Vec
# echo "Environment activated."
# echo

# 2. 定义要进行实验的语言数组 (与您的训练脚本保持一致)
LANGUAGES=("go" "java" "javascript" "php" "python" "ruby")

# 3. 定义要测试的舍弃比例数组 (与您的训练脚本保持一致)
# RATIOS=(0.1 0.2 0.5 0.8 1.0)
RATIOS=(1.0)

# 4. 固定路径和参数配置
# 基础大模型的路径
BASE_MODEL_PATH="/share/home/chenyuxuan/Research_CodeSearch/McGill-NLP/codegemma-7b-it"

# 所有PEFT模型 checkpoint 存放的基础目录 (即训练脚本中的 BASE_OUTPUT_DIR)
PEFT_MODELS_BASE_DIR="/share/home/chenyuxuan/Research_CodeSearch/llm2v/llm2vec_Supervised_contrastive_training/model_output"

# 评测结果存放的基础目录
RESULTS_BASE_DIR="/share/home/chenyuxuan/Research_CodeSearch/DecoderLLMs-CodeSearch/Result_output"

# 评测脚本的路径
CSN_EVAL_SCRIPT="/share/home/chenyuxuan/Research_CodeSearch/DecoderLLMs-CodeSearch/Fine-tuning/CSN_Test_Finetuning_Decoder_Model.py"
COSQA_EVAL_SCRIPT="/share/home/chenyuxuan/Research_CodeSearch/DecoderLLMs-CodeSearch/Fine-tuning/CoSQA_Plus_Test_Finetuning_Decoder_Model.py"

# 测试数据集的路径
CSN_DATA_DIR="/share/home/chenyuxuan/Research_CodeSearch/CodeSearchNet/resources"
COSQA_DATA_DIR="/share/home/chenyuxuan/Research_CodeSearch/CoSQA+/dataset"

# 其他参数
EMBEDDING_BATCH_SIZE=500
CHECKPOINT_NAME="checkpoint-1000" # 假设您总是使用 checkpoint-1000 进行评测

# Python 解释器路径
PYTHON_EXEC="$HOME/.conda/envs/LLM2Vec/bin/python"
# ~/.conda/envs/LLM2Vec/bin/python
# ======================== 脚本执行区域 ========================

echo "Starting evaluation for all trained models..."
echo "Base Model: $BASE_MODEL_PATH"
echo

# 外层循环: 遍历每一种语言
for LANG in "${LANGUAGES[@]}"
do
    # 内层循环: 遍历每一种比例
    for RATIO in "${RATIOS[@]}"
    do
        # --- 1. 准备本次评测的动态路径 ---
        EXPERIMENT_NAME="Codegemma-7B-Instruct-it-supervised-single-${LANG}-ratio-${RATIO}"
        PEFT_MODEL_PATH="${PEFT_MODELS_BASE_DIR}/${EXPERIMENT_NAME}/${CHECKPOINT_NAME}"

        CSN_RESULT_PATH="${RESULTS_BASE_DIR}/CSN/${EXPERIMENT_NAME}"
        COSQA_RESULT_PATH="${RESULTS_BASE_DIR}/COSQA+/${EXPERIMENT_NAME}"
        
        echo "======================================================================="
        echo "Preparing Evaluation for:"
        echo "  - Single Language: $LANG"
        echo "  - Ratio:    $RATIO"
        echo "  - PEFT Model Path:     $PEFT_MODEL_PATH"
        echo "======================================================================="

        # --- 2. 检查模型路径是否存在 ---
        if [ ! -d "$PEFT_MODEL_PATH" ]; then
            echo "!! WARNING: Checkpoint directory not found, skipping..."
            echo "!! Searched Path: $PEFT_MODEL_PATH"
            echo "======================================================================="
            echo
            echo
            continue # 跳过当前循环，继续下一个
        fi

        # 确保结果目录存在
        mkdir -p "$CSN_RESULT_PATH"
        mkdir -p "$COSQA_RESULT_PATH"

        # --- 3. 执行 CSN 评测 ---
        echo "Evaluating on CodeSearchNet..."
        $PYTHON_EXEC "$CSN_EVAL_SCRIPT" \
            --model_name_or_path "$BASE_MODEL_PATH" \
            --peft_model_name_or_path "$PEFT_MODEL_PATH" \
            --result_path "$CSN_RESULT_PATH" \
            --test_data_path_dir "$CSN_DATA_DIR" \
            --embedding_batch_size $EMBEDDING_BATCH_SIZE
        
        # --- 4. 执行 CoSQA+ 评测 ---
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