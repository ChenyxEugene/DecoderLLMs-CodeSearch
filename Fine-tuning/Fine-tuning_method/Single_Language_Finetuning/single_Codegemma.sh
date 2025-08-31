#!/bin/bash
#SBATCH -p gpu_se
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o /share/home/chenyuxuan/Research_CodeSearch/DecoderLLMs-CodeSearch/Fine-tuning/Fine-tuning_method/finetuning_script/CodeGemma/single_Codegemma.out


# 遇到错误立即退出
set -e

# ========================== 用户配置区域 ==========================
# 初始化环境
conda init bash
source ~/.bashrc
module load cuda/12.2.0
export WANDB_MODE=disabled
conda activate LLM2Vec

# 1. 定义要进行实验的全部语言数组
LANGUAGES=("go" "java" "javascript" "php" "python" "ruby")

# 2. 定义要为每种语言测试的保留比例数组
# RATIOS=(0.1 0.2 0.5 0.8 1.0)
RATIOS=(1.0)
# 3. Python 训练脚本路径
TRAIN_SCRIPT="/share/home/chenyuxuan/Research_CodeSearch/llm2v/llm2vec_Supervised_contrastive_training/run_test/run_supervised_single_language.py"

# 4. 基础 JSON 配置文件路径
CONFIG_FILE="/share/home/chenyuxuan/Research_CodeSearch/DecoderLLMs-CodeSearch/Fine-tuning/Fine-tuning_method/finetuning_script/CodeGemma/single_supervised_csn_CodeGemma.json"

# 5. 所有实验结果存放的基础目录 (脚本会自动在此目录下创建子目录)
BASE_OUTPUT_DIR="/share/home/chenyuxuan/Research_CodeSearch/llm2v/llm2vec_Supervised_contrastive_training/model_output"

# 6. torchrun 使用的端口号
MASTER_PORT=29545

# ======================== 脚本执行区域 ========================

# 确保基础输出目录存在
mkdir -p "$BASE_OUTPUT_DIR"
echo "All experiment outputs will be saved under: $BASE_OUTPUT_DIR"
echo

# 外层循环: 遍历每一种语言
for LANG in "${LANGUAGES[@]}"
do
    # 内层循环: 遍历每一种比例
    for RATIO in "${RATIOS[@]}"
    do
        # --- 1. 准备本次实验的参数 ---
        # 为本次实验创建一个唯一的、有描述性的输出目录
        # 示例: .../output/Qwen2.5-supervised-python-ratio-0.5
        EXPERIMENT_NAME="Codegemma-7B-Instruct-it-supervised-single-${LANG}-ratio-${RATIO}"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"

        echo "======================================================================="
        echo "Starting Experiment: Lang=${LANG}, Ratio=${RATIO}"
        echo "  - Output Directory:    $OUTPUT_DIR"
        echo "======================================================================="

        # --- 2. 执行训练 ---
        # 每次都从 JSON 文件中定义的基础模型开始训练
        torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} \
            "$TRAIN_SCRIPT" \
            "$CONFIG_FILE" \
            --use_language "$LANG" \
            --use_ratio "$RATIO" \
            --output_dir "$OUTPUT_DIR" \
            --overwrite_output_dir

        # --- 3. 检查实验是否成功 ---
        if [ $? -ne 0 ]; then
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "!!           EXPERIMENT FAILED!                                      !!"
            echo "!!   Language: $LANG, Ratio: $RATIO"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            # 你可以选择在这里退出脚本，或者注释掉 exit 1 让脚本继续尝试下一个组合
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
echo "********** All experiments completed successfully! **********"
echo "************************************************************************"