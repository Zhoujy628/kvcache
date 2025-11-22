#!/bin/bash
#SBATCH --job-name=kivi_niah_3.1    # 任务名
#SBATCH --partition=ADA6000     # 分区（根据你们集群改）
#SBATCH --gres=gpu:1                # 申请 1 张 GPU
#SBATCH --cpus-per-task=8            # CPU 核心数
#SBATCH --mem=128G                    # 内存
#SBATCH --time=48:00:00              # 最长运行时间
#SBATCH --output=logs/%x-%j.out      # 输出日志
#SBATCH --error=logs/%x-%j.err       # 错误日志

export PYTHONPATH=$HOME/OpenCompass:$PYTHONPATH
export PYTHONPATH=$HOME/OpenCompass/KIVI:$PYTHONPATH
export PYTHONPATH=$HOME/OpenCompass/KIVI/quant:$PYTHONPATH
# 运行 OpenCompass + KIVI baseline
srun python run.py eval/eval_on_niah_kivi_llama3.1.py --dump-eval-details -r
