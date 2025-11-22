#!/bin/bash
#SBATCH --job-name=palu_ruler_llama3.2    # 任务名
#SBATCH --partition=L40S      # 分区（根据你们集群改）
#SBATCH --gres=gpu:1                 # 申请 1 张 GPU
#SBATCH --cpus-per-task=8            # CPU 核心数
#SBATCH --mem=128G                    # 内存
#SBATCH --time=48:00:00              # 最长运行时间
#SBATCH --output=logs/%x-%j.out      # 输出日志
#SBATCH --error=logs/%x-%j.err       # 错误日志
#SBATCH --qos=iclr2026
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/jyzhou/OpenCompass/vqkvcache/opencompass:$PYTHONPATH
export PYTHONPATH=/home/jyzhou/OpenCompass/Palu:$PYTHONPATH
srun python run.py eval/eval_on_ruler_palu_llama3.2.py --dump-eval-details -r