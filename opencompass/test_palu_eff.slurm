#!/bin/bash
#SBATCH --job-name=eff_palu
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=eff_logs/%x-%j.out     # slurm log 文件（基本别动）
#SBATCH --error=eff_logs/%x-%j.error

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/jyzhou/OpenCompass/vqkvcache/opencompass:$PYTHONPATH
export PYTHONPATH=/home/jyzhou/OpenCompass/Palu:$PYTHONPATH

# 运行并把标准输出和错误都存到 test_log.log
python test_palu_efficiency.py 