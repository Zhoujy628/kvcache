#!/bin/bash
#SBATCH --job-name=ppl_palu
#SBATCH --partition=ADA6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=ppl_logs/%x-%j.out     # slurm log 文件（基本别动）
#SBATCH --error=ppl_logs/%x-%j.error

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/jyzhou/OpenCompass/vqkvcache/opencompass:$PYTHONPATH
export PYTHONPATH=/home/jyzhou/OpenCompass/Palu:$PYTHONPATH

python test_ppl_palu.py --model /home/jyzhou/OpenCompass/Palu2/Llama-3.1-8B_ratio-0.7_gs-4-fisher_uniform-whiten \
    2>&1 | tee testu_log_palu.log