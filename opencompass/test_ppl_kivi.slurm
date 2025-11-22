#!/bin/bash
#SBATCH --job-name=ppl_kivi
#SBATCH --partition=ADA6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=ppl_logs/%x-%j.out     # slurm log 文件（基本别动）
#SBATCH --error=ppl_logs/%x-%j.error
#SBATCH --qos=iclr2026
export PYTHONPATH=$HOME/OpenCompass:$PYTHONPATH
export PYTHONPATH=$HOME/OpenCompass/KIVI:$PYTHONPATH
export PYTHONPATH=$HOME/OpenCompass/KIVI/quant:$PYTHONPATH

# 运行并把标准输出和错误都存到 test_log.log
python test_ppl_kivi.py --model /data1/public/hf/meta-llama/Llama-3.1-8B \
    2>&1 | tee test_log_kivi.log