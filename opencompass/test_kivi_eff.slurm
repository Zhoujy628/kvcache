#!/bin/bash
#SBATCH --job-name=eff_kivi
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=eff_logs/%x-%j.out     # slurm log 文件（基本别动）
#SBATCH --error=eff_logs/%x-%j.error
#SBATCH --qos=rebuttal
export PYTHONPATH=$HOME/OpenCompass:$PYTHONPATH
export PYTHONPATH=$HOME/OpenCompass/KIVI:$PYTHONPATH
export PYTHONPATH=$HOME/OpenCompass/KIVI/quant:$PYTHONPATH

# 运行并把标准输出和错误都存到 test_log.log
python test_kivi_efficiency.py 