#!/usr/bin/env bash
#SBATCH -A naiss2023-22-751 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1  # Launching 1 node with 1 Nvidia A100
#SBATCH -t 0-05:00:00 # Set a time-out of 5 hour

# Here you should typically call your GPU-hungry application
module load SciPy-bundle/2022.05-foss-2022a matplotlib/3.5.2-foss-2022a
module load PyTorch-bundle/1.13.1-foss-2022a-CUDA-11.7.0

source ./microct_env/bin/activate
python3 instant-NGP-cunerf/model/main.py --rhino --fp16 --workspace "optuna-studies-2" --tune