#!/bin/bash
#SBATCH --job-name=inara_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

echo "Starting Training... "

module load python
source ~/envs/ml/bin/activate

cd ../src

python xgboost_train.py

echo "Training Completed"