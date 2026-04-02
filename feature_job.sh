#!/bin/bash
#SBATCH --job-name=inara_feature
#SBATCH --output=logs/feature_%j.out
#SBATCH --error=logs/feature_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

echo "Starting Feature Engineering... "

module load python
source ~/envs/ml/bin/activate

cd ../src

python xgboost_feature_engineering.py

echo "Feature Engineering Completed"