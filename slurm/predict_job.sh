#!/bin/bash
#SBATCH --job-name=inara_predict
#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

echo "Starting Prediction...  "

module load python
source ~/envs/ml/bin/activate

cd ../src

python predict.py ../inara_data/

echo "Prediction + Report Completed"