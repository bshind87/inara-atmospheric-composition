#!/bin/bash
#SBATCH --job-name=inara_baseline
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shinde.b/inara/logs/03_train_baseline_%j.out
#SBATCH --error=/scratch/shinde.b/inara/logs/03_train_baseline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shinde.b@northeastern.edu
# #SBATCH --account=<your_account>

# ── Environment ──────────────────────────────────────────────────────────────
CODE_DIR=/home/shinde.b/inara_vscode
CONDA_ENV=inara_env

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

mkdir -p /scratch/shinde.b/inara/logs

echo "========================================"
echo "  Job: inara_baseline   ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME   Date: $(date)"
echo "========================================"

cd ${CODE_DIR}

# --save persists the trained RF model to scratch
python pipeline/steps/03_train_baseline.py \
    --config pipeline/config.yaml \
    --profile hpc \
    --save

EXIT_CODE=$?
echo "Finished with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
