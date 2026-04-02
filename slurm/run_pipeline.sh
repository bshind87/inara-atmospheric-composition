#!/bin/bash

mkdir -p logs

echo "Submitting pipeline.. "

# Step 1: Feature Engineering
jid1=$(sbatch --parsable feature_job.sh)
echo "Feature job ID: $jid1"

# Step 2: Training (after feature)
jid2=$(sbatch --parsable --dependency=afterok:$jid1 train_job.sh)
echo "Train job ID: $jid2"

# Step 3: Prediction (after training)
jid3=$(sbatch --parsable --dependency=afterok:$jid2 predict_job.sh)
echo "Predict job ID: $jid3"

echo "Pipeline submitted successfully 🚀"