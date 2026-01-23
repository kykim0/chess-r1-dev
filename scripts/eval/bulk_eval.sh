#!/bin/bash
#SBATCH --job-name=bulk-eval
#SBATCH --output=/home/kykim/slurm-logs/%x-%j.out
#SBATCH --error=/home/kykim/slurm-logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G
#SBATCH --time=4:00:00

EXPERIMENT_DIR=${1:?"Error: Experiment dir is required"}
EVAL_DATA=${2:-""}
N_GPUS=${3:-""}


if [ -z "$EVAL_DATA" ]; then
    EVAL_DATA=(
        "data/verl/valid_legal-rules-detailed.parquet"
        "data/verl/valid_rules-detailed.parquet"
        "data/verl/valid_next-state.parquet"
    )
else
    EVAL_DATA=("$EVAL_DATA")
fi

if [ -z "$N_GPUS" ] || [ "$N_GPUS" -eq 0 ]; then
    if command -v nvidia-smi &> /dev/null; then
        N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        N_GPUS=0
    fi
fi

# Evaluate all checkpoints on all eval data.
for CKPT_DIR in $(ls -d $EXPERIMENT_DIR/checkpoint/global_step_*); do
    for EVAL_DATUM in "${EVAL_DATA[@]}"; do
        FILE_NAME=$(basename "$EVAL_DATUM" .parquet)
        if [ ! -f ${CKPT_DIR}/actor/eval_results/${FILE_NAME}_eval.json ]; then
            # bash scripts/eval/eval_checkpoint.sh $CKPT_DIR/actor/huggingface $EVAL_DATUM $N_GPUS
            sbatch -J eval scripts/eval/eval_checkpoint.sh $CKPT_DIR/actor/huggingface $EVAL_DATUM 0
        fi
    done
done
