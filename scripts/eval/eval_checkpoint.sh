#!/bin/bash
#SBATCH --job-name=cr1_eval
#SBATCH --output=/home/kykim/slurm-logs/%x-%j.out
#SBATCH --error=/home/kykim/slurm-logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G
#SBATCH --time=24:00:00

# Usage examples:
# $ bash scripts/eval/eval_checkpoint.sh Qwen/Qwen3-0.6B data/verl/valid_next-state.parquet 2 4
# $ bash scripts/eval/eval_checkpoint.sh \
#       outputs/chess-r1/.../checkpoint/global_step_90/actor/huggingface \
#       data/verl/valid_next-state.parquet 2 4

# Parse arguments.
CHECKPOINT_PATH=${1:?"Error: Checkpoint path is required"}
EVAL_DATA=${2:?"Error: Eval data is required"}
N_GPUS=${3:-0}
NUM_SAMPLES=${4:--1}

# Validate eval data.
if [ ! -f "$EVAL_DATA" ]; then
    echo "Error: Evaluation data file does not exist: $EVAL_DATA"
    exit 1
fi

if [ -z "$N_GPUS" ] || [ "$N_GPUS" -eq 0 ]; then
    if command -v nvidia-smi &> /dev/null; then
        N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        echo "Warning: nvidia-smi not found. Defaulting N_GPUS to 0."
        N_GPUS=0
    fi
fi

export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/tmp/ray_$$

# Environment variables.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export ROLLOUT_TP_SIZE=1

# Determine output directory.
SCRIPT_PATH=$(readlink -f "$0")
if [ -z "$OUTPUT_DIR" ]; then
    # Hack to check if it's a HF model.
    SLASH_COUNT=$(echo "$CHECKPOINT_PATH" | tr -cd '/' | wc -c)
    if [ "$SLASH_COUNT" -eq 1 ]; then
        REPO_ROOT=$(dirname "$(dirname "$(dirname "$SCRIPT_PATH")")")
        MODEL_NAME=$(basename "$CHECKPOINT_PATH")
        OUTPUT_DIR="$REPO_ROOT/eval_results/$MODEL_NAME"
    else
        OUTPUT_DIR=$(dirname "$CHECKPOINT_PATH")
        OUTPUT_DIR="$OUTPUT_DIR/eval_results"
    fi
fi

echo "=============================================="
echo "Offline Evaluation Configuration"
echo "=============================================="
echo "Checkpoint:    $CHECKPOINT_PATH"
echo "Eval Data:     $EVAL_DATA"
echo "Num GPUs:      $N_GPUS"
echo "Output Dir:    $OUTPUT_DIR"
echo "=============================================="

# Hydra arguments.
hydra_args=" \
    hydra.run.dir=$OUTPUT_DIR \
"

# Model configuration.
model_args=" \
    actor_rollout_ref.model.path=$CHECKPOINT_PATH \
"

# Data configuration.
data_args=" \
    data.eval_files=$EVAL_DATA \
    data.eval_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.max_samples=${NUM_SAMPLES} \
    data.dataloader_num_workers=0 \
"

# Rollout configuration.
rollout_args=" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
"

# Trainer configuration.
trainer_args=" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
"

# Reward manager configuration.
reward_args=" \
    reward_model.evaluation.reward_manager=chess_lichess \
"

# Evaluation configuration.
eval_args=" \
    eval.n_samples=1 \
    eval.save_generations=true \
    eval.verbose=true \
    eval.output_dir=$OUTPUT_DIR \
"

EVAL_ARGS="$hydra_args $model_args $data_args $rollout_args $trainer_args $reward_args $eval_args"

# Start Ray (if not already running).
ray stop --force 2>/dev/null
ray start --head --include-dashboard=True

# Create output directory.
mkdir -p "$OUTPUT_DIR"

# Run evaluation.
echo ""
echo "Starting evaluation..."
python -m verl.trainer.main_offline_eval --config-name offline_eval $EVAL_ARGS 2>&1 | tee "${OUTPUT_DIR}/main_offline_eval.log"

echo ""
echo "Evaluation complete. Results saved to: $OUTPUT_DIR"
