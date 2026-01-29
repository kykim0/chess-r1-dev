#!/bin/bash
#SBATCH --job-name=cr1_next-state
#SBATCH --output=/home/kykim/slurm-logs/%x-%j-test.out
#SBATCH --error=/home/kykim/slurm-logs/%x-%j-test.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G
#SBATCH --time=72:00:00
#SBATCH --nodelist=
#SBATCH --exclude=node7,node9,node10

MAX_LEN=${1:-2048}

export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/tmp/ray_$$

# Environment variables.
export N_GPUS=export N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export ROLLOUT_TP_SIZE=1  # Set tensor parallel
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Define model and dataset
export DATA_DIR=${DATA_DIR:-"data/verl"}
export BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen3-1.7B"}

# Experiment metadata
export PROJECT_NAME=${PROJECT_NAME:-"chess-r1"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"next-state_l${MAX_LEN}"}

timestamp=$(date +"%m%d-%H%M")
DATA_NAME=$(basename "$DATA_DIR")       
BASE_MODEL_NAME=$(basename "$BASE_MODEL")
export EXPERIMENT_NAME="${timestamp}_${EXPERIMENT_NAME}_${DATA_NAME}_${BASE_MODEL_NAME}"
export LOG_DIR="outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}"
export CHECKPOINT_DIR="${LOG_DIR}/checkpoint"


hydra_args="
    hydra.run.dir=$LOG_DIR \
"

trainer_args=" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['wandb'] \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_training_steps=100 \
    trainer.resume_from_path=False \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.val_before_train=true \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.rollout_data_dir=$LOG_DIR/rollouts \
    trainer.log_val_generations=1024 \
"

# batch_size: data.train_batch_size * actor.num_response
# data.train_batch_size: number of prompts
# actor.mini_batch_size: batch_size for gradient update
# actor.micro_batch_size_per_gpu: batch size per gpu for gradient accum
# updates per rollout: actor.epochs * (batch_size / actor.mini_batch_size)
data_args=" \
    data.train_files=$DATA_DIR/train_next-state.parquet \
    data.val_files=[$DATA_DIR/valid_next-state.parquet] \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=${MAX_LEN} \
    data.dataloader_num_workers=0 \
"

actor_args=" \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
"

rollout_args=" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.n=8 \
"

reference_args=" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
"

algorithm_args=" \
    algorithm.gamma=1.0 \
    algorithm.filter_groups.enable=false \
    algorithm.filter_groups.max_num_gen_batches=10 \
"

reward_args=" \
    reward_model.reward_manager=chess_base \
    reward_model.evaluation.reward_manager=chess_lichess \
"

TRAIN_ARGS="$hydra_args $trainer_args $data_args $actor_args $rollout_args $reference_args $algorithm_args $reward_args"

##########################
# Train Model
ray stop --force && ray start --head --include-dashboard=True

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

python -m verl.trainer.main_ppo --config-name grpo_trainer $TRAIN_ARGS 2>&1 | tee ${LOG_DIR}/main_ppo.log
