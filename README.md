See [OLD_README.md](OLD_README.md)

## Instalation
```
conda create -n zero python=3.9
pip install -e .
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# flash attention 2
pip3 install flash-attn --no-build-isolation

# quality of life
pip install wandb IPython matplotlib
```

## Generate Data
```
conda activate zero
python verl/examples/data_preprocess/countdown.py
```

## Run Training
```
conda activate zero
```
```
export CUDA_VISIBLE_DEVICES=7
export N_GPUS=1
export BASE_MODEL=Qwen/Qwen2.5-0.5B
export DATA_DIR=$HOME/data/countdown
PYTHONUNBUFFERE=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$DATA_DIR/train.parquet \
 data.val_files=$DATA_DIR/test.parquet \
 data.train_batch_size=256 \
 data.val_batch_size=1312 \
 data.max_prompt_length=256 \
 data.max_response_length=1024 \
 actor_rollout_ref.model.path=$BASE_MODEL \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=$BASE_MODEL \
 critic.ppo_micro_batch_size=8 \
 algorithm.kl_ctrl.kl_coef=0.01 \
 trainer.logger=['wandb'] \
 +trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=$N_GPUS \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.project_name=zero \
 trainer.experiment_name=countdown \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log
```
