# Chess-LLM

## Installation

```
conda create -n chess-r1 python=3.12
conda activate chess-r1

# RL
bash install.sh

# SkyThought
cd verl/third_party/SkyThought
pip install -e .
cd ../../../

# LLaMA-Factory
cd verl/third_party/SkyThought/skythought/train/LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ../../../../../../
```


## Chess task

### Download dataset
```
# Download entire dataset
bash run_dataset.sh
```

### RL Training

#### RL Dataset
```
# If you want to collect your own RL dataset by yourself
wget https://storage.googleapis.com/searchless_chess/data/puzzles.csv
wget https://database.lichess.org/lichess_db_puzzle.csv.zst
unzstd lichess_db_puzzle.csv.zst

python ./searchless_chess/data/extract_lichess.py --save_path ./raw_data --data_path ./lichess_db_puzzle.csv 

# Change template type according to your needs
python ./data_preprocess/lichess_quiz.py --template_type qwen_instruct_fen_legal_rule_table
```

#### Setup RL feedback model
```
cd searchless_chess
mkdir checkpoints
cd checkpoints
wget https://storage.googleapis.com/searchless_chess/checkpoints/270M.zip
unzip 270M.zip
rm 270M.zip
```

#### RL Fine-Tuning
```
# w/o RL feedback
bash ./scripts/grpo_train/Qwen25_7B_Base_reastemp_fen_legal_rule_norlfeedback.sh
bash ./scripts/grpo_train/Llama31_8B_Base_reastemp_fen_legal_rule_norlfeedback.sh
# w/ RL feedback
bash ./scripts/grpo_train/Qwen25_7B_Base_reastemp_fen_legal_rule_rlfeedback.sh
bash ./scripts/grpo_train/Llama31_8B_Base_reastemp_fen_legal_rule_rlfeedback.sh
# o3 pre-trained w/ RL feedback
bash ./scripts/grpo_train/Qwen25_7B_Base_o3_SFT_reastemp_fen_legal_rule_rlfeedback.sh
bash ./scripts/grpo_train/Llama31_8B_Base_o3_SFT_reastemp_fen_legal_rule_rlfeedback.sh
```

### o3 Reasoning SFT Training
```
# Use LLaMA-Factory
cd verl/third_party/SkyThought/skythought/train/LLaMA-Factory
GRADIO_SHARE=1 llamafactory-cli webui
# Select `preprocessed_o3_chess_reasoning_dataset.json` as training dataset
```

### Zero-shot chess evaluation
```
# Example scripts
bash ./scripts/zeroshot_eval/qwen7b_base/eval_all.sh
```



## Tensorboard Logging
```
tensorboard --logidr=outputs/{USER_NAME}/{GROUP_NAME}/{EXPERIMENT_NAME}
```


## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).
* [Chess Dataset](https://github.com/google-deepmind/searchless_chess)
* [Codebase](https://github.com/Jiayi-Pan/TinyZero)
