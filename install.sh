#!/bin/bash

# conda create -n chess_llm python=3.10 sqlite==3.42.0
# source /home/kykim/miniconda3/bin/activate cr1


# install vllm
# pip3 install vllm==0.11.0
pip3 install ray[default]

# ignore the warning for the conflict with torch2.4
# pip install nvidia-cublas-cu12==12.4.5.8

# verl
# pip install -e .

# flash attention 2
# pip3 install flash-attn --no-build-isolation

# spacy_fastlang
pip install spacy_fastlang
pip install spacy-langdetect
# pip install cupy-cuda12x
# pip install cupy
# pip install cupy-cuda13x
# pip install cupy cupy-cuda13x
python -m spacy download en_core_web_lg

# install tensorboard (allow host view in kubeflow)
pip install tensorboard
# sed -i "s/\"--bind_all\", default=True,/\"--bind_all\",/g" /home/jovyan/conda/chess_llm/lib/python3.10/site-packages/tensorboard/plugins/core/core_plugin.py

# skythought
cd verl/third_party/SkyThought
pip install -e .
cd ../../../

# LLamaFactory
cd verl/third_party/SkyThought/skythought/train/LLaMA-Factory
pip install -e ".[torch,metrics]"
# pip install deepspeed==0.15.4
pip install deepspeed
cd ../../../../../../

# JAX
# pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# utilities
pip install -r requirements.txt
# pip install apache_beam
pip install apache-beam[tfrecord]

pip install jax==0.5.2
pip install chess
pip install jaxtyping==0.2.38
pip install google-cloud-storage
pip install chex==0.1.89
pip install grain
pip install dm-haiku
pip install optax==0.2.4
pip install orbax-checkpoint==0.11.8

pip install gdown

pip install ipdb