#!/bin/bash

# Install dependencies.
pip install -e .
pip install flash-attn --no-build-isolation
pip install -r requirements.txt

# Download the RL feedback model.
mkdir -p searchless_chess/checkpoints
cd searchless_chess/checkpoints
wget https://storage.googleapis.com/searchless_chess/checkpoints/270M.zip
unzip -q 270M.zip
rm 270M.zip
cd ../..
