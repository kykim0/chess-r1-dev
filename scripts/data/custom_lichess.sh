#!/bin/bash

# Optimal move prediction w/ legal moves provided.
for split in train valid test; do
    python data_preprocess/custom_lichess.py \
        --save_dir ./data/verl \
        --save_filename ${split}_legal-rules-detailed \
        --data_path ./data/lichess_db_puzzle_recent/puzzles_${split}_processed.csv \
        --template_type legal/rules/detailed \
        --data_source lichess
done

# Optimal move prediction w/o legal moves provided.
for split in train valid test; do
    python data_preprocess/custom_lichess.py \
        --save_dir ./data/verl \
        --save_filename ${split}_rules-detailed \
        --data_path ./data/lichess_db_puzzle_recent/puzzles_${split}_processed.csv \
        --template_type rules/detailed \
        --data_source lichess
done

# Next state prediction.
for split in train valid test; do
    python data_preprocess/custom_lichess.py \
        --save_dir ./data/verl \
        --save_filename ${split}_next-state \
        --data_path ./data/lichess_db_puzzle_recent/puzzles_${split}_processed.csv \
        --data_source next_state
done

# Legal move prediction.
for split in train valid test; do
    python data_preprocess/custom_lichess.py \
        --save_dir ./data/verl \
        --save_filename ${split}_legal-moves \
        --data_path ./data/lichess_db_puzzle_recent/puzzles_${split}_processed.csv \
        --data_source legal_moves
done

# FEN-to-ASCII translation.
for split in train valid test; do
    python data_preprocess/custom_lichess.py \
        --save_dir ./data/verl \
        --save_filename ${split}_state-f2a \
        --data_path ./data/lichess_db_puzzle_recent/puzzles_${split}_processed.csv \
        --data_source state_f2a
done

# ASCII-to-FEN translation.
for split in train valid test; do
    python data_preprocess/custom_lichess.py \
        --save_dir ./data/verl \
        --save_filename ${split}_state-a2f \
        --data_path ./data/lichess_db_puzzle_recent/puzzles_${split}_processed.csv \
        --data_source state_a2f
done
