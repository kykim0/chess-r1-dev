#!/bin/bash

for split in train valid test; do
    python searchless_chess/data/extract_lichess.py \
        --save_path ./data/lichess_db_puzzle_recent/puzzles_${split}_processed.csv \
        --data_path ./data/lichess_db_puzzle_recent/puzzles_${split}.csv
done
