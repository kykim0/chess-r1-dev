"""Preprocess dataset for chess task.

Example usage:
$ python custom_lichess.py \
    --save_dir ./data \
    --save_filename train \
    --data_path ./data/ours/puzzles_train_processed.csv \
    --model_name Qwen/Qwen3-0.6B \
    --template_type legal/rules/detailed \
    --num_samples 100000
"""

import argparse
from collections import defaultdict
import os

from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer


def get_value(record, key):
    return record.get(key, "")


def make_system_message(
    include_legal_moves=True,
    include_rules=True,
    include_table=False,
    include_pgn=False,
    reasoning_detail="standard",
):
    """Creates a dynamic system message based on the requested features."""

    base_instructions = """You are a helpful assistant who plays chess professionally.
First think through the reasoning process internally and then provides the user with the best move.
The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively."""
    
    if reasoning_detail == "detailed":
        reasoning_instructions = """The reasoning process should describe how you analyze the position and decide on the best move, including:
  - A strategic evaluation of the position.
  - A comparison of key candidate moves.
  - For each candidate, consider the opponent's likely response and outcome.
  - Conclude with a clear justification for the final choice."""
    else:
        reasoning_instructions = "The reasoning process should describe how you analyze the position and decide on the best move."
    
    answer_format = """The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5)."""
    
    context_info = "Now, the user provides a FEN string"
    if include_legal_moves:
        context_info += ", and a list of legal moves for the given board"
    if include_pgn:
        context_info += ", the PGN representing the game history up to the current FEN position"
    if include_table:
        context_info += ", and a square by square chess piece position table"
    context_info += "."
    
    final_instruction = """After analyzing the position, clearly state the best move in SAN notation within <answer> </answer> tags. i.e., <answer> Nf3 </answer>"""
    
    rules_reminder = ""
    if include_rules:
        rules_reminder = """
Reminder of chess rules:
- Bishops move diagonally.
- Rooks move horizontally or vertically.
- Knights jump in an L-shape.
- Queens combine rook and bishop movement.
- Kings move one square in any direction.
- Pawns move forward, capture diagonally, and can promote."""
    
    system_message = "\n".join([
        base_instructions,
        reasoning_instructions,
        answer_format,
        context_info,
        final_instruction,
        rules_reminder,
    ])
    return system_message.strip()


def make_user_message(chess_record, include_legal_moves=True, include_table=False, include_pgn=False):
    """Creates a dynamic user message based on the requested features."""
    board_fen = get_value(chess_record, "board_fen")
    legal_moves_san = get_value(chess_record, "legal_moves_san")
    piece_table = get_value(chess_record, "piece_table")
    prev_moves_san_numbering = get_value(chess_record, "prev_moves_san_numbering")

    user_content = f"Current FEN string: {board_fen}"
    if include_legal_moves and legal_moves_san:
        user_content += f"\nLegal moves: {legal_moves_san}"
    if include_pgn and prev_moves_san_numbering:
        user_content += f"\nPGN leading to FEN: {prev_moves_san_numbering}"
    if include_table and piece_table:
        user_content += f"\nChess piece position table: {piece_table}"
    return user_content


def make_prompt_with_tokenizer(tokenizer, chess_record, config):
    """Creates a prompt using the tokenizer's chat template.
    
    Args:
        tokenizer: The model's tokenizer with apply_chat_template
        chess_record: Dictionary containing chess game data
        config: Dictionary with configuration for what to include
    """
    # Extract configs.
    include_legal_moves = config.get("include_legal_moves", True)
    include_rules = config.get("include_rules", True)
    include_table = config.get("include_table", False)
    include_pgn = config.get("include_pgn", False)
    reasoning_detail = config.get("reasoning_detail", "standard")

    # Build the conversation.
    messages = []

    # System message.
    system_message = make_system_message(
        include_legal_moves=include_legal_moves,
        include_rules=include_rules,
        include_table=include_table,
        include_pgn=include_pgn,
        reasoning_detail=reasoning_detail
    )
    messages.append({"role": "system", "content": system_message})

    # User message.
    user_message = make_user_message(
        chess_record,
        include_legal_moves=include_legal_moves,
        include_table=include_table,
        include_pgn=include_pgn
    )
    messages.append({"role": "user", "content": user_message})

    return messages


def parse_template_config(template_type):
    """Parses the template type string into a configuration dictionary."""
    config = {
        "include_legal_moves": False,
        "include_rules": False,
        "include_table": False,
        "include_pgn": False,
        "reasoning_detail": "standard"
    }
    if "legal" in template_type:
        config["include_legal_moves"] = True
    if "rule" in template_type:
        config["include_rules"] = True
    if "table" in template_type:
        config["include_table"] = True
    if "pgn" in template_type:
        config["include_pgn"] = True
    if "reastemp" in template_type or "detailed" in template_type:
        config["reasoning_detail"] = "detailed"
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--save_filename", type=str, default=None, required=True)
    parser.add_argument("--data_path", type=str, default=None, required=True)
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Model name to load tokenizer from (e.g., Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--template_type", type=str, default="legal-rule",
                        help="Template configuration: combinations of legal, rule, table, pgn, detailed (e.g., 'legal-rule' or 'legal-rule_table-pgn')")
    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    # Parse template configuration.
    config = parse_template_config(args.template_type)
    print(f"Template configuration: {config}")

    # We use only odd number data (our turn data).
    def gen_from_csv(path):
        df = pd.read_csv(path)
        id_counts = defaultdict(int)
        for row in df.to_dict(orient="records"):
            cid = row["id"]
            # id_counts[cid] is the zero-based index of this row for that id
            if id_counts[cid] % 2 == 0:
                yield row
            id_counts[cid] += 1

    raw_dataset = Dataset.from_generator(gen_from_csv, gen_kwargs={"path": args.data_path})
    print(f"Total dataset size: {len(raw_dataset)}")

    if args.num_samples:
        assert len(raw_dataset) >= args.num_samples, \
            f"Dataset has {len(raw_dataset)} examples but need {args.num_samples}"
        raw_dataset = raw_dataset.select(range(args.num_samples))

    def make_map_fn(split):
        def process_fn(example, idx):
            # Generate prompt using tokenizer's chat template.
            messages = make_prompt_with_tokenizer(tokenizer, example, config)
            data = {
                "data_source": "lichess",  # lichess, deepmind_lichess_accuracy
                "prompt": messages,
                "ability": "chess",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "id": example["id"],
                        "answer": example["next_move_san"],
                        "rating": example["rating"],
                        "board_fen": example["board_fen"],
                        "next_move_san": example["next_move_san"],
                        "legal_moves_san": example["legal_moves_san"],
                    }
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "model_name": args.model_name,
                    "template_config": config,
                }
            }
            return data
        return process_fn

    print("Processing dataset...")
    raw_dataset = raw_dataset.map(function=make_map_fn(args.save_filename), with_indices=True)

    # Create local directory if needed.
    model_id = args.model_name.split("/")[-1].split("-")[0].lower()
    save_dir = os.path.join(args.save_dir, f"{model_id}")
    os.makedirs(os.path.expanduser(save_dir), exist_ok=True)

    filename = f"{args.save_filename}.parquet"
    raw_dataset.to_parquet(os.path.join(save_dir, filename))
    print(f"Saved to: {save_dir}/{filename}")
