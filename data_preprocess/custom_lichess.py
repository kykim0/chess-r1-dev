"""Preprocess dataset for chess task.

Example usage:
$ python custom_lichess.py \
    --save_dir ./data/verl \
    --save_filename train_legal-rules-detailed \
    --data_path ./data/lichess_db_puzzle_recent/puzzles_train_processed.csv \
    --template_type legal/rules/detailed \
    --data_source lichess \
    --num_samples 100000
"""

import argparse
from collections import defaultdict
import os

import chess
from datasets import Dataset
import pandas as pd


def _optimal_move_conv(example, config):
    """Creates a conversation for the optimal move task."""

    def _system_message(
        include_legal_moves=True,
        include_rules=True,
        include_pgn=False,
        reasoning_detail="detailed",
    ):
        base_inst = (
            f"You are a helpful assistant who plays chess professionally. "
            f"First, think through the reasoning process internally and then provide the user with the best move. "
            f"The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively."
        )

        if reasoning_detail == "detailed":
            reasoning_inst = "\n".join([
                f"The reasoning process should describe how you analyze the position and decide on the best move, including:"
                f"- A strategic evaluation of the position."
                f"- A comparison of key candidate moves."
                f"- For each candidate, consider the opponent's likely response and outcome."
                f"- Conclude with a clear justification for the final choice."
            ])
        else:
            reasoning_inst = f"The reasoning process should describe how you analyze the position and decide on the best move."

        format_inst = "The answer must be in SAN notation, restricted to the moving piece and destination square (e.g., Nf3, Rxf2, c5)."

        context_info = "Now, the user provides the board in FEN format"
        if include_legal_moves:
            context_info += ", a list of legal moves for the given board"
        if include_pgn:
            context_info += ", the PGN representing the game history up to the current FEN position"
        context_info += "."

        final_inst = "After analyzing the position, clearly state the best move in SAN notation within <answer> </answer> tags. i.e., <answer> Nf3 </answer>."

        rules_reminder = ""
        if include_rules:
            rules_reminder = "\n".join([
                f"Reminder of chess rules:"
                f"- Bishops move diagonally."
                f"- Rooks move horizontally or vertically."
                f"- Knights jump in an L-shape."
                f"- Queens combine rook and bishop movements."
                f"- Kings move one square in any direction."
                f"- Pawns move forward, capture diagonally, and can promote."
            ])

        system_message = "\n".join([base_inst, reasoning_inst, format_inst, context_info, final_inst, rules_reminder])
        return system_message.strip()

    def _user_message(example, include_legal_moves=True, include_pgn=False):
        board_fen = example["board_fen"]
        legal_moves_san = example["legal_moves_san"]
        prev_moves_san_numbering = example["prev_moves_san_numbering"]

        user_content = f"Current board in FEN: {board_fen}."
        if include_legal_moves and legal_moves_san:
            user_content += f"\nLegal moves: {legal_moves_san}."
        if include_pgn and prev_moves_san_numbering:
            user_content += f"\nPGN leading to the FEN: {prev_moves_san_numbering}."
        return user_content

    # Extract configs.
    include_legal_moves = config.get("include_legal_moves", True)
    include_rules = config.get("include_rules", True)
    include_pgn = config.get("include_pgn", False)
    reasoning_detail = config.get("reasoning_detail", "detailed")

    # Build the conversation.
    system_message = _system_message(
        include_legal_moves=include_legal_moves,
        include_rules=include_rules,
        include_pgn=include_pgn,
        reasoning_detail=reasoning_detail,
    )
    user_message = _user_message(
        example,
        include_legal_moves=include_legal_moves,
        include_pgn=include_pgn,
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return messages


def _next_state_conv(example, config):
    """Creates a conversation for the next state prediction task."""

    def _system_message(include_legal_moves):
        base_inst = (
            f"You are a helpful assistant with a strong understanding of chess. "
            f"Given a board state in FEN and a move in SAN, output the resulting board state in FEN after applying the move. "
            f"The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively."
        )
        reasoning_inst = f"The reasoning process should describe the analysis used to derive the next board state in FEN from the current position and move."
        format_inst = "The answer must be in FEN within <answer> </answer> tags, e.g., <answer> 6rk/ppp2R1p/7Q/4P3/8/2q4P/P5r1/5R1K b - - 1 27 </answer>."
        context_info = "Now, the user provides the board in FEN format, the move in SAN"
        if include_legal_moves:
            context_info += ", and a list of possible next board states in FEN derived from legal moves"
        context_info += "."
        system_message = "\n".join([base_inst, reasoning_inst, format_inst])
        return system_message.strip()

    def _user_message(example, include_legal_moves):
        board_fen = example["board_fen"]
        next_move_san = example["next_move_san"]
        user_content = f"Current board in FEN: {board_fen}."
        user_content += f"\nNext move in SAN: {next_move_san}."
        if include_legal_moves:
            next_state_fens = []
            for move_san in example["legal_moves_san"].split():
                board = chess.Board(example["board_fen"])
                board.push_san(move_san)
                next_state_fens.append(board.fen())
            fens_str = ", ".join(next_state_fens)
            user_content += f"\nPossible next board states in FEN: {fens_str}."
        return user_content

    # Extract configs.
    include_legal_moves = config.get("include_legal_moves", True)

    system_message = _system_message(
        include_legal_moves=include_legal_moves,
    )
    user_message = _user_message(
        example,
        include_legal_moves=include_legal_moves,
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return messages


def _legal_moves_conv(example):
    """Creates a conversation for the legal move prediction task."""

    def _system_message():
        base_inst = (
            f"You are a helpful assistant with a strong understanding of how chess is played. "
            f"Given a board state in FEN, list all moves in SAN that are legal from the current position. "
            f"The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively."
        )
        reasoning_inst = f"The reasoning process should describe the analysis of the current position used to determine all legal moves."
        format_inst = "Provide space-separated SAN moves within <answer> </answer> tags, e.g., <answer> Nf3 Rxf2 c5 </answer>."
        system_message = "\n".join([base_inst, reasoning_inst, format_inst])
        return system_message.strip()

    def _user_message():
        board_fen = example["board_fen"]
        user_content = f"Current board in FEN: {board_fen}."
        return user_content

    system_message = _system_message()
    user_message = _user_message()
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return messages


def _state_f2a_conv(example):
    """Creates a conversation for the FEN-to-ASCII translation task."""

    def _system_message():
        base_inst = (
            f"You are a helpful assistant with a strong understanding of different chess board representations. "
            f"Given a board state in FEN, output the same board state in ASCII format, with each rank separated by a newline. "
            f"The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively."
        )
        reasoning_inst = f"The reasoning process should explain how the current position in FEN maps to its ASCII representation."
        format_inst = (
            f"Provide the board state in ASCII within <answer> </answer> tags,"
            f" e.g., <answer> . . . . . . r k \n p p p . . R . p \n . . . . . . . Q \n . . . . P . . . \n . . . . . . . . \n . . q . . . . P \n P . . . . . r . \n . . . . . R K . </answer>."
        )
        system_message = "\n".join([base_inst, reasoning_inst, format_inst])
        return system_message.strip()

    def _user_message():
        board_fen = example["board_fen"]
        user_content = f"Current board in FEN: {board_fen}."
        return user_content

    system_message = _system_message()
    user_message = _user_message()
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return messages


def _state_a2f_conv(example):
    """Creates a conversation for the ASCII-to-FEN translation task."""

    def _system_message():
        base_inst = (
            f"You are a helpful assistant with a strong understanding of different chess board representations. "
            f"Given a board state in ASCII format, output the same board state in FEN. "
            f"The reasoning process and the answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively."
        )
        ###
        reasoning_inst = f"The reasoning process should explain how the current position in ASCII maps to its FEN representation."
        format_inst = f"Provide the board state in FEN within <answer> </answer> tags, e.g., <answer> 6rk/ppp2R1p/7Q/4P3/8/6qP/P7/5R1K b - - 2 29 </answer>."
        system_message = "\n".join([base_inst, reasoning_inst, format_inst])
        return system_message.strip()

    def _user_message():
        board_str = example["board_str"]
        user_content = f"Current board in ASCII: {board_str}."
        return user_content

    system_message = _system_message()
    user_message = _user_message()
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return messages


def make_conversation(example, config, data_source):
    if data_source in ("lichess", "deepmind_lichess_accuracy"):
        messages = _optimal_move_conv(example, config)
    elif data_source == "next_state":
        messages = _next_state_conv(example, config)
    elif data_source == "legal_moves":
        messages = _legal_moves_conv(example)
    elif data_source == "state_f2a":
        messages = _state_f2a_conv(example)
    elif data_source == "state_a2f":
        messages = _state_a2f_conv(example)
    else:
        raise ValueError(f"Unsupported data source: {data_source}")
    return messages


def parse_template_config(template_type):
    """Parses the template type string into a configuration dictionary."""
    config = {
        "include_legal_moves": False,
        "include_rules": False,
        "include_pgn": False,
        "reasoning_detail": "standard",
    }
    if "legal" in template_type:
        config["include_legal_moves"] = True
    if "rule" in template_type:
        config["include_rules"] = True
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
    parser.add_argument("--template_type", type=str, default="legal-rule",
                        help="Template configuration: combinations of legal, rule, table, pgn, detailed (e.g., 'legal-rule' or 'legal-rule_table-pgn')")
    parser.add_argument("--data_source", type=str, default=None, required=True)
    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()

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

    data_source = args.data_source
    # TODO(kykim): A hack for backward compatibility.
    if data_source == "lichess" and "valid" in args.save_filename:
        data_source = "deepmind_lichess_accuracy"

    def make_map_fn(split):
        def process_fn(example, idx):
            messages = make_conversation(example, config, data_source)
            board = chess.Board(example["board_fen"])
            board.push_san(example["next_move_san"])
            next_board_fen = board.fen()
            data = {
                "data_source": data_source,  # lichess, deepmind_lichess_accuracy
                "prompt": messages,
                "ability": "chess",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "id": example["id"],
                        "rating": example["rating"],
                        "board_fen": example["board_fen"],
                        "next_board_fen": next_board_fen,
                        "board_str": example["board_str"],
                        "next_move_san": example["next_move_san"],
                        "legal_moves_san": example["legal_moves_san"],
                    }
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "template_config": config,
                }
            }
            return data
        return process_fn

    print("Processing dataset...")
    raw_dataset = raw_dataset.map(function=make_map_fn(args.save_filename), with_indices=True)

    # Create local directory if needed.
    save_dir = args.save_dir
    os.makedirs(os.path.expanduser(save_dir), exist_ok=True)

    filename = f"{args.save_filename}.parquet"
    raw_dataset.to_parquet(os.path.join(save_dir, filename))
    print(f"Saved to: {save_dir}/{filename}")
