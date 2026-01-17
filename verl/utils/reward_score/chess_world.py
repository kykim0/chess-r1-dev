from typing import Dict

import chess

from verl.utils.reward_score.chess_utils import extract_solution


def f1_score(ground_truth_list, predicted_list):
    """Computes the F1 score."""
    gt_set = set(ground_truth_list)
    pred_set = set(predicted_list)

    intersection = len(gt_set.intersection(pred_set))
    len_gt = len(gt_set)
    len_pred = len(pred_set)

    if len_gt + len_pred == 0:
        return 0.0 # Avoid division by zero
    return (2 * intersection) / (len_gt + len_pred)


def compute_score(
    solution_str: str,
    ground_truth_dict: Dict[str, str],
    data_source: str,
) -> float:
    """Computes comprehensive score for model response."""
    # Parse ground truth data.
    if "next_state" in data_source:
        board = chess.Board(ground_truth_dict["board_fen"])
        board.push_san(ground_truth_dict["next_move_san"])
        answer = board.fen()  # Next state in FEN.
    elif "legal_moves" in data_source:
        answer = ground_truth_dict["legal_moves_san"].split()
    elif "state_f2a" in data_source:
        answer = ground_truth_dict["board_str"]
    elif "state_a2f" in data_source:
        answer = ground_truth_dict["board_fen"]
    else:
        raise ValueError(f"Unsupported chess data source: {data_source}")

    # Initialize log metrics.
    logs = {"format": 0, "accuracy": 0}

    # Extract model answer (pass logs to the function).
    answer_text, _, logs = extract_solution(solution_str, logs)

    reward = 0.0
    correct = 0.0
    try:
        if answer_text:
            if isinstance(answer, list):
                pred_list = answer_text.split()
                reward = f1_score(answer, pred_list)
                correct = reward == 1.0
            else:
                correct = answer_text == answer
                reward = correct
            if correct:
                logs["accuracy"] = 1
    except Exception as e:
        print("\n[Content Validation] Skipped due to format errors or missing answer:", e)                

    # Include the data source in the keys.
    logs = {f"{key}_{data_source}": value for key, value in logs.items()}
    return reward, correct, logs
