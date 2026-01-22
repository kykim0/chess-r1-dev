from typing import Dict

from verl.utils.reward_score.chess_utils import extract_solution


def compute_score(
    solution_str: str,
    ground_truth_dict: Dict[str, str],
    answer_reward: float = 1.0,
) -> float:
    """Computes comprehensive score for model response."""
    # Parse ground truth data.
    answer = ground_truth_dict["next_move_san"]

    # Initialize log metrics.
    # Note that when evaluating on multiple datasets, the log dict keys need
    # to match across datasets to appease _validate() in ray_trainer.py.
    logs = {"answer_text": "", "format": 0, "accuracy": 0}

    # Extract model answer (pass logs to the function)
    answer_text, _, logs = extract_solution(solution_str, logs)
    if answer_text:
        logs["answer_text"] = answer_text

    reward = 0.0
    correct = 0.0
    try:
        if answer_text:
            correct = answer_text == answer
            if correct:
                logs["accuracy"] = 1
    except Exception as e:
        print("\n[Content Validation] Skipped due to format errors or missing answer:", e)                

    reward = correct
    return reward, correct, logs
