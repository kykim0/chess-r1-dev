import re
from typing import Tuple, Optional


def extract_solution(solution_str: str, logs: dict) -> Tuple[Optional[str], str, dict]:
    """Extracts the final answer from the model"s response string by isolating the assistant's output.

    Args:
        solution_str: Raw response string from the language model
        logs: Dictionary to store logging metrics.

    Returns:
        Tuple containing (extracted_answer, processed_string, logs)
    """
    header_markers = [
        "Assistant:",
        "<|start_header_id|>assistant<|end_header_id|><|begin_of_text|>",
        "<|im_start|>assistant"
    ]
    processed_str = None
    for marker in header_markers:
        if marker in solution_str:
            processed_str = solution_str.split(marker, 1)[1]
            break
    if processed_str is None:
        print("[Error] Failed to locate model response header")
        return None, solution_str, logs

    # Extract final answer using XML-style tags.
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        return None, processed_str, logs
    else:
        logs["format"] = 1
        final_answer = matches[-1].group(1).strip()
        return final_answer, processed_str, logs


def validate_response_structure(processed_str: str, logs: dict) -> Tuple[bool, dict]:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model
        logs: Dictionary to store logging metrics.

    Returns:
        Tuple (validation_passed, logs)
    """
    validation_passed = True

    # Check required tags.
    tags = {
        "think_start": ("<think>", 1),
        "think_end": ("</think>", 1),
        "answer_start": ("<answer>", 1),
        "answer_end": ("</answer>", 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = processed_str.find(tag_str)
        if count != expected_count:
            validation_passed = False
            logs["format"] = 0
            return validation_passed, logs

    # Verify tag order: <think> ... </think> ... <answer> ... </answer>
    if (positions["think_start"] > positions["think_end"] or
        positions["think_end"] > positions["answer_start"] or
        positions["answer_start"] > positions["answer_end"]):
        validation_passed = False
        logs["format"] = 0
    else:
        logs["format"] = 1

    return validation_passed, logs
