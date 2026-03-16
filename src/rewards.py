import re


def parse_timestamp_output(output_string):
    """Parses timestamp output, similar to the example code."""
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)

    if not answer_matches:
        return None  # No <answer> tags found.

    last_answer_content = answer_matches[-1]
    # print("last_answer_content:", last_answer_content)

    matches = re.findall(
        r"(\d+\.?\d*) (seconds to) (\d+\.?\d*) seconds", last_answer_content, re.IGNORECASE
    )
    if not matches:
        return None
    last_match = matches[-1]
    start_time = float(last_match[0])
    end_time = float(last_match[2])
    return start_time, end_time


def parse_single_timestamp_output(output_string):
    """Parses timestamp output, similar to the example code."""
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)

    if not answer_matches:
        return None  # No <answer> tags found.

    last_answer_content = answer_matches[-1]
    # print("last_answer_content:", last_answer_content)

    matches = re.findall(
        r"(\d+\.?\d*) seconds", last_answer_content, re.IGNORECASE
    )
    if not matches:
        return None
    last_match = matches[-1]
    return float(last_match)


def iou_timestamp_reward_v2(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    **kwargs,
):
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    assert data_source == 'vtg' or data_source == 'status'

    if data_source == 'status':
        parsed_times = parse_single_timestamp_output(solution_str)
        if parsed_times is None:
            return 0
        error = abs(parsed_times - ground_truth[0])
        if error <= 1:
            return 1
        elif error >= 6:
            return 0
        else:
            return 1 - (error - 1) / 5

    duration = extra_info.get("duration")
    reward = 0.0
    parsed_times = parse_timestamp_output(solution_str)
    start_time, end_time = 0, 0
    gt_start, gt_end = ground_truth
    s, e = gt_start, gt_end
    if parsed_times:
        start_time, end_time = parsed_times
        from_number = start_time
        to_number = end_time

        intersection = max(0, min(to_number, e) - max(from_number, s))
        union = max(to_number, e) - min(from_number, s)
        if union > 0:
            iou = intersection / union  # 0.1 0.3

        gt_start_norm = 1.0 * s / duration
        gt_end_norm = 1.0 * e / duration
        pred_start_norm = 1.0 * start_time / duration
        pred_end_norm = 1.0 * end_time / duration
        reward = (
            iou
            * (1 - abs(gt_start_norm - pred_start_norm))
            * (1 - abs(gt_end_norm - pred_end_norm))
        )

    return reward


def format_reward(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    **kwargs,
):
    """Reward function that checks if the completion has a specific format."""
    if data_source == 'status':
        pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    else:
        pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)

    matche = re.fullmatch(pattern, solution_str.strip())
    
    return 1.0 if matche else 0.0


def compute_vtg_score(**kwargs):
    format_score = format_reward(**kwargs)
    iou_score = iou_timestamp_reward_v2(**kwargs)

    return {
        "score": format_score + iou_score,
        "format_score": format_score,
        "iou_score": iou_score
    }
