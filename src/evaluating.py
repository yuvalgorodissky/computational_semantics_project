
from tqdm import tqdm
from collections import Counter
import string
import re
from tqdm.auto import tqdm

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punct(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punct(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common_tokens.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_metrics(predictions, references):
    assert len(predictions) == len(references)
    f1_with_ans = exact_match_with_ans = total_with_ans = 0
    f1_no_ans = exact_match_no_ans = total_no_ans = 0

    for prediction, ground_truth in tqdm(zip(predictions, references), total=len(predictions), desc="Evaluating"):
        if ground_truth == "unanswerable":
            ground_truth = ""
        if ground_truth.strip():  # This question has an answer
            total_with_ans += 1
            exact_match_with_ans += exact_match_score(prediction, ground_truth)
            f1_with_ans += f1_score(prediction, ground_truth)
        else:  # This question does not have an answer
            total_no_ans += 1
            # For no-answer, exact match occurs if the prediction is also an empty or whitespace-only string
            exact_match_no_ans += int(prediction.strip() == ground_truth.strip())
            # F1 can be considered as 1 if both are empty (perfect match), otherwise 0 (no overlap)
            f1_no_ans += int(prediction.strip() == ground_truth.strip())

    # Compute overall metrics
    overall_exact_match = ((exact_match_with_ans + exact_match_no_ans) / (total_with_ans + total_no_ans)) * 100
    overall_f1 = ((f1_with_ans + f1_no_ans) / (total_with_ans + total_no_ans)) * 100

    # Compute separate metrics
    exact_match_with_ans_rate = 100.0 * exact_match_with_ans / total_with_ans if total_with_ans else 0
    f1_with_ans_rate = 100.0 * f1_with_ans / total_with_ans if total_with_ans else 0

    exact_match_no_ans_rate = 100.0 * exact_match_no_ans / total_no_ans if total_no_ans else 0
    f1_no_ans_rate = 100.0 * f1_no_ans / total_no_ans if total_no_ans else 0

    return {
        "average_exact_match": overall_exact_match,
        "average_f1": overall_f1,
        "exact_match_has_ans": exact_match_with_ans_rate,
        "f1_has_ans": f1_with_ans_rate,
        "exact_match_no_ans": exact_match_no_ans_rate,
        "f1_no_ans": f1_no_ans_rate}
