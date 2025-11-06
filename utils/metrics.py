from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class Prediction:
    predictions: tuple
    label_ids: tuple


def get_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy": (tn + tp) / (tn + fp + fn + tp),
        "Precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "Precision_0": precision[0],
        "Precision_1": precision[1],
        "Recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "Recall_0": recall[0],
        "Recall_1": recall[1],
        "F1": (
            2
            * ((tp / (tp + fp)) * (tp / (tp + fn)))
            / ((tp / (tp + fp)) + (tp / (tp + fn)))
            if ((tp / (tp + fp)) + (tp / (tp + fn))) > 0
            else 0
        ),
        "F1_0": f1[0],
        "F1_1": f1[1],
        "FAR": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "MAR": fn / (fn + tp) if (fn + tp) > 0 else 0,
    }


def evaluate_k(pred_t, true_s):
    k_values = range(1, 51)
    best_k = 0
    best_f1 = 0
    results = {}

    for k in k_values:
        pred_labels = np.array([1 if np.sum(seq) >= k else 0 for seq in pred_t])
        results[k] = get_metrics(true_s, pred_labels)
        if results[k]["F1"] > best_f1:
            best_f1 = results[k]["F1"]
            best_k = k

    return best_k, results


def get_result_dict(labels, scores, threshold, prefix=""):

    predictions = (scores >= threshold).astype(np.int16)
    results = get_metrics(labels, predictions)
    results["threshold"] = threshold

    results["num_label_1"] = np.sum(labels)
    results["num_pred_1"] = np.sum(predictions)

    results["num_label_0"] = labels.shape[0] - results["num_label_1"]
    results["num_pred_0"] = predictions.shape[0] - results["num_pred_1"]

    if prefix:
        results = {f"{prefix}_{k}": v for k, v in results.items()}
    return results


def compute_metrics_for_dual(
    pred,
    token_threshold=0.5,
    sequence_threshold=0.5,
):
    token_scores, sequence_scores = pred.predictions
    token_labels, sequence_labels = pred.label_ids

    token_scores_ori = (
        torch.nn.functional.softmax(torch.from_numpy(token_scores).float(), dim=-1)[
            :, :, -1
        ]
        .detach()
        .numpy()
    )
    sequence_scores = (
        torch.nn.functional.softmax(torch.from_numpy(sequence_scores).float(), dim=-1)[
            :, -1
        ]
        .detach()
        .numpy()
    )
    token_labels = token_labels

    mask = token_labels != -100
    token_scores_ori[~mask] = 0
    token_scores = token_scores_ori[mask]
    token_labels = token_labels[mask]

    results_token = get_result_dict(
        token_labels, token_scores, token_threshold, prefix="token"
    )
    results_sequence = get_result_dict(
        sequence_labels, sequence_scores, sequence_threshold, prefix="sequence"
    )

    results = results_token | results_sequence

    pred_token = (token_scores_ori >= results_token["token_threshold"]).astype(np.int16)

    best_k, results_k = evaluate_k(pred_token, sequence_labels)
    results["best_k"] = best_k
    results["k_results"] = results_k

    return results


def compute_metrics_for_seq(
    pred: Prediction,
    sequence_threshold=0.5,
):
    sequence_scores = pred.predictions
    sequence_labels = pred.label_ids

    sequence_scores = (
        torch.nn.functional.softmax(torch.from_numpy(sequence_scores).float(), dim=-1)[
            :, -1
        ]
        .detach()
        .numpy()
    )
    results_sequence = get_result_dict(
        sequence_labels, sequence_scores, sequence_threshold, prefix="sequence"
    )
    return results_sequence


def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj
