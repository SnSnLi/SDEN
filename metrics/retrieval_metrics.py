# metrics/retrieval_metrics.py

import numpy as np
from typing import List, Dict

def recall_at_k(pred_ids: List[str], gt_ids: List[str], k: int) -> float:
    return float(any(gt in pred_ids[:k] for gt in gt_ids))

def hit_at_k(pred_ids: List[str], gt_ids: List[str], k: int) -> float:
    hits = sum(1 for gt in gt_ids if gt in pred_ids[:k])
    return hits / len(gt_ids) if gt_ids else 0.0

def mrr(pred_ids: List[str], gt_ids: List[str]) -> float:
    for i, pid in enumerate(pred_ids):
        if pid in gt_ids:
            return 1.0 / (i + 1)
    return 0.0

def average_precision(pred_ids: List[str], gt_ids: List[str]) -> float:
    hits, score = 0, 0.0
    for i, pid in enumerate(pred_ids):
        if pid in gt_ids:
            hits += 1
            score += hits / (i + 1)
    return score / len(gt_ids) if gt_ids else 0.0

def r_precision(pred_ids: List[str], gt_ids: List[str]) -> float:
    r = len(gt_ids)
    retrieved = pred_ids[:r]
    hits = sum(1 for pid in retrieved if pid in gt_ids)
    return hits / r if r > 0 else 0.0

def ndcg(pred_ids: List[str], gt_ids: List[str], k: int) -> float:
    dcg = 0.0
    for i, pid in enumerate(pred_ids[:k]):
        if pid in gt_ids:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0

def compute_all_metrics(pred_ids: List[str], gt_ids: List[str], k_values=[1, 5, 10]) -> Dict[str, float]:
    metrics = {
        "mrr": mrr(pred_ids, gt_ids),
        "map": average_precision(pred_ids, gt_ids),
        "r_precision": r_precision(pred_ids, gt_ids),
    }
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(pred_ids, gt_ids, k)
        metrics[f"hit@{k}"] = hit_at_k(pred_ids, gt_ids, k)
        metrics[f"ndcg@{k}"] = ndcg(pred_ids, gt_ids, k)
    return metrics
