# scripts/demo_retrieval.py

import os
import argparse
from tqdm import tqdm
from collections import defaultdict
from sden.model import SDENWrapperModel
import torch.nn.functional as F
from dataset import QADataset  # Avatar 原始的数据类


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def evaluate_flickr30k(
    dataset_root,
    image_root,
    split="test",
    top_k_list=[1, 5, 10]
):
    # 加载数据集
    dataset = QADataset(name="flickr30k_entities", root=dataset_root)
    subset = dataset.get_subset(split)
    model = SDENWrapperModel()
    model.eval()

    results = []

    for idx in tqdm(subset.split_indices[split], desc=f"Evaluating {split}"):
        query, query_id, answer_ids, _ = dataset[idx.item()]

        # 编码 query
        query_vec = model.encode_text(query)

        # 遍历所有 candidate 图像
        scored = []
        for img_id in dataset.indices:
            image_path = os.path.join(image_root, f"{img_id}.jpg")
            if not os.path.exists(image_path):
                continue
            image_vec = model.encoder.encode_image(image_path)
            sim = cosine_sim(query_vec, image_vec.mean(dim=0))
            scored.append((img_id, sim))

        # 排序
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        pred_ids = [x[0] for x in ranked]

        results.append({
            "query": query,
            "gt_ids": answer_ids,
            "pred_ids": pred_ids
        })

    # 评估
    return compute_recall(results, top_k_list)


def compute_recall(results, top_k_list):
    total = len(results)
    recall_scores = defaultdict(int)

    for r in results:
        pred = r["pred_ids"]
        gt_ids = set(r["gt_ids"])
        for k in top_k_list:
            if any(gt in pred[:k] for gt in gt_ids):
                recall_scores[f"recall@{k}"] += 1

    metrics = {k: v / total for k, v in recall_scores.items()}
    print("\n 检索评估结果：")
    for k in top_k_list:
        print(f"Recall@{k}: {metrics.get(f'recall@{k}', 0):.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to flickr30k_entities dataset root")
    parser.add_argument("--image_root", type=str, required=True, help="Path to flickr30k image folder")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    evaluate_flickr30k(
        dataset_root=args.dataset_root,
        image_root=args.image_root,
        split=args.split
    )

