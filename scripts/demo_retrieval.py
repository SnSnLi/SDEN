# scripts/demo_retrieval.py

import os
import argparse
from tqdm import tqdm
from collections import defaultdict
from encoder.clip import CLIPWrapperModel as Model
import torch.nn.functional as F
from data.dataset import QADataset



def cosine_sim(a, b):
    """计算两个向量的余弦相似度
    参数:
        a: 第一个嵌入向量 (形状 [1, 768])
        b: 第二个嵌入向量 (形状 [1, 768])
    返回:
        相似度分数 (-1到1之间):
          1 表示完全相同
          0 表示无关
         -1 表示完全相反
    """
    # 确保输入形状匹配 [1,768] vs [1,768]
    a = a.unsqueeze(0) if a.dim() == 1 else a
    b = b.unsqueeze(0) if b.dim() == 1 else b
    
    # 计算并返回相似度
    sim = F.cosine_similarity(a, b).mean().item()
    print(f"计算相似度: 向量a均值={a.mean().item():.4f}, 向量b均值={b.mean().item():.4f} -> 相似度={sim:.4f}")
    return sim


def evaluate_flickr30k(
    dataset_root,
    image_root,
    split="test",
    top_k_list=[1, 5, 20]  # 添加R@20
):
    # 加载数据集
    dataset = QADataset(name="flickr30k_entities", root=dataset_root)
    subset = dataset.get_subset(split)
    model = Model()
    results = []

    for idx in tqdm(subset.split_indices[split], desc=f"Evaluating {split}"):
        query, query_id, answer_ids, _ = dataset[idx.item()]

        # 编码 query 和第一个图像
        first_img_id = answer_ids[0] if answer_ids else dataset.indices[0]
        image_path = os.path.join(image_root, f"{first_img_id}.jpg")
        if not os.path.exists(image_path):
            continue
        embeddings = model.encode_text_and_image(query, image_path)
        query_vec = embeddings['text_embeds']
        print(f"Query embedding shape: {query_vec.shape}, mean: {query_vec.mean().item():.4f}")

        # 遍历所有 candidate 图像
        scored = []
        for img_id in dataset.indices:
            image_path = os.path.join(image_root, f"{img_id}.jpg")
            if not os.path.exists(image_path):
                continue
            embeddings = model.encode_text_and_image("", image_path)
            image_vec = embeddings['image_embeds']
            print(f"Image {img_id} embedding shape: {image_vec.shape}, mean: {image_vec.mean().item():.4f}")
            sim = cosine_sim(query_vec, image_vec)
            print(f"Similarity with image {img_id}: {sim:.4f}")
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
