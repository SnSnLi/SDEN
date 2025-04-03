# scripts/evaluate.py

import os
import json
from tqdm import tqdm
from sden.model import SDENWrapperModel
from dataset import QADataset
from scripts.eval_utils import (
    load_embedding_cache, save_embedding_cache,
    get_cache_path, get_result_path, get_json_result_path,
    filter_unfinished_queries
)
from metrics.retrieval_metrics import compute_all_metrics
import torch.nn.functional as F
import csv


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def evaluate_dataset(dataset_name, dataset_root, image_root, split="test"):
    model = SDENWrapperModel()
    model.eval()

    # Load dataset (QADataset, Avatar format)
    dataset = QADataset(name=dataset_name, root=dataset_root)
    subset = dataset.get_subset(split)
    all_query_ids = [int(i) for i in subset.split_indices[split]]

    # Try loading cached query/image embeddings
    query_cache_path = get_cache_path("query", dataset_name, split)
    image_cache_path = get_cache_path("images", dataset_name, split)

    query_embeddings = load_embedding_cache(query_cache_path)
    image_embeddings = load_embedding_cache(image_cache_path)

    # Determine which queries need to be processed (support resume)
    result_json_path = get_json_result_path(dataset_name, split)
    query_ids_to_run = filter_unfinished_queries(result_json_path, all_query_ids)

    results = []
    for idx in tqdm(query_ids_to_run, desc=f"Evaluating {split} set"):
        query, query_id, answer_ids, _ = dataset[idx]

        # Query embedding (cache or compute)
        if query_id not in query_embeddings:
            query_vec = model.encode_text(query).detach().cpu()
            query_embeddings[query_id] = query_vec
        else:
            query_vec = query_embeddings[query_id]

        # Score all candidate images
        scored = []
        for img_id in dataset.indices:
            if img_id not in image_embeddings:
                image_path = os.path.join(image_root, f"{img_id}.jpg")
                if not os.path.exists(image_path):
                    continue
                image_vec = model.encoder.encode_image(image_path).mean(dim=0).detach().cpu()
                image_embeddings[img_id] = image_vec
            else:
                image_vec = image_embeddings[img_id]

            score = cosine_sim(query_vec, image_vec)
            scored.append((img_id, score))

        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        pred_ids = [x[0] for x in ranked]

        metrics = compute_all_metrics(pred_ids, answer_ids)
        results.append({
            "query_id": query_id,
            "query": query,
            "gt_ids": answer_ids,
            "pred_ids": pred_ids,
            **metrics
        })

        # Optional: Save partial result every 10 queries (for resume)
        if len(results) % 10 == 0:
            with open(result_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    # Save full JSON and CSV
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    result_csv_path = get_result_path(dataset_name, split)
    with open(result_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Save updated embedding cache
    save_embedding_cache(query_cache_path, query_embeddings)
    save_embedding_cache(image_cache_path, image_embeddings)

    print("\n Done. Results saved to:")
    print(f"  → JSON: {result_json_path}")
    print(f"  → CSV : {result_csv_path}")


if __name__ == "__main__":
    evaluate_dataset(
        dataset_name="flickr30k_entities",
        dataset_root="/root/onethingai-tmp/avatar/data",
        image_root="/root/onethingai-tmp/avatar/data/flickr30k_entities/raw/flickr30k_images",
        split="test"
    )
