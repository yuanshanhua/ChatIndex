from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch


def show_embedding_info(embeddings: dict[str, "torch.Tensor"], print_fn=print):
    for key, embedding in embeddings.items():
        print_fn(f"嵌入 {key} 的向量形状: {embedding.shape}")
        print_fn(f"嵌入 {key} 的 sum: {embedding.sum()}")
        print_fn(f"嵌入 {key} 的向量前10个值: {embedding[:10]}")
    # 计算两两之间的相似度
    from torch.nn.functional import cosine_similarity

    for key1, emb1 in embeddings.items():
        for key2, emb2 in embeddings.items():
            if key1 != key2:
                sim = cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                print_fn(f"相似度 {key1} & {key2}: {sim:.4f}")
