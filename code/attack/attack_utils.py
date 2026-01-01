import torch
import torch.nn.functional as F
from typing import List


def cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    dot_product = torch.dot(tensor1.flatten(), tensor2.flatten())
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    similarity = dot_product / (norm1 * norm2)
    return similarity.item()

