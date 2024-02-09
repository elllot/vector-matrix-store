from typing import List

import numpy as np
from tqdm import tqdm

from vector_matrix_store.schema import EmbeddingVector, EmbeddingMethodType

VECTOR_DIMENSION = 789


def generate_embedding_list(
    dimension: int = VECTOR_DIMENSION, vector_count: int = 50
) -> List[EmbeddingVector]:
    print(f"Generating {vector_count} embeddings of dimension {dimension}.")
    embeddings = np.random.randn(vector_count, dimension)
    return [
        EmbeddingVector(vector=v, embedding_method_type=EmbeddingMethodType.CUSTOM)
        for v in tqdm(embeddings)
    ]


def generate_embedding(dimension: int = VECTOR_DIMENSION) -> EmbeddingVector:
    print(f"Generating query embedding of dimension {dimension}.")
    return EmbeddingVector(
        vector=np.random.randn(dimension),
        embedding_method_type=EmbeddingMethodType.CUSTOM,
    )
