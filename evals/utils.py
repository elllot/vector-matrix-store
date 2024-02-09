from typing import List

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import numpy as np
from numpy.typing import NDArray
import random as rrandom
from tqdm import tqdm
from vector_matrix_store.schema import EmbeddingMethodType, EmbeddingVector

VECTOR_SPACE_SIZE = 100000
VECTOR_DIMENSION = 768


def generate_jax_vector(seed_key: int = 0) -> jax.Array:
    key = random.PRNGKey(seed_key)
    return random.normal(key, (VECTOR_DIMENSION,))


def generate_jax_vector_static(i: int) -> jax.Array:
    return jnp.array([i for _ in range(VECTOR_DIMENSION)])


def generate_jax_vecotr_list(vector_space_size: int = VECTOR_SPACE_SIZE):
    return [generate_jax_vector(i) for i in tqdm(range(vector_space_size))]


def generate_jax_matrix(vector_space_size: int = VECTOR_SPACE_SIZE) -> jax.Array:
    return jnp.array(generate_jax_vecotr_list(vector_space_size))


def generate_np_vector() -> NDArray:
    return np.array(
        [rrandom.uniform(-1, 1) for _ in range(VECTOR_DIMENSION)], dtype=np.float32
    )


def generate_vector_list(vector_space_size: int = VECTOR_SPACE_SIZE) -> List[NDArray]:
    return [generate_np_vector() for _ in tqdm(range(vector_space_size))]


def generate_np_matrix(vector_space_size: int = VECTOR_SPACE_SIZE) -> NDArray:
    return np.array(generate_vector_list(vector_space_size))


def vectors_to_embeddings(vector_list: List[NDArray]) -> List[EmbeddingVector]:
    return [
        EmbeddingVector(vector=v, embedding_method_type=EmbeddingMethodType.CUSTOM)
        for v in vector_list
    ]


def generate_embeddings(
    vector_space_size: int = VECTOR_SPACE_SIZE,
) -> List[EmbeddingVector]:
    return vectors_to_embeddings(generate_vector_list(vector_space_size))
