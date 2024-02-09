from typing import List

import numpy as np
from tqdm import tqdm

from vector_matrix_store.file_system_context import FileSystemContext
from vector_matrix_store.schema import EmbeddingVector, EmbeddingMethodType
from vector_matrix_store.vector_store import (
    JaxVectorStore,
    VectorStoreConfig,
    VectorStoreType,
)

from recipes.utils import generate_embedding, generate_embedding_list

# ----------------------------------------------------------------------------------
# JaxVectorStore examples. ---------------------------------------------------------
# ----------------------------------------------------------------------------------


def jax_from_memory():
    """Example of initializing a JaxVectorStore from vectors loaded in memory."""
    embeddings = generate_embedding_list(vector_count=5)
    store_cfg = VectorStoreConfig(
        vector_dimension=embeddings[0].get_dimension(),
        embedding_method_type=EmbeddingMethodType.CUSTOM,
        vector_store_type=VectorStoreType.JAX,
    )
    store = JaxVectorStore.from_embeddings(embeddings, store_cfg)
    query_embedding = generate_embedding()
    top_embedding_ids = store.search(query_embedding.vector)
    print(top_embedding_ids)
    print(len(top_embedding_ids))

    # Add a new embedding vector to the store (e.g. from a passage).
    new_embedding = generate_embedding()
    store.add_embedding(new_embedding)
    print(f"matrix shape: {store.vector_matrix.matrix.shape}")
    top_embedding_ids = store.search(query_embedding.vector)
    print(top_embedding_ids)
    print(len(top_embedding_ids))

    # Deleting the newly added embedding.
    store.delete_embedding(new_embedding.embedding_id)
    top_embedding_ids = store.search(query_embedding.vector)
    print(top_embedding_ids)
    print(len(top_embedding_ids))


def jax_from_file_system():
    """Example of loading a JaxVectorStore from the file system."""
    fs_ctx = FileSystemContext()
    # Loading target config from file system.
    store_cfg = fs_ctx.load_vector_store_config("vector_store_config_jax.json")
    # Loading store with target config.
    loaded_store = fs_ctx.load_store(store_cfg)
    top_embedding_ids = loaded_store.search(generate_embedding().vector)
    print(top_embedding_ids[0])


def jax_from_memory_to_file_system():
    embeddings = generate_embedding_list()

    store_cfg = VectorStoreConfig(
        vector_dimension=embeddings[0].get_dimension(),
        embedding_method_type=EmbeddingMethodType.CUSTOM,
        vector_store_type=VectorStoreType.JAX,
        vector_index_relative_path="vector_index_jax.json",
        vector_matrix_relative_path="vector_matrix_jax.npy",
        vector_store_config_relative_path="vector_store_config_jax.json",
    )
    store = JaxVectorStore.from_embeddings(embeddings, store_cfg)
    query_embedding = generate_embedding()
    top_embedding_ids = store.search(query_embedding.vector)
    print(top_embedding_ids[0])

    fs_ctx = FileSystemContext()
    fs_ctx.write_store(store)

    loaded_store = fs_ctx.load_store(store_cfg)
    loaded_top_embedding_ids = loaded_store.search(query_embedding.vector)

    assert top_embedding_ids == loaded_top_embedding_ids


if __name__ == "__main__":
    jax_from_memory()
