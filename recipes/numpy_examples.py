"""Recipes for using the NumPyVectorStore."""

from typing import List

import numpy as np
from tqdm import tqdm

from vector_matrix_store.file_system_context import FileSystemContext
from vector_matrix_store.schema import EmbeddingMethodType
from vector_matrix_store.vector_store import (
    NumPyVectorStore,
    VectorStoreConfig,
    VectorStoreType,
)

from recipes.utils import generate_embedding, generate_embedding_list


def numpy_from_memory():
    """Example of initializing a NumPyVectorStore from vectors loaded in memory."""
    embeddings = generate_embedding_list()
    # Loading with default generated config.
    store = NumPyVectorStore.from_embeddings(embeddings)
    query_embedding = generate_embedding()
    top_embedding_ids = store.search(query_embedding.vector)
    print(top_embedding_ids[0])


def numpy_from_file_system():
    """Example of loading a NumPyVectorStore from the file system."""
    # Load the store.
    fs_ctx = FileSystemContext()
    loaded_store = fs_ctx.load_store()

    # Generate a test query embedding vector
    query_embedding = generate_embedding()

    # Perform search.
    top_embedding_ids = loaded_store.search(query_embedding.vector)
    print(top_embedding_ids[0])


def numpy_from_memory_to_file_system():
    embeddings = generate_embedding_list()
    store_cfg = VectorStoreConfig(
        vector_dimension=embeddings[0].get_dimension(),
        embedding_method_type=EmbeddingMethodType.CUSTOM,
        vector_store_type=VectorStoreType.NUMPY,
    )
    store = NumPyVectorStore.from_embeddings(embeddings, store_cfg)
    query_embedding = generate_embedding()
    top_embedding_ids = store.search(query_embedding.vector)
    print(top_embedding_ids[0])

    fs_ctx = FileSystemContext()
    fs_ctx.write_store(store)

    loaded_store = fs_ctx.load_store(store_cfg)
    loaded_top_embedding_ids = loaded_store.search(query_embedding.vector)

    assert top_embedding_ids == loaded_top_embedding_ids


if __name__ == "__main__":
    numpy_from_memory()
