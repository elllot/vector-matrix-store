from typing import List, Optional

import abc
from dataclasses import dataclass, field
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from uuid import uuid4

from vector_matrix_store.constants import DEFAULT_VECTOR_STORE_CONFIG_PATH
from vector_matrix_store.matrix import NumPyVectorMatrix, JaxVectorMatrix, VectorMatrix
from vector_matrix_store.metrics_util import timer
from vector_matrix_store.schema import (
    EmbeddingMethodType,
    EmbeddingVector,
    ScoredNeighbor,
    Serializable,
    VectorStoreType,
)
from vector_matrix_store.vector_index import VectorIndex


@dataclass
class VectorStoreConfig(Serializable):
    """Configuration for a vector store instance."""

    # The dimension of the vectors in the store. This is expected to be uniform.
    vector_dimension: int
    # The method used to generate the embeddings. This is expected to be uniform.
    embedding_method_type: EmbeddingMethodType
    # The type of the underlying vector store.
    vector_store_type: VectorStoreType
    # The relative path to where this store configuration is stored.
    vector_store_config_relative_path: str = DEFAULT_VECTOR_STORE_CONFIG_PATH
    # Relative file path for where the vector matrix is stored. This path is used for
    # reading and writing the embeddings vector matrix.
    vector_matrix_relative_path: str = "vector_matrix.npy"
    # Relative file path for where the vector index is stored. This path is used for
    # reading and writing the index mapping for the embeddings in the store.
    vector_index_relative_path: str = "vector_index.json"
    store_instance_id: str = field(default_factory=lambda: uuid4().hex)


def _validate_embeddings(
    embeddings: List[EmbeddingVector],
    store_config: VectorStoreConfig,
):
    """Validates that all embeddings have the same dimension and method type.

    A vector store instance should not have multiple embedding types as that would
    result in invalid similarity computations.
    """
    for e in embeddings:
        if e.get_dimension() != store_config.vector_dimension:
            raise ValueError(
                f"Embedding vector (ID: {e.embedding_id}) has dimension "
                f"{e.get_dimension()}, which does not match the configured dimension "
                f"in the VectorStoreConfig: {store_config.vector_dimension}."
            )
        if e.embedding_method_type != store_config.embedding_method_type:
            raise ValueError(
                f"Embedding vector (ID: {e.embedding_id}) has method type "
                f"{e.embedding_method_type} does not match the configured method type "
                f"in the VectorStoreConfig: {store_config.embedding_method_type}."
            )


def _build_default_config(
    embeddings: List[EmbeddingVector], vector_store_type: VectorStoreType
) -> VectorStoreConfig:
    """Builds a default vector store configuration based on the provided embeddings."""
    return VectorStoreConfig(
        # All embeddings for a given store should have the same embedding method
        # type.
        embedding_method_type=embeddings[0].embedding_method_type,
        # All embeddings for a given store should have the same vector dimension.
        vector_dimension=embeddings[0].get_dimension(),
        vector_store_type=vector_store_type,
    )


class InMemoryVectorStore(abc.ABC):
    @abc.abstractproperty
    def store_config(self) -> VectorStoreConfig: ...

    @abc.abstractproperty
    def vector_matrix(self) -> VectorMatrix: ...

    @abc.abstractproperty
    def vector_index(self) -> VectorIndex: ...

    @abc.abstractmethod
    def add_embedding(self, embedding: EmbeddingVector): ...

    @abc.abstractmethod
    def delete_embedding(self, eid: str): ...

    @classmethod
    @abc.abstractmethod
    def from_embeddings(cls, embeddings: List[EmbeddingVector]): ...

    @abc.abstractmethod
    def search(self, target_vector: NDArray, k: int = 10): ...


class VectorStore(InMemoryVectorStore):
    def __init__(
        self,
        matrix: VectorMatrix,
        vector_index: VectorIndex,
        vector_store_config: VectorStoreConfig,
    ):
        self._matrix = matrix
        # TODO: explore how to embed id into matrix directly to avoid having to keep
        # this mapping. There could be an optimal approach in manipulating the numpy /
        # jax.numpy matrix to include the ids directly in the matrix without taking much
        # of a performance hit.
        self._index = vector_index
        self._store_config = vector_store_config

    @property
    def store_config(self) -> VectorStoreConfig:
        return self._store_config

    @property
    def vector_matrix(self) -> VectorMatrix:
        return self._matrix

    @property
    def vector_index(self) -> VectorIndex:
        return self._index

    def add_embedding(self, embedding: EmbeddingVector):
        """Adds an embedding entry to the store.

        Leverages a swap mechanism where...

        Args:
            eid: ID of the embedding entry.
        """
        if embedding.embedding_method_type != self._store_config.embedding_method_type:
            raise ValueError(
                f"Embedding method type {embedding.embedding_method_type} does not match "
                f"store's method type {self._store_config.embedding_method_type}."
            )

        self.validate_store()
        # Update index mappings.
        self._index.add(embedding.embedding_id)
        # Update matrix.
        self._matrix.add(embedding.vector)

    def delete_embedding(self, eid: str):
        """Deletes an embedding entry from the store.

        Validates the

        Args:
            eid: ID of the embedding entry.
        """
        self.validate_store()
        # Update index mappings. Returned value is the deleted index.
        idx = self._index.delete(eid)
        # Update matrix.
        # TODO: implement automic update in case matrix update errors.
        self._matrix.delete_at(idx)

    def search(self, target_vector: NDArray, k: int = 10) -> List[ScoredNeighbor]:
        """Searches for most similar vectors to a target vector.

        Args:
            target_vector: The vector to search for.
            k: The number of nearest neighbors to return.

        Returns:
            A list of scored neighbors, sorted descending by score.
        """
        self.validate_store()
        neighbors: List[ScoredNeighbor] = self._matrix.search_nearest_neighbors(
            target_vector, k
        )
        # Populate embedding IDs for neighbors.
        for neighbor in neighbors:
            neighbor.embedding_id = self._index.get_vid(neighbor.index)
        return neighbors

    def validate_store(self):
        matrix_entries = self._matrix.get_entry_count()
        mapping_entries = self._index.get_vector_count()
        if mapping_entries != matrix_entries:
            # TODO: consider adding check for making sure matrix dimensions line up with
            # expected
            raise ValueError(
                "Matrix entries does not match embedding mappings. Matrix has "
                f"{matrix_entries} entries while mapping has {mapping_entries} entries."
            )


class NumPyVectorStore(VectorStore):
    """An in-memory vector store implementation using NumPy.

    All operations are based on NumPy matrices. Optimized for mutations (insert & delete
    ). Searching is notably slower than the JAX implementation.
    """

    def __init__(
        self,
        vector_matrix: NumPyVectorMatrix,
        vector_index: VectorIndex,
        store_config: VectorStoreConfig,
    ):
        assert (
            store_config.vector_store_type == VectorStoreType.NUMPY
        ), f"Store type must be {VectorStoreType.NUMPY}"
        assert (
            vector_matrix.get_vector_dimension() == store_config.vector_dimension
        ), f"Invalid dimension: {vector_matrix.get_vector_dimension()}"

        super().__init__(vector_matrix, vector_index, store_config)

    @classmethod
    def from_embeddings(
        cls,
        embeddings: List[EmbeddingVector],
        store_config: Optional[VectorStoreConfig] = None,
    ):
        """Creates a NumPyVectorStore from a list of embeddings.

        Builds a vector index and a NumPyVectorMatrix from the provided embeddings.

        Args:
            embeddings: List of embeddings to load into the store.
            store_config: Optional store configuration. If not provided, a default
                configuration is built based on the embeddings.

        Returns:
            A NumPyVectorStore instance.
        """
        if not store_config:
            store_config = _build_default_config(embeddings, VectorStoreType.NUMPY)

        assert store_config.vector_store_type == VectorStoreType.NUMPY
        _validate_embeddings(embeddings, store_config)

        vectors = []
        idx_map = VectorIndex()
        for emb in embeddings:
            idx_map.add(emb.embedding_id)
            vectors.append(emb.vector)
        return cls(NumPyVectorMatrix(np.array(vectors)), idx_map, store_config)


class JaxVectorStore(VectorStore):
    """An in-memory vector store implementation using JAX.

    All operations are based on jax.numpy matrices. Optimized for searching. Similarity
    search using JAX numpy 2D matrix is much faster than pure Numpy implementations.

    While adding new vectors to the matrix is comparable to the Numpy implementation,
    deleting embeddings is notably slower.
    """

    def __init__(
        self,
        vector_matrix: JaxVectorMatrix,
        vector_index: VectorIndex,
        store_config: VectorStoreConfig,
    ):
        assert (
            store_config.vector_store_type == VectorStoreType.JAX
        ), f"Store type must be {VectorStoreType.JAX}"
        assert (
            vector_matrix.get_vector_dimension() == store_config.vector_dimension
        ), f"Invalid dimension: {vector_matrix.get_vector_dimension()}"

        super().__init__(vector_matrix, vector_index, store_config)

    @classmethod
    @timer
    def from_embeddings(
        cls,
        embeddings: List[EmbeddingVector],
        store_config: Optional[VectorStoreConfig] = None,
    ):
        """Creates a JaxVectorStore from a list of embeddings.

        Builds a vector index and a JaxVectorMatrix from the provided embeddings.

        Args:
            embeddings: List of embeddings to load into the store.
            store_config: Optional store configuration. If not provided, a default
                configuration is built based on the embeddings.

        Returns:
            A JaxVectorStore instance.
        """
        if not store_config:
            store_config = _build_default_config(embeddings, VectorStoreType.JAX)

        assert store_config.vector_store_type == VectorStoreType.JAX
        _validate_embeddings(embeddings, store_config)

        matrix_ = []
        idx_map = VectorIndex()
        for emb in embeddings:
            idx_map.add(emb.embedding_id)
            matrix_.append(jnp.array(emb.vector))
        return cls(JaxVectorMatrix(jnp.array(matrix_)), idx_map, store_config)
