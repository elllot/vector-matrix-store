from typing import Any, List, Optional, Tuple, Union

import abc
from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from vector_matrix_store.schema import ScoredNeighbor


class VectorMatrix(abc.ABC):
    @abc.abstractproperty
    def synced(self) -> bool: ...

    @abc.abstractproperty
    def matrix(self) -> Union[NDArray, jax.Array]: ...

    @abc.abstractmethod
    def sync(self): ...

    @abc.abstractmethod
    def add(self, vector: NDArray): ...

    @abc.abstractmethod
    def delete_at(self, idx: int): ...

    # TODO: fix return type to include JAX.
    @abc.abstractmethod
    def search_nearest_neighbors(
        self,
        vector: NDArray,
        k: int,
    ) -> List[ScoredNeighbor]: ...

    @abc.abstractmethod
    def get_entry_count(self) -> int: ...

    @abc.abstractmethod
    def get_vector_dimension(self) -> int: ...


class NumPyVectorMatrix(VectorMatrix):
    """Vector Matrix for computing similarities using NumPy.

    To optimize for faster mutation (add/delete) operations, vectors are maintained in a
    list and the matrix is lazily loaded on a search operation. Once loaded the matrix
    stays cached until the next mutation.
    """

    def __init__(self, matrix: Optional[NDArray] = None):
        """Initialize the NumPyVectorMatrix."""
        self._vectors: List[NDArray] = []
        self._matrix: NDArray = np.array([])
        self._synced = False
        if matrix is not None:
            self._vectors = list(matrix)
            self._matrix = matrix
            # Store is expected to be synced on init when a matrix is provided.
            self._synced = True

    @property
    def synced(self):
        return self._synced

    @property
    def matrix(self) -> NDArray:
        return self._matrix

    def reindex(self, matrix_or_vectors: Union[NDArray, List[NDArray]]):
        # Provided type is a matrix.
        if isinstance(matrix_or_vectors, np.ndarray):
            self._vectors = list(matrix_or_vectors)
            self._matrix = matrix_or_vectors
            self._synced = True
            return

        self._vectors = matrix_or_vectors[:]
        self._synced = False

    def add(self, vector: NDArray):
        """Add a target vector to the list maintaining all vectors."""
        if vector.shape[0] != self.get_vector_dimension():
            raise ValueError(
                "Received add request for vector of unexpected dimension. Expected: "
                f"{self.get_vector_dimension()}, Received: {vector.shape[0]}"
            )
        self._vectors.append(vector)
        self._synced = False

    def delete_at(self, idx: int):
        """Delete vector at index a specified index.

        Removes from the vector from the underlying list rather than mutating the
        matrix.

        Args:
            idx (int): Index of vector to delete.
        """
        # Swap target index with last element.
        self._vectors[idx], self._vectors[-1] = self._vectors[-1], self._vectors[idx]
        # Remove target element (at end after swap).
        self._vectors.pop()
        self._synced = False

    def delete_at_np(self, idx: int):
        """Delete vector at index a specified index.

        This approach is much faster than `np.delete`. Benchmark on 100K vectors yeild
        236ms for `np.delete` vs 250us (microseconds) for this approach.

        Args:
            idx (int): Index of vector to delete.
        """
        # Rewrite target delete index with last vector in the matrix.
        self._matrix[idx] = self._matrix[self.get_entry_count() - 1]
        # Remove last row.
        self._matrix = self._matrix[:-1, :]

    def sync(self):
        """Syncs the search matrix with the list of vectors."""
        if not self._synced:
            self._matrix = np.array(self._vectors)
            self._synced = True

    def search_nearest_neighbors(
        self, vector: NDArray, k: int = 10
    ) -> List[ScoredNeighbor]:
        # Sync compute matrix for computation if not currently loaded / cached.
        self.sync()

        similarities, sorted_indices = self._compute_similarity_scores(vector)
        # Make sure k is less than total entry count.
        k = min(k, self.get_entry_count())
        return [
            ScoredNeighbor(index=idx, score=float(similarities[idx]))
            for idx in sorted_indices[:k]
        ]

    def _compute_similarity_scores(self, vector: NDArray) -> Tuple[NDArray, NDArray]:
        m = self._matrix / np.sqrt((self._matrix**2).sum(1, keepdims=True))
        v = vector / np.sqrt((vector**2).sum())
        similarities = m.dot(v)
        sorted_indices = np.argsort(-similarities)
        return similarities, sorted_indices

    def get_entry_count(self) -> int:
        return self._matrix.shape[0]

    def get_vector_dimension(self) -> int:
        return self._matrix.shape[1]


@jit
def rewrite_idx_with_last_row(matrix: jax.Array, idx: int) -> jax.Array:
    """A jit-compiled function for rewriting an index with the last row in a matrix.

    The jit-compiled variant is much faster than implementation without.

    Args:
        matrix: The matrix to modify.
        idx: The index to rewrite with the last row in the matrix.

    Returns:
        A copy of the modified matrix with the last row written to the target index.
    """
    return matrix.at[idx].set(matrix[-1])


@jit
def array_eq(a: jax.Array, b: jax.Array) -> bool:
    """A jit-compiled function for comparing equality of two JAX arrays."""
    return all(jnp.array_equal(a, b))


@jit
def _compute_similarity_scores(
    vector: jax.Array, matrix: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Compute nearest neighbors using JAX.

    Args:
        vector: Target vector to compare against the matrix.
        matrix: The target vector space to compute similarities against.

    Returns:
        A JAX array containing similarity scores for each vector in the matrix
        compared to the target vector.
    """
    # TODO: evaluate performance of normalizing during compute vs. "re-normalizing"
    # on each add / delete.
    m = matrix / jnp.sqrt((matrix**2).sum(1, keepdims=True))
    v = vector / jnp.sqrt((vector**2).sum())
    similarities = m.dot(v)
    sorted_indices = jnp.argsort(-similarities)
    return similarities, sorted_indices


@jit
def _append_to_matrix(matrix: jax.Array, vector: NDArray) -> jax.Array:
    """Appends a given vector to the JAX matrix.

    `jax.numpy.append` creates a copy, which is returned as the appended result. Jitted
    function runs much faster comparatively (vs non-jitted).
    """
    return jnp.append(matrix, jnp.array([vector]), axis=0)


# TODO: introduce optimization type to either optimize for search vs mutation (addition
# / deletion)


class JaxVectorMatrix(VectorMatrix):
    """Vector Matrix for computing similarities using JAX (jax.numpy).

    JAX directly uses matrix rather than lazily loading the matrix on a search operation
    due to conversion (from list of vectors) being the bottleneck for JAX matrices.
    """

    def __init__(self, matrix: Optional[jax.Array] = None):
        matrix = jnp.array([]) if matrix is None else matrix
        self._matrix: jax.Array = jax.device_put(matrix)
        self._synced = matrix is not None

    @property
    def synced(self):
        return self._synced

    @property
    def matrix(self) -> jax.Array:
        return self._matrix

    def rebuild_matrix(self, vector_matrix: NDArray):
        self._matrix = jnp.array(vector_matrix)

    def add(self, vector: NDArray):
        self._matrix = _append_to_matrix(self._matrix, vector)

    def delete_at(self, idx: int):
        self._matrix = rewrite_idx_with_last_row(self._matrix, idx)
        self._matrix = self._matrix[:-1, :]

    def sync(self):
        raise NotImplementedError("Sync operation not supported for JAX matrix.")

    def search_nearest_neighbors(
        self, vector: NDArray, k: int = 10
    ) -> List[ScoredNeighbor]:
        k = min(k, self.get_entry_count())
        scores, indices = _compute_similarity_scores(jnp.array(vector), self._matrix)
        # return [(i, similarities[i]) for i in sorted_ix[:k]]
        return [
            ScoredNeighbor(index=int(idx), score=float(scores[int(idx)]))
            for idx in indices[:k]
        ]

    def get_entry_count(self) -> int:
        return self._matrix.shape[0]

    def get_vector_dimension(self) -> int:
        return self._matrix.shape[1]
