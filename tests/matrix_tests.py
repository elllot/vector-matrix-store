import unittest
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from vector_matrix_store.matrix import NumPyVectorMatrix, JaxVectorMatrix
from vector_matrix_store.schema import ScoredNeighbor

from tests.testing_utils import (
    generate_jax_matrix,
    generate_jax_vector,
    generate_matrix,
    generate_vector,
    jax_array_eq,
    numpy_array_eq,
    vector_list_eq,
)


class NumPyVectorMatrixTest(unittest.TestCase):

    def test_initialization(self):
        matrix = generate_matrix()
        vector_matrix = NumPyVectorMatrix(matrix)
        self.assertTrue(numpy_array_eq(vector_matrix.matrix, matrix))
        self.assertTrue(vector_matrix.synced)
        self.assertTrue(vector_list_eq(vector_matrix._vectors, list(matrix)))

        vector_matrix = NumPyVectorMatrix()
        self.assertFalse(vector_matrix.synced)

    def test_initialization_empty_list(self):
        vector_matrix = NumPyVectorMatrix()
        self.assertSequenceEqual(vector_matrix._vectors, [])
        self.assertTrue((vector_matrix.matrix == np.array([])).all())
        self.assertFalse(vector_matrix.synced)

    def test_reindex_matrix(self):
        vector_matrix = NumPyVectorMatrix()
        matrix = generate_matrix()
        vector_matrix.reindex(matrix)
        self.assertTrue(numpy_array_eq(vector_matrix.matrix, matrix))
        self.assertTrue(vector_list_eq(vector_matrix._vectors, list(matrix)))
        self.assertTrue(vector_matrix.synced)

    def test_reindex_list(self):
        vector_matrix = NumPyVectorMatrix()
        vector_list = list(generate_matrix())
        vector_matrix.reindex(vector_list)
        self.assertTrue(vector_list_eq(vector_matrix._vectors, vector_list))
        self.assertFalse(vector_matrix.synced)

    def test_add(self):
        matrix = generate_matrix()
        vector_matrix = NumPyVectorMatrix(matrix)

        vector = generate_vector()
        vector_matrix.add(vector)
        self.assertTrue(vector_list_eq(vector_matrix._vectors, list(matrix) + [vector]))
        self.assertFalse(vector_matrix.synced)

    def test_add_fails_with_wrong_dimension(self):
        matrix = generate_matrix()
        vector_matrix = NumPyVectorMatrix(matrix)

        vector = generate_vector(dimension=matrix.shape[1] + 1)
        with self.assertRaises(ValueError) as err:
            vector_matrix.add(vector)
        self.assertEqual(
            err.exception.args[0],
            "Received add request for vector of unexpected dimension. Expected: 10, "
            "Received: 11",
        )

    def test_delete_at(self):
        matrix = generate_matrix()
        vector_matrix = NumPyVectorMatrix(matrix)

        idx_to_remove = 3
        expected_vectors = list(matrix)
        expected_vectors[idx_to_remove], expected_vectors[-1] = (
            expected_vectors[-1],
            expected_vectors[idx_to_remove],
        )
        expected_vectors.pop()

        vector_matrix.delete_at(idx_to_remove)

        self.assertTrue(vector_list_eq(vector_matrix._vectors, expected_vectors))
        self.assertFalse(vector_matrix.synced)

        # remove last element
        vector_matrix.delete_at(len(vector_matrix._vectors) - 1)
        expected_vectors.pop()
        self.assertTrue(vector_list_eq(vector_matrix._vectors, expected_vectors))
        self.assertFalse(vector_matrix.synced)

    def test_sync(self):
        matrix = generate_matrix()
        vector_matrix = NumPyVectorMatrix(matrix)

        vector = generate_vector()
        vector_matrix.add(vector)
        vector_matrix.sync()

        self.assertTrue(
            numpy_array_eq(vector_matrix.matrix, np.array(list(matrix) + [vector]))
        )
        self.assertTrue(vector_matrix.synced)

    @patch("numpy.array")
    def test_sync_skips_when_synced(self, MockNumPyArrayInit: MagicMock):
        matrix = generate_matrix()
        vector_matrix = NumPyVectorMatrix(matrix)
        vector_matrix.sync()
        # Only should have been called in init when setting up default values.
        MockNumPyArrayInit.assert_called_once_with([])

    @patch("vector_matrix_store.matrix.NumPyVectorMatrix._compute_similarity_scores")
    @patch("vector_matrix_store.matrix.NumPyVectorMatrix.sync")
    def test_search_nearest_neighbors(
        self, MockSync: MagicMock, MockCompute: MagicMock
    ):
        matrix = generate_matrix()
        vector_matrix = NumPyVectorMatrix(matrix)

        MockCompute.return_value = (np.array([0.2, 0.1, 0.3]), np.array([2, 0, 1]))

        vector = generate_vector()
        results = vector_matrix.search_nearest_neighbors(vector, k=5)

        self.assertCountEqual(
            results,
            [ScoredNeighbor(2, 0.3), ScoredNeighbor(0, 0.2), ScoredNeighbor(1, 0.1)],
        )

        MockSync.assert_called_once()
        MockCompute.assert_called_once_with(vector)

    def test_get_entry_count(self):
        matrix = generate_matrix()
        vector_matrix = NumPyVectorMatrix(matrix)
        self.assertEqual(vector_matrix.get_entry_count(), matrix.shape[0])

    def test_get_vector_dimension(self):
        matrix = generate_matrix()
        vector_matrix = NumPyVectorMatrix(matrix)
        self.assertEqual(vector_matrix.get_vector_dimension(), matrix.shape[1])


class JaxVectorMatrixTest(unittest.TestCase):
    def test_initialization(self):
        matrix = generate_jax_matrix()
        vector_matrix = JaxVectorMatrix(matrix)
        self.assertTrue(jax_array_eq(vector_matrix.matrix, matrix))

    def test_add(self):
        matrix = generate_jax_matrix()
        vector_matrix = JaxVectorMatrix(matrix)

        vector = generate_vector()
        vector_matrix.add(vector)

        self.assertTrue(
            jax_array_eq(
                vector_matrix.matrix, jnp.append(matrix, jnp.array([vector]), axis=0)
            )
        )

    def test_delete_at(self):
        matrix = generate_jax_matrix()
        vector_matrix = JaxVectorMatrix(matrix)

        target_idx = 2

        vector_matrix.delete_at(target_idx)

        expected_matrix = matrix.at[target_idx].set(matrix[-1])[:-1, :]
        self.assertTrue(jax_array_eq(vector_matrix.matrix, expected_matrix))

    def test_sync_unimplemented(self):
        vector_matrix = JaxVectorMatrix(generate_jax_matrix())
        with self.assertRaises(NotImplementedError):
            vector_matrix.sync()

    @patch("vector_matrix_store.matrix._compute_similarity_scores")
    def test_search_nearest_neighbors(self, MockCompute: MagicMock):
        matrix = generate_jax_matrix()
        vector_matrix = JaxVectorMatrix(matrix)

        MockCompute.return_value = (jnp.array([0.2, 0.1, 0.3]), jnp.array([2, 0, 1]))

        vector = generate_vector()
        results = vector_matrix.search_nearest_neighbors(vector, k=5)

        expected_results = [
            ScoredNeighbor(2, 0.3),
            ScoredNeighbor(0, 0.2),
            ScoredNeighbor(1, 0.1),
        ]
        for i in range(len(results)):
            self.assertEqual(results[i].index, expected_results[i].index)
            self.assertAlmostEqual(results[i].score, expected_results[i].score)

        mock_compute_call_args = MockCompute.call_args[0]
        self.assertTrue(jax_array_eq(mock_compute_call_args[0], jnp.array(vector)))
        self.assertTrue(jax_array_eq(mock_compute_call_args[1], vector_matrix.matrix))

    def test_get_entry_count(self):
        matrix = generate_jax_matrix()
        vector_matrix = JaxVectorMatrix(matrix)
        self.assertEqual(vector_matrix.get_entry_count(), matrix.shape[0])

    def test_get_vector_dimension(self):
        matrix = generate_jax_matrix()
        vector_matrix = JaxVectorMatrix(matrix)
        self.assertEqual(vector_matrix.get_vector_dimension(), matrix.shape[1])


if __name__ == "__main__":
    unittest.main()
