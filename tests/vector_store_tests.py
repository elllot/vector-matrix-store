import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np

from vector_matrix_store.schema import (
    EmbeddingMethodType,
    EmbeddingVector,
    ScoredNeighbor,
    VectorStoreType,
)
from vector_matrix_store.vector_index import VectorIndex
from vector_matrix_store.vector_store import (
    _build_default_config,
    _validate_embeddings,
    JaxVectorStore,
    NumPyVectorStore,
    VectorStoreConfig,
)

from tests.mocks import (
    MockNumPyVectorMatrix,
    MockJaxVectorMatrix,
    MockVectorIndex,
    MOCK_DIMENSION,
)


class NumPyVectorStoreTest(unittest.TestCase):

    def test_initialization(self):
        store_config = VectorStoreConfig(
            vector_dimension=MOCK_DIMENSION,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
        )
        vector_matrix = MockNumPyVectorMatrix()
        vector_index = MockVectorIndex()
        NumPyVectorStore(vector_matrix, vector_index, store_config)

    def test_initialization_invalid_type_fails(self):
        store_config = VectorStoreConfig(
            vector_dimension=MOCK_DIMENSION,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.JAX,
        )
        vector_matrix = MockNumPyVectorMatrix()
        vector_index = MockVectorIndex()
        with self.assertRaises(AssertionError) as err:
            NumPyVectorStore(vector_matrix, vector_index, store_config)
        self.assertTrue(
            "Store type must be VectorStoreType.NUMPY" in err.exception.args[0]
        )

    def test_initialization_invalid_dimension_fails(self):
        store_config = VectorStoreConfig(
            vector_dimension=1,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
        )
        vector_matrix = MockNumPyVectorMatrix()
        vector_index = MockVectorIndex()
        with self.assertRaises(AssertionError) as err:
            NumPyVectorStore(vector_matrix, vector_index, store_config)
        self.assertTrue("Invalid dimension" in err.exception.args[0])

    # Patching initializer to avoid calling the actual NumPyVectorMatrix constructor,
    # which is tested separately.
    @patch("vector_matrix_store.vector_store.NumPyVectorStore.__init__")
    @patch("vector_matrix_store.vector_store.NumPyVectorMatrix")
    @patch("vector_matrix_store.vector_store.VectorIndex")
    @patch("vector_matrix_store.vector_store._validate_embeddings")
    def test_from_embeddings(
        self,
        MockValidate: MagicMock,
        MockVectorIndexInit: MagicMock,
        MockNumPyVectorMatrixInit: MagicMock,
        MockNumPyVectorStoreInit: MagicMock,
    ):
        MockNumPyVectorStoreInit.return_value = None
        vectors = [
            np.array([1, 2, 3]),
            np.array([3, 4, 5]),
            np.array([6, 7, 8]),
        ]
        embeddings = [
            EmbeddingVector(
                vector=v,
                embedding_method_type=EmbeddingMethodType.CUSTOM,
            )
            for v in vectors
        ]
        store_config = VectorStoreConfig(
            vector_dimension=3,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
        )

        mock_vector_index = MockVectorIndex()
        MockVectorIndexInit.return_value = mock_vector_index

        mock_vector_matrix = MockNumPyVectorMatrix()
        MockNumPyVectorMatrixInit.return_value = mock_vector_matrix

        NumPyVectorStore.from_embeddings(embeddings, store_config)

        MockValidate.assert_called_once_with(embeddings, store_config)
        mock_vector_index.add.assert_has_calls(
            [call(e.embedding_id) for e in embeddings]
        )
        MockNumPyVectorMatrixInit.assert_called_once()
        self.assertTrue(
            (MockNumPyVectorMatrixInit.call_args[0] == np.array(vectors)).all()
        )
        MockNumPyVectorStoreInit.assert_called_once_with(
            mock_vector_matrix, mock_vector_index, store_config
        )

    @patch("vector_matrix_store.vector_store.NumPyVectorStore.validate_store")
    def test_add_embedding(self, MockValidateStore: MagicMock):
        store_config = VectorStoreConfig(
            vector_dimension=MOCK_DIMENSION,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
        )
        vector_matrix = MockNumPyVectorMatrix()
        vector_index = MockVectorIndex()

        # Take action.
        store = NumPyVectorStore(vector_matrix, vector_index, store_config)
        embedding = EmbeddingVector(
            vector=np.array([1, 2, 3]),
            embedding_method_type=EmbeddingMethodType.CUSTOM,
        )
        store.add_embedding(embedding)

        # Assert behavior.
        vector_index.add.assert_called_once_with(embedding.embedding_id)
        vector_matrix.add.assert_called_once_with(embedding.vector)
        MockValidateStore.assert_called_once()

    @patch("vector_matrix_store.vector_store.NumPyVectorStore.validate_store")
    def test_delete_embedding(self, MockValidateStore: MagicMock):
        test_embedding_id = "test_eid"
        test_embedding_idx = 1
        store_config = VectorStoreConfig(
            vector_dimension=MOCK_DIMENSION,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
        )
        vector_matrix = MockNumPyVectorMatrix()
        vector_index = MockVectorIndex()
        vector_index.delete.return_value = test_embedding_idx

        # Take action.
        store = NumPyVectorStore(vector_matrix, vector_index, store_config)
        store.delete_embedding(test_embedding_id)

        # Assert behavior.
        vector_index.delete.assert_called_once_with(test_embedding_id)
        vector_matrix.delete_at.assert_called_once_with(test_embedding_idx)
        MockValidateStore.assert_called_once()

    @patch("vector_matrix_store.vector_store.NumPyVectorStore.validate_store")
    def test_search(self, MockValidateStore: MagicMock):
        store_config = VectorStoreConfig(
            vector_dimension=MOCK_DIMENSION,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
        )
        vector_matrix = MockNumPyVectorMatrix()
        vector_matrix.search_nearest_neighbors.return_value = [
            ScoredNeighbor(index=0, score=0.82),
            ScoredNeighbor(index=5, score=0.71),
            ScoredNeighbor(index=4, score=0.55),
            ScoredNeighbor(index=2, score=0.11),
        ]
        vids = ["vid1", "vid2", "vid3", "vid4"]

        vector_index = MockVectorIndex()
        vector_index.get_vid.side_effect = vids

        store = NumPyVectorStore(vector_matrix, vector_index, store_config)

        target_vector = np.array([1, 2, 3])
        results = store.search(target_vector, k=10)

        vector_matrix.search_nearest_neighbors.assert_called_once_with(
            target_vector, 10
        )
        vector_index.get_vid.assert_has_calls([call(0), call(5), call(4), call(2)])

        self.assertSequenceEqual(
            results,
            [
                ScoredNeighbor(index=0, score=0.82, embedding_id="vid1"),
                ScoredNeighbor(index=5, score=0.71, embedding_id="vid2"),
                ScoredNeighbor(index=4, score=0.55, embedding_id="vid3"),
                ScoredNeighbor(index=2, score=0.11, embedding_id="vid4"),
            ],
        )
        MockValidateStore.assert_called_once()

    def test_validate_store(self):
        store_config = VectorStoreConfig(
            vector_dimension=MOCK_DIMENSION,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
        )
        # Initializing mock matrix and index with equal count values.
        vector_matrix = MockNumPyVectorMatrix(entry_count=10)
        vector_index = MockVectorIndex(vector_count=10)

        store = NumPyVectorStore(vector_matrix, vector_index, store_config)
        store.validate_store()

        vector_matrix.assert_get_entry_count_called_once()
        vector_index.assert_get_vector_count_called_once()

    def test_validate_store_differing_counts_fails(self):
        store_config = VectorStoreConfig(
            vector_dimension=MOCK_DIMENSION,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
        )
        # Initializing mock matrix and index with differing count values.
        vector_matrix = MockNumPyVectorMatrix(entry_count=5)
        vector_index = MockVectorIndex(vector_count=10)

        store = NumPyVectorStore(vector_matrix, vector_index, store_config)
        with self.assertRaises(ValueError) as err:
            store.validate_store()

        vector_matrix.assert_get_entry_count_called_once()
        vector_index.assert_get_vector_count_called_once()
        self.assertEqual(
            "Matrix entries does not match embedding mappings. Matrix has 5 entries "
            "while mapping has 10 entries.",
            err.exception.args[0],
        )


class JaxVectorMatrixTest(unittest.TestCase):

    def test_initialization(self):
        store_config = VectorStoreConfig(
            vector_dimension=MOCK_DIMENSION,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.JAX,
        )
        vector_matrix = MockNumPyVectorMatrix()
        vector_index = MockVectorIndex()
        JaxVectorStore(vector_matrix, vector_index, store_config)

    def test_initialization_invalid_type_fails(self):
        store_config = VectorStoreConfig(
            vector_dimension=MOCK_DIMENSION,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
        )
        vector_matrix = MockNumPyVectorMatrix()
        vector_index = MockVectorIndex()
        with self.assertRaises(AssertionError) as err:
            JaxVectorStore(vector_matrix, vector_index, store_config)
        self.assertTrue(
            "Store type must be VectorStoreType.JAX" in err.exception.args[0]
        )

    def test_initialization_invalid_dimension_fails(self):
        store_config = VectorStoreConfig(
            vector_dimension=1,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.JAX,
        )
        vector_matrix = MockNumPyVectorMatrix()
        vector_index = MockVectorIndex()
        with self.assertRaises(AssertionError) as err:
            JaxVectorStore(vector_matrix, vector_index, store_config)
        self.assertTrue("Invalid dimension" in err.exception.args[0])

    # Patching initializer to avoid calling the actual NumPyVectorMatrix constructor,
    # which is tested separately.
    @patch("vector_matrix_store.vector_store.JaxVectorStore.__init__")
    @patch("vector_matrix_store.vector_store.JaxVectorMatrix")
    @patch("vector_matrix_store.vector_store.VectorIndex")
    @patch("vector_matrix_store.vector_store._validate_embeddings")
    def test_from_embeddings(
        self,
        MockValidate: MagicMock,
        MockVectorIndexInit: MagicMock,
        MockJaxVectorMatrixInit: MagicMock,
        MockJaxVectorStoreInit: MagicMock,
    ):
        MockJaxVectorStoreInit.return_value = None
        vectors = [
            np.array([1, 2, 3]),
            np.array([3, 4, 5]),
            np.array([6, 7, 8]),
        ]
        embeddings = [
            EmbeddingVector(
                vector=v,
                embedding_method_type=EmbeddingMethodType.CUSTOM,
            )
            for v in vectors
        ]
        store_config = VectorStoreConfig(
            vector_dimension=3,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.JAX,
        )

        mock_vector_index = MockVectorIndex()
        MockVectorIndexInit.return_value = mock_vector_index

        mock_vector_matrix = MockNumPyVectorMatrix()
        MockJaxVectorMatrixInit.return_value = mock_vector_matrix

        JaxVectorStore.from_embeddings(embeddings, store_config)

        MockValidate.assert_called_once_with(embeddings, store_config)
        mock_vector_index.add.assert_has_calls(
            [call(e.embedding_id) for e in embeddings]
        )
        MockJaxVectorMatrixInit.assert_called_once()
        self.assertTrue(
            (MockJaxVectorMatrixInit.call_args[0] == np.array(vectors)).all()
        )
        MockJaxVectorStoreInit.assert_called_once_with(
            mock_vector_matrix, mock_vector_index, store_config
        )

    def test_add_embedding(self): ...

    @patch("vector_matrix_store.vector_store.JaxVectorStore.validate_store")
    def test_delete_embedding(self, MockValidateStore: MagicMock):
        test_embedding_id = "test_eid"
        test_embedding_idx = 1
        store_config = VectorStoreConfig(
            vector_dimension=MOCK_DIMENSION,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.JAX,
        )
        vector_matrix = MockNumPyVectorMatrix()
        vector_index = MockVectorIndex()
        vector_index.delete.return_value = test_embedding_idx

        # Take action.
        store = JaxVectorStore(vector_matrix, vector_index, store_config)
        store.delete_embedding(test_embedding_id)

        # Assert behavior.
        vector_index.delete.assert_called_once_with(test_embedding_id)
        vector_matrix.delete_at.assert_called_once_with(test_embedding_idx)
        MockValidateStore.assert_called_once()

    @patch("vector_matrix_store.vector_store.JaxVectorStore.validate_store")
    def test_search(self, MockValidateStore: MagicMock):
        store_config = VectorStoreConfig(
            vector_dimension=MOCK_DIMENSION,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.JAX,
        )
        vector_matrix = MockJaxVectorMatrix()
        vector_matrix.search_nearest_neighbors.return_value = [
            ScoredNeighbor(index=0, score=0.82),
            ScoredNeighbor(index=5, score=0.71),
            ScoredNeighbor(index=4, score=0.55),
            ScoredNeighbor(index=2, score=0.11),
        ]
        vids = ["vid1", "vid2", "vid3", "vid4"]

        vector_index = MockVectorIndex()
        vector_index.get_vid.side_effect = vids

        store = JaxVectorStore(vector_matrix, vector_index, store_config)

        target_vector = np.array([1, 2, 3])
        results = store.search(target_vector, k=10)

        vector_matrix.search_nearest_neighbors.assert_called_once_with(
            target_vector, 10
        )
        vector_index.get_vid.assert_has_calls([call(0), call(5), call(4), call(2)])

        self.assertSequenceEqual(
            results,
            [
                ScoredNeighbor(index=0, score=0.82, embedding_id="vid1"),
                ScoredNeighbor(index=5, score=0.71, embedding_id="vid2"),
                ScoredNeighbor(index=4, score=0.55, embedding_id="vid3"),
                ScoredNeighbor(index=2, score=0.11, embedding_id="vid4"),
            ],
        )
        MockValidateStore.assert_called_once()

    def test_validate_store(self):
        store_config = VectorStoreConfig(
            vector_dimension=MOCK_DIMENSION,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.JAX,
        )
        # Initializing mock matrix and index with equal count values.
        vector_matrix = MockJaxVectorMatrix(entry_count=10)
        vector_index = MockVectorIndex(vector_count=10)

        store = JaxVectorStore(vector_matrix, vector_index, store_config)
        store.validate_store()

        vector_matrix.assert_get_entry_count_called_once()
        vector_index.assert_get_vector_count_called_once()

    def test_validate_store_differing_counts_fails(self):
        store_config = VectorStoreConfig(
            vector_dimension=MOCK_DIMENSION,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.JAX,
        )
        # Initializing mock matrix and index with differing count values.
        vector_matrix = MockJaxVectorMatrix(entry_count=5)
        vector_index = MockVectorIndex(vector_count=10)

        store = JaxVectorStore(vector_matrix, vector_index, store_config)
        with self.assertRaises(ValueError) as err:
            store.validate_store()

        vector_matrix.assert_get_entry_count_called_once()
        vector_index.assert_get_vector_count_called_once()
        self.assertEqual(
            "Matrix entries does not match embedding mappings. Matrix has 5 entries "
            "while mapping has 10 entries.",
            err.exception.args[0],
        )


class HelperMethodsTest(unittest.TestCase):

    def test_build_default_config_success(self):
        vectors = [
            EmbeddingVector(
                vector=np.array([1, 2, 3]),
                embedding_method_type=EmbeddingMethodType.CUSTOM,
            ),
            EmbeddingVector(
                vector=np.array([3, 4, 5]),
                embedding_method_type=EmbeddingMethodType.CUSTOM,
            ),
        ]
        store_cfg = _build_default_config(
            vectors, vector_store_type=VectorStoreType.NUMPY
        )

        self.assertEqual(store_cfg.vector_dimension, 3)
        self.assertEqual(store_cfg.embedding_method_type, EmbeddingMethodType.CUSTOM)
        self.assertEqual(store_cfg.vector_store_type, VectorStoreType.NUMPY)

    def test_build_default_config_empty_embeddings(self):
        # Empty embedding vector should error.
        with self.assertRaises(IndexError):
            _build_default_config([], VectorStoreType.NUMPY)

    def test_validate_embeddings_success(self):
        vectors = [
            EmbeddingVector(
                vector=np.array([1, 2, 3]),
                embedding_method_type=EmbeddingMethodType.CUSTOM,
            ),
            EmbeddingVector(
                vector=np.array([3, 4, 5]),
                embedding_method_type=EmbeddingMethodType.CUSTOM,
            ),
        ]
        store_config = VectorStoreConfig(
            vector_dimension=3,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
        )

        # Should raise error when appropriate.
        _validate_embeddings(vectors, store_config)

    def test_validate_embeddings_empty_embeddings(self):
        store_config = VectorStoreConfig(
            vector_dimension=3,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
        )
        _validate_embeddings([], store_config)

    def test_validate_embeddings_embedding_method_mismatch(self):
        vectors = [
            EmbeddingVector(
                vector=np.array([1, 2, 3]),
                embedding_method_type=EmbeddingMethodType.CUSTOM,
            ),
            EmbeddingVector(
                vector=np.array([3, 4, 5]),
                embedding_method_type=EmbeddingMethodType.GENAI_GECKO,
            ),
        ]
        store_config = VectorStoreConfig(
            vector_dimension=3,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
        )

        with self.assertRaises(ValueError) as ve:
            _validate_embeddings(vectors, store_config)
        self.assertEqual(
            ve.exception.args[0],
            f"Embedding vector (ID: {vectors[1].embedding_id}) has method type "
            f"{EmbeddingMethodType.GENAI_GECKO} does not match the configured method "
            f"type in the VectorStoreConfig: {EmbeddingMethodType.CUSTOM}.",
        )

    def test_validate_embeddings_dimension_mismatch(self):
        vectors = [
            EmbeddingVector(
                vector=np.array([1, 2, 3]),
                embedding_method_type=EmbeddingMethodType.CUSTOM,
            ),
            EmbeddingVector(
                vector=np.array([3, 4, 5]),
                embedding_method_type=EmbeddingMethodType.GENAI_GECKO,
            ),
        ]
        store_config = VectorStoreConfig(
            vector_dimension=5,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
        )

        with self.assertRaises(ValueError) as ve:
            _validate_embeddings(vectors, store_config)
        self.assertEqual(
            ve.exception.args[0],
            f"Embedding vector (ID: {vectors[0].embedding_id}) has dimension "
            f"{vectors[0].get_dimension()}, which does not match the configured "
            f"dimension in the VectorStoreConfig: {store_config.vector_dimension}.",
        )


if __name__ == "__main__":
    unittest.main()
