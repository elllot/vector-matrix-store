import jax.numpy as jnp
import json
import numpy as np
import unittest
from unittest.mock import MagicMock, Mock, PropertyMock, mock_open, patch

from vector_matrix_store.file_system_context import FileSystemContext
from vector_matrix_store.schema import EmbeddingMethodType, VectorStoreType
from vector_matrix_store.vector_store import VectorStoreConfig

from mocks import (
    MockJaxVectorMatrix,
    MockJaxVectorStore,
    MockNumPyVectorMatrix,
    MockVectorIndex,
)


class FileSystemContextTests(unittest.TestCase):

    @patch("os.getcwd")
    def test_initialization(self, MockOsGetCwd: MagicMock):
        dir_path = "/path/to/dir"
        file_system_context = FileSystemContext(dir_path)
        MockOsGetCwd.assert_not_called()
        self.assertEqual(file_system_context._directory_path, dir_path)

    @patch("os.getcwd")
    def test_initialization_cwd(self, MockOsGetCwd: MagicMock):
        mock_cwd = "/path/to/cwd"
        MockOsGetCwd.return_value = mock_cwd
        file_system_context = FileSystemContext()
        MockOsGetCwd.assert_called_once()
        self.assertEqual(file_system_context._directory_path, mock_cwd)

    def test_load_vector_store_config(self):
        # Setup mock file.
        test_config = VectorStoreConfig(
            vector_dimension=3,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
        )

        # Make call.
        root_dir = "/path/to/root"
        fs_ctx = FileSystemContext(root_dir)
        config_relative = "to/config"
        # Patching builtin open function behavior to return preconfigured config.
        with patch(
            "builtins.open", mock_open(read_data=test_config.to_json())
        ) as mock_builtin_open:
            loaded_config = fs_ctx.load_vector_store_config(config_relative)
        mock_builtin_open.assert_called_with(f"{root_dir}/{config_relative}", "r")

        self.assertEqual(loaded_config, test_config)

    def test_load_vector_store_config_default(self): ...

    @patch("os.path.exists")
    def test_write_vector_store_config(
        self,
        MockOsPathExists: MagicMock,
    ):
        store_config_path = "nested/config.json"
        store_config = VectorStoreConfig(
            vector_dimension=3,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
            vector_store_config_relative_path=store_config_path,
        )
        MockOsPathExists.return_value = False

        root_dir = "/path/to/root"
        fs_ctx = FileSystemContext(root_dir)
        with patch("builtins.open", mock_open()) as mock_builtin_open:
            fs_ctx.write_vector_store_config(store_config)

        mock_builtin_open.assert_called_once_with(
            f"{root_dir}/{store_config_path}", "w"
        )
        mock_builtin_open.return_value.write.assert_called_once_with(
            store_config.to_json(indent=2)
        )

    @patch("os.path.exists")
    def test_write_vector_store_config_no_overwrite(
        self,
        MockOsPathExists: MagicMock,
    ):
        store_config_path = "nested/config.json"
        store_config = VectorStoreConfig(
            vector_dimension=3,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.NUMPY,
            vector_store_config_relative_path=store_config_path,
        )
        MockOsPathExists.return_value = True

        root_dir = "/path/to/root"
        fs_ctx = FileSystemContext(root_dir)
        with patch("builtins.open", mock_open()) as mock_builtin_open:
            with self.assertRaises(FileExistsError) as err:
                fs_ctx.write_vector_store_config(store_config, overwrite=False)

        MockOsPathExists.assert_called_once_with(f"{root_dir}/{store_config_path}")
        mock_builtin_open.assert_not_called()

    @patch("jax.numpy.load")
    @patch("vector_matrix_store.file_system_context.JaxVectorMatrix")
    @patch("vector_matrix_store.file_system_context.JaxVectorStore")
    @patch("vector_matrix_store.file_system_context.VectorIndex")
    def test_load_store_jax_vector_store(
        self,
        MockVectorIndexInit: MagicMock,
        MockJaxVectorStoreInit: MagicMock,
        MockJaxVectorMatrixInit: MagicMock,
        MockJaxNumPyLoad: MagicMock,
    ):
        mock_jax_vector_matrix = MockJaxVectorMatrix()
        MockJaxVectorMatrixInit.return_value = mock_jax_vector_matrix

        mock_index = MockVectorIndex()
        MockVectorIndexInit.return_value = mock_index

        # Set up json.loads for reading data.
        mock_data = {"0": "vector_id_0", "1": "vector_id_1"}

        mock_vector_store = MockJaxVectorStore()
        MockJaxVectorStoreInit.return_value = mock_vector_store

        mock_jax_matrix = jnp.array([[1, 2, 3], [3, 4, 5]])
        MockJaxNumPyLoad.return_value = mock_jax_matrix

        root_dir = "/path/to/root"
        fs_ctx = FileSystemContext(root_dir)
        store_config_path = "nested/config.json"
        store_config = VectorStoreConfig(
            vector_dimension=3,
            embedding_method_type=EmbeddingMethodType.CUSTOM,
            vector_store_type=VectorStoreType.JAX,
            vector_store_config_relative_path=store_config_path,
        )
        expected_vector_matrix_full_path = (
            f"{root_dir}/{store_config.vector_matrix_relative_path}"
        )

        # Make the call.
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(mock_data))
        ) as mock_builtin_open:
            store = fs_ctx.load_store(store_config)

        # Assert behaviors.
        mock_builtin_open.assert_called_once_with(expected_vector_matrix_full_path, "r")
        MockJaxNumPyLoad.assert_called_once_with(
            expected_vector_matrix_full_path, allow_pickle=False
        )
        MockJaxVectorMatrixInit.assert_called_once_with(mock_jax_matrix)
        MockVectorIndexInit.assert_called_once_with(
            {0: "vector_id_0", 1: "vector_id_1"}
        )
        MockJaxVectorStoreInit.assert_called_once_with(
            mock_jax_vector_matrix, mock_index, store_config
        )
        self.assertEqual(store, mock_vector_store)

    @patch("numpy.load")
    @patch("vector_matrix_store.file_system_context.NumPyVectorMatrix")
    @patch("vector_matrix_store.file_system_context.NumPyVectorStore")
    @patch("vector_matrix_store.file_system_context.VectorIndex")
    def test_load_store_numpy_vector_store(
        self,
        MockVectorIndexInit: MagicMock,
        MockNumPyVectorStoreInit: MagicMock,
        MockNumPyVectorMatrixInit: MagicMock,
        MockNumPyLoad: MagicMock,
    ):
        mock_vector_matrix = MockNumPyVectorMatrix()
        MockNumPyVectorMatrixInit.return_value = mock_vector_matrix

        mock_index = MockVectorIndex()
        MockVectorIndexInit.return_value = mock_index

        # Set up json.loads for reading data.
        mock_data = {"0": "vector_id_a", "1": "vector_id_b"}

        mock_vector_store = MockJaxVectorStore()
        MockNumPyVectorStoreInit.return_value = mock_vector_store

        mock_numpy_matrix = np.array([[1, 2, 3], [3, 4, 5]])
        MockNumPyLoad.return_value = mock_numpy_matrix

        root_dir = "/path/to/root"
        fs_ctx = FileSystemContext(root_dir)
        store_config_path = "nested/config.json"
        store_config = VectorStoreConfig(
            vector_dimension=3,
            embedding_method_type=EmbeddingMethodType.GENAI_GECKO,
            vector_store_type=VectorStoreType.NUMPY,
            vector_store_config_relative_path=store_config_path,
        )
        expected_vector_matrix_full_path = (
            f"{root_dir}/{store_config.vector_matrix_relative_path}"
        )

        # Make the call.
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(mock_data))
        ) as mock_builtin_open:
            store = fs_ctx.load_store(store_config)

        # Assert behaviors.
        mock_builtin_open.assert_called_once_with(expected_vector_matrix_full_path, "r")
        MockNumPyLoad.assert_called_once_with(expected_vector_matrix_full_path)
        MockNumPyVectorMatrixInit.assert_called_once_with(mock_numpy_matrix)
        MockVectorIndexInit.assert_called_once_with(
            {0: "vector_id_a", 1: "vector_id_b"}
        )
        MockNumPyVectorStoreInit.assert_called_once_with(
            mock_vector_matrix, mock_index, store_config
        )
        self.assertEqual(store, mock_vector_store)

    @patch(
        "vector_matrix_store.file_system_context.FileSystemContext.write_vector_store_config"
    )
    @patch("jax.numpy.save")
    def test_write_store_jax_vector_store(
        self,
        MockJNPSave: MagicMock,
        MockWriteVectorStoreConfig: MagicMock,
    ):
        mock_store = Mock()
        mock_matrix = MockJaxVectorMatrix()
        mock_store_config = VectorStoreConfig(
            vector_dimension=3,
            embedding_method_type=EmbeddingMethodType.GENAI_GECKO,
            vector_store_type=VectorStoreType.JAX,
        )
        mock_index_map = {0: "vector_id_0", 1: "vector_id_1"}
        mock_store.store_config = mock_store_config
        mock_store.vector_index.index_map = mock_index_map
        mock_store.vector_matrix.matrix = mock_matrix

        root_dir = "/path/to/root"
        fs_ctx = FileSystemContext(root_dir)

        # Make call.
        with patch("builtins.open", mock_open()) as mock_builtin_open:
            fs_ctx.write_store(mock_store)

        # Assert Behaviors.
        mock_builtin_open.assert_called_once_with(
            f"{root_dir}/{mock_store_config.vector_index_relative_path}", "w"
        )
        mock_builtin_open.return_value.write.assert_called_once_with(
            json.dumps(mock_index_map, indent=2)
        )
        MockJNPSave.assert_called_once_with(
            f"{root_dir}/{mock_store_config.vector_matrix_relative_path}",
            mock_matrix,
        )
        MockWriteVectorStoreConfig.assert_called_once_with(mock_store_config)

    @patch(
        "vector_matrix_store.file_system_context.FileSystemContext.write_vector_store_config"
    )
    @patch("numpy.save")
    def test_write_store_numpy_vector_store(
        self,
        MockNumPySave: MagicMock,
        MockWriteVectorStoreConfig: MagicMock,
    ):
        mock_store = Mock()
        mock_matrix = MockNumPyVectorMatrix()
        mock_store_config = VectorStoreConfig(
            vector_dimension=3,
            embedding_method_type=EmbeddingMethodType.GENAI_GECKO,
            vector_store_type=VectorStoreType.NUMPY,
        )
        mock_index_map = {0: "vector_id_0", 1: "vector_id_1"}
        mock_store.store_config = mock_store_config
        mock_store.vector_index.index_map = mock_index_map
        mock_store.vector_matrix.matrix = mock_matrix

        root_dir = "/path/to/root"
        fs_ctx = FileSystemContext(root_dir)

        # Make call.
        with patch("builtins.open", mock_open()) as mock_builtin_open:
            fs_ctx.write_store(mock_store)

        # Assert Behaviors.
        mock_builtin_open.assert_called_once_with(
            f"{root_dir}/{mock_store_config.vector_index_relative_path}", "w"
        )
        mock_builtin_open.return_value.write.assert_called_once_with(
            json.dumps(mock_index_map, indent=2)
        )
        MockNumPySave.assert_called_once_with(
            f"{root_dir}/{mock_store_config.vector_matrix_relative_path}",
            mock_matrix,
        )
        # Sync on the vector matrix should have been called for the numpy impl.
        mock_store.vector_matrix.sync.assert_called_once()
        MockWriteVectorStoreConfig.assert_called_once_with(mock_store_config)

    @patch("os.path.exists")
    def test_write_store_no_overwrite_matrix_exists(self, MockOsExists: MagicMock):
        root_dir = "/path/to/root"
        mock_store = Mock()
        mock_store_config = VectorStoreConfig(
            vector_dimension=3,
            embedding_method_type=EmbeddingMethodType.GENAI_GECKO,
            vector_store_type=VectorStoreType.JAX,
        )
        mock_store.store_config = mock_store_config

        def mock_os_exists_side_effect(*args, **kwargs):
            if f"{root_dir}/{mock_store_config.vector_index_relative_path}" in args:
                return False
            if f"{root_dir}/{mock_store_config.vector_matrix_relative_path}" in args:
                return True
            return False

        MockOsExists.side_effect = mock_os_exists_side_effect

        fs_ctx = FileSystemContext(root_dir)
        with self.assertRaises(FileExistsError) as err:
            fs_ctx.write_store(mock_store, overwrite=False)
        self.assertTrue("Vector Matrix file already exists" in err.exception.args[0])

    @patch("os.path.exists")
    def test_write_store_no_overwrite_index_exists(self, MockOsExists: MagicMock):
        root_dir = "/path/to/root"
        mock_store = Mock()
        mock_store_config = VectorStoreConfig(
            vector_dimension=3,
            embedding_method_type=EmbeddingMethodType.GENAI_GECKO,
            vector_store_type=VectorStoreType.JAX,
        )
        mock_store.store_config = mock_store_config

        def mock_os_exists_side_effect(*args, **kwargs):
            if f"{root_dir}/{mock_store_config.vector_index_relative_path}" in args:
                return True
            if f"{root_dir}/{mock_store_config.vector_matrix_relative_path}" in args:
                return False
            return False

        MockOsExists.side_effect = mock_os_exists_side_effect

        fs_ctx = FileSystemContext(root_dir)
        with self.assertRaises(FileExistsError) as err:
            fs_ctx.write_store(mock_store, overwrite=False)
        self.assertTrue("Vector Index file already exists" in err.exception.args[0])


if __name__ == "__main__":
    unittest.main()
