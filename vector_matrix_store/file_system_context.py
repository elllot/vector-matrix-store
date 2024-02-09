from typing import Dict, Optional, Tuple

import jax.numpy as jnp
import json
import numpy as np
import os

from vector_matrix_store.constants import DEFAULT_VECTOR_STORE_CONFIG_PATH
from vector_matrix_store.matrix import NumPyVectorMatrix, JaxVectorMatrix
from vector_matrix_store.schema import (
    VectorStoreType,
)
from vector_matrix_store.vector_index import VectorIndex
from vector_matrix_store.vector_store import (
    JaxVectorStore,
    NumPyVectorStore,
    VectorStore,
    VectorStoreConfig,
)


class FileSystemContext:
    def __init__(self, directory_path: str = ""):
        self._directory_path = directory_path or os.getcwd()

    def load_vector_store_config(
        self, relative_path: str = DEFAULT_VECTOR_STORE_CONFIG_PATH
    ) -> VectorStoreConfig:
        with open(self._build_path_from_relative(relative_path), "r") as file:
            return VectorStoreConfig.from_json(file.read())

    def write_vector_store_config(self, cfg: VectorStoreConfig, overwrite: bool = True):
        config_path = self._build_path_from_relative(
            cfg.vector_store_config_relative_path
        )
        if not overwrite and os.path.exists(config_path):
            raise FileExistsError(
                f"Vector Store Config file already exists at {config_path}. Aborting "
                "writing due to overwrite option not specified"
            )
        with open(config_path, "w") as file:
            file.write(cfg.to_json(indent=2))

    def load_store(self, cfg: Optional[VectorStoreConfig] = None) -> VectorStore:
        cfg = cfg or self.load_vector_store_config()
        vstore_type = cfg.vector_store_type
        if vstore_type not in VectorStoreType:
            raise ValueError(f"Unsupported vector store type: {vstore_type}")

        vector_matrix_full_path = self._build_path_from_relative(
            cfg.vector_matrix_relative_path
        )
        with open(vector_matrix_full_path, "r") as file:
            data = json.loads(file.read())
            index = VectorIndex({int(k): v for k, v in data.items()})
            print("vector_count: " + str(index.get_vector_count()))

        if vstore_type == VectorStoreType.JAX:
            matrix = jnp.load(vector_matrix_full_path, allow_pickle=False)
            store = JaxVectorStore(JaxVectorMatrix(matrix), index, cfg)
        else:
            matrix = np.load(vector_matrix_full_path)
            store = NumPyVectorStore(NumPyVectorMatrix(matrix), index, cfg)

        return store

    def write_store(self, store: VectorStore, overwrite: bool = True):
        cfg = store.store_config
        matrix_path = self._build_path_from_relative(cfg.vector_matrix_relative_path)
        index_path = self._build_path_from_relative(cfg.vector_index_relative_path)
        if not overwrite:
            if os.path.exists(matrix_path):
                raise FileExistsError(
                    f"Vector Matrix file already exists at {matrix_path}. Aborting "
                    "writing due to overwrite option not specified"
                )
            if os.path.exists(index_path):
                raise FileExistsError(
                    f"Vector Index file already exists at {index_path}. Aborting "
                    "writing due to overwrite option not specified"
                )

        # Write the vector matrix to disc.
        if cfg.vector_store_type == VectorStoreType.JAX:
            jnp.save(matrix_path, store.vector_matrix.matrix)
        else:
            # Ensure the vector matrix is in sync before writing to disc. The NumPy
            # matrix requires roughly 1/3 of the memory required for storing as a json
            # array of arrays.
            store.vector_matrix.sync()
            np.save(matrix_path, store.vector_matrix.matrix)

        # Write the vector index to disc.
        with open(index_path, "w") as file:
            # Save the index mapping to a JSON file. Only the idx_to_vid mapping is
            # preserved to minimize storage size, as the vid_to_idx mapping can be
            # generated from it.
            file.write(json.dumps(store.vector_index.index_map, indent=2))
        self.write_vector_store_config(cfg)

    def _build_path_from_relative(self, relative_path: str) -> str:
        return f"{self._directory_path}/{relative_path}"
