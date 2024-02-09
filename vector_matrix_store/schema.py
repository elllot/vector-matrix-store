from typing import Dict

from dataclasses import dataclass, field
import dataclasses_json
from numpy.typing import NDArray
import uuid

from vector_matrix_store.base_enum import BaseEnum


@dataclass
class Serializable(dataclasses_json.DataClassJsonMixin):
    """A utility class for simplifying serialization and deserialization of dataclasses."""

    dataclass_json_config = dataclasses_json.config(
        letter_case=dataclasses_json.LetterCase.CAMEL,
        undefined=None,
    )["dataclasses_json"]


@dataclass
class ScoredNeighbor:
    """A scored vector relative to a query vector.

    Index is the index of the vector in the matrix.
    """

    index: int
    score: float
    embedding_id: str = ""


class VectorStoreType(str, BaseEnum):
    """Type of the underlying vector store."""

    NUMPY = "NUMPY"
    JAX = "JAX"


class EmbeddingMethodType(str, BaseEnum):
    """Type of the method used to generate the embeddings."""

    CUSTOM = "CUSTOM"
    GENAI_GECKO = "GENAI_GECKO"
    GENAI_TWO_TOWER = "GENAI_TWO_TOWER"
    T5 = "T5"


@dataclass
class EmbeddingVector:
    """A vector embedding and its associated metadata."""

    vector: NDArray
    embedding_method_type: EmbeddingMethodType
    embedding_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def get_dimension(self):
        return self.vector.shape[0]
