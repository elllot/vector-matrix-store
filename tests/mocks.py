from typing import Any

from unittest.mock import Mock


MOCK_ENTRY_COUNT = 5
MOCK_DIMENSION = 100


class MockNumPyVectorMatrix(Mock):
    """Mocked out NumPyVectorMatrix."""

    def __init__(
        self,
        entry_count: int = MOCK_ENTRY_COUNT,
        vector_dimension: int = MOCK_DIMENSION,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._entry_count = entry_count
        self._vector_dimension = vector_dimension
        self._get_entry_count_calls = 0
        self._get_vector_dimension_calls = 0

    def get_entry_count(self) -> int:
        self._get_entry_count_calls += 1
        return self._entry_count

    def get_vector_dimension(self) -> int:
        self._get_vector_dimension_calls += 1
        return self._vector_dimension

    def assert_get_entry_count_called_once(self) -> None:
        assert self._get_entry_count_calls == 1

    def assert_get_vector_dimension_called_once(self) -> None:
        assert self._get_vector_dimension_calls == 1


class MockJaxVectorMatrix(Mock):
    """Mocked out JaxVectorMatrix."""

    def __init__(
        self,
        entry_count: int = MOCK_ENTRY_COUNT,
        vector_dimension: int = MOCK_DIMENSION,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._entry_count = entry_count
        self._vector_dimension = vector_dimension
        self._get_entry_count_calls = 0
        self._get_vector_dimension_calls = 0

    def get_entry_count(self) -> int:
        self._get_entry_count_calls += 1
        return self._entry_count

    def get_vector_dimension(self) -> int:
        self._get_vector_dimension_calls += 1
        return self._vector_dimension

    def assert_get_entry_count_called_once(self) -> None:
        assert self._get_entry_count_calls == 1

    def assert_get_vector_dimension_called_once(self) -> None:
        assert self._get_vector_dimension_calls == 1


class MockVectorIndex(Mock):

    def __init__(
        self,
        vector_count: int = MOCK_ENTRY_COUNT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vector_count = vector_count
        self._vector_count_calls = 0

    def contains_vector(self, vid: str) -> bool: ...

    def get_idx(self, vid: str) -> int: ...

    def get_vector_count(self) -> int:
        self._vector_count_calls += 1
        return self._vector_count

    def assert_get_vector_count_called_once(self) -> None:
        assert self._vector_count_calls == 1


class MockJaxVectorStore(Mock): ...
