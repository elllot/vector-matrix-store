from typing import Dict, List, Optional

from types import MappingProxyType


class VectorIndex:
    def __init__(self, idx_to_vid: Optional[Dict[int, str]] = None):
        # Maps which Vector ID corresponds to a given index in the matrix.
        self._idx_to_vid: Dict[int, str] = idx_to_vid or {}
        # Maps Vector ID to their corresponding index in matrix. This reverse map is
        # generated from the idx_to_vid mapping when initialized.
        self._vid_to_idx: Dict[str, int] = {
            vid: idx for idx, vid in self._idx_to_vid.items()
        }

    @property
    def index_map(self) -> MappingProxyType[int, str]:
        return MappingProxyType(self._idx_to_vid)

    @property
    def vector_map(self) -> MappingProxyType[str, int]:
        return MappingProxyType(self._vid_to_idx)

    def add(self, vid: str) -> int:
        """Adds a vector to the index mappings.

        Args:
            vid: ID of the vector to register.

        Returns:
            int: Index of the registered vector.
        """
        if self.contains_vector(vid):
            raise ValueError(f"{vid} already registered.")

        idx = len(self._idx_to_vid)
        self._idx_to_vid[idx] = vid
        self._vid_to_idx[vid] = idx

        return idx

    def delete(self, vid: str) -> int:
        """Removes a vector from the index mappings.

        Args:
            vid: ID of the vector to remove.

        Returns:
            int: Index of the removed vector.
        """
        idx = self._vid_to_idx[vid]
        # Get the ID of the last vector currently in the matrix.
        last_idx = len(self._idx_to_vid) - 1
        swap_id = self._idx_to_vid[last_idx]
        # Update removal index with data from the vector currently at the last index.
        self._vid_to_idx[swap_id] = idx
        self._idx_to_vid[idx] = swap_id
        # Remove deletion target from mappings.
        del self._vid_to_idx[vid]
        del self._idx_to_vid[last_idx]

        return idx

    def contains_vector(self, vid: str) -> bool:
        """Checks if a vector is registered in the index."""
        return vid in self._vid_to_idx

    def get_vid(self, idx: int) -> str:
        """Returns the vector ID for a given index."""
        return self._idx_to_vid[idx]

    def get_idx(self, vid: str) -> int:
        """Returns the index position associated with a vector."""
        return self._vid_to_idx[vid]

    def get_vector_count(self) -> int:
        """Returns the number of vectors in the index."""
        return len(self._vid_to_idx)
