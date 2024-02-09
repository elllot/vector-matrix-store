import unittest

from vector_matrix_store.vector_index import VectorIndex

_VID_0 = "vid0"
_VID_1 = "vid1"
_VID_2 = "vid2"


class VectorIndexTests(unittest.TestCase):

    def test_initialization(self):
        vector_index = VectorIndex()
        self.assertEqual(vector_index.index_map, {})
        self.assertEqual(vector_index.vector_map, {})

        vector_index_with_data = VectorIndex({0: "vid0", 1: "vid1"})
        self.assertCountEqual(vector_index_with_data.index_map, {0: "vid0", 1: "vid1"})
        self.assertCountEqual(vector_index_with_data.vector_map, {"vid0": 0, "vid1": 1})

    def test_add(self):
        vector_index = VectorIndex()
        vector_index.add(_VID_0)
        expected_idx_map = {0: _VID_0}
        expected__VID_map = {_VID_0: 0}
        self.assertCountEqual(vector_index.index_map, expected_idx_map)
        self.assertCountEqual(vector_index.vector_map, expected__VID_map)

        vector_index.add(_VID_1)
        expected_idx_map = {0: _VID_0, 1: _VID_1}
        expected__VID_map = {_VID_0: 0, _VID_1: 1}
        self.assertCountEqual(vector_index.index_map, expected_idx_map)
        self.assertCountEqual(vector_index.vector_map, expected__VID_map)

        # Test adding a duplicate vector raises an exception.
        with self.assertRaises(ValueError) as ve:
            vector_index.add(_VID_1)
        self.assertEqual(ve.exception.args[0], f"{_VID_1} already registered.")

    def test_delete(self):
        vector_index = VectorIndex()
        vector_index.add(_VID_0)
        vector_index.add(_VID_1)
        vector_index.add(_VID_2)

        # Test deleting a non-existent vector.
        with self.assertRaises(KeyError):
            vector_index.delete("vid3")

        vector_index.delete(_VID_0)
        expected_idx_map = {0: _VID_2, 1: _VID_1}
        expected__VID_map = {_VID_1: 1, _VID_2: 0}
        self.assertCountEqual(vector_index.index_map, expected_idx_map)
        self.assertCountEqual(vector_index.vector_map, expected__VID_map)

        vector_index.delete(_VID_1)
        expected_idx_map = {0: _VID_2}
        expected__VID_map = {_VID_2: 0}
        self.assertCountEqual(vector_index.index_map, expected_idx_map)
        self.assertCountEqual(vector_index.vector_map, expected__VID_map)

        vector_index.delete(_VID_2)
        self.assertCountEqual(vector_index.index_map, {})
        self.assertCountEqual(vector_index.vector_map, {})

    def test_contains_vector(self):
        vector_index = VectorIndex()
        vector_index.add(_VID_0)

        self.assertTrue(vector_index.contains_vector(_VID_0))
        self.assertFalse(vector_index.contains_vector(_VID_1))

    def test_get_vid(self):
        vector_index = VectorIndex()
        vector_index.add(_VID_0)
        vector_index.add(_VID_1)

        self.assertEqual(vector_index.get_vid(0), _VID_0)
        self.assertEqual(vector_index.get_vid(1), _VID_1)
        with self.assertRaises(KeyError):
            vector_index.get_vid(2)

    def test_get_idx(self):
        vector_index = VectorIndex()
        vector_index.add(_VID_0)
        vector_index.add(_VID_1)

        self.assertEqual(vector_index.get_idx(_VID_0), 0)
        self.assertEqual(vector_index.get_idx(_VID_1), 1)
        with self.assertRaises(KeyError):
            vector_index.get_idx(_VID_2)

    def test_get_vector_count(self):
        vector_index = VectorIndex()
        vector_index.add(_VID_0)
        vector_index.add(_VID_1)
        vector_index.add(_VID_2)

        self.assertEqual(vector_index.get_vector_count(), 3)


if __name__ == "__main__":
    unittest.main()
