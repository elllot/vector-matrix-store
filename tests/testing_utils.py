import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


MOCK_VECTOR_DIMENSION = 10


def generate_matrix(
    dimension: int = MOCK_VECTOR_DIMENSION, vector_count: int = 5
) -> NDArray:
    return np.random.randn(vector_count, dimension)


def generate_jax_matrix(
    dimension: int = MOCK_VECTOR_DIMENSION, vector_count: int = 5
) -> jax.Array:
    return jnp.array(np.random.randn(vector_count, dimension))


def generate_vector(dimension: int = MOCK_VECTOR_DIMENSION) -> NDArray:
    return np.random.randn(dimension)


def generate_jax_vector(dimension: int = MOCK_VECTOR_DIMENSION) -> jax.Array:
    return jnp.array(np.random.randn(dimension))


def numpy_array_eq(array_1: NDArray, array_2: NDArray) -> bool:
    return (array_1 == array_2).all()


def jax_array_eq(array_1: jax.Array, array_2: jax.Array) -> bool:
    return jnp.equal(array_1, array_2).all()  # type: ignore


def vector_list_eq(vector_list_1: list[NDArray], vector_list_2: list[NDArray]) -> bool:
    if len(vector_list_1) != len(vector_list_2):
        return False
    return all(
        [
            numpy_array_eq(vector_list_1[i], vector_list_2[i])
            for i in range(len(vector_list_1))
        ]
    )
