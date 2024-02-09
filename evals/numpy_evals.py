from typing import List

import numpy as np
from numpy.typing import NDArray
import random as rrandom
import time
from tqdm import tqdm
from vector_matrix_store.metrics_util import eval_timer
from vector_matrix_store.vector_store import NumPyVectorStore

from evals.utils import generate_np_vector, generate_vector_list, vectors_to_embeddings


@eval_timer(nruns=20)
def profile_search(vector: NDArray, vector_store: NumPyVectorStore):
    vector_store.search(vector)


if __name__ == "__main__":
    embeddings = vectors_to_embeddings(generate_vector_list())
    vector_store = NumPyVectorStore.from_embeddings(embeddings)
    np_target_vec = generate_np_vector()
    profile_search(np_target_vec, vector_store)
