from numpy.typing import NDArray
from vector_matrix_store.metrics_util import eval_timer
from vector_matrix_store.vector_store import JaxVectorStore

from evals.utils import generate_embeddings, generate_np_vector


@eval_timer(nruns=30)
def profile_search(vector: NDArray, vector_store: JaxVectorStore):
    vector_store.search(vector)


if __name__ == "__main__":
    embeddings = generate_embeddings(vector_space_size=100000)
    # TODO: add timer for this.
    vector_store = JaxVectorStore.from_embeddings(embeddings)
    target_vec = generate_np_vector()
    profile_search(target_vec, vector_store)
