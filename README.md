# VectorMatrixStore

A simple in-memory vector store that leverages only `numpy` and `jax` to
optimize vector search (e.g. for embedding semantic similarity search). Intended
for situations where alternatives like [FAISS](https://github.com/facebookresearch/faiss)
is not available as a dependency. The **GPU-enabled** JAX implementation beats
the **CPU** version of FAISS by ~5x. See the benchmark section for additional
details.

`VectorMatrix` is the underlying `numpy` / `jax.numpy` arrays used for computing
similarity scores. `VectorIndex` maintains the mapping between an embedding ID
and its position in the search matrix.

Provides basic utility around I/O for reading and writing the underlying objects
(matrix and index) of the store.

## Quick Start Guide

### Install with poetry (see docs for more information on installation: <https://python-poetry.org/docs/>)

> **_NOTE_** If using `VSCode`, creating the virtual env in the same folder allows VSCode to automatically pick up the venv on creation (install). This can be configured via: `poetry config virtualenvs.in-project true`

```shell
poetry install
```

If the project was already install via poetry and, therefore, has a a venv (generated in the default cache dir) already associated with the project, remove the current associated venv with the following sequence of commands:

```shell
poetry env list
poetry env remove <current environment>
poetry install  # will create a new environment using your updated configuration
```

### Running tests

```shell
poetry run python tests/<test_file>
```

## API examples

Note: See `recipes.py` for full list of examples.

## Rough benchmarks

Note: Benchmarks are based on vectors of dimension 768 and search space O(n) = 100000

Below outline benchmarks on searching. `evals/` contains the code for running
eval based on arbitrary vectors.

On CPU, the JAX implementation gives about 3x the performance of NumPy.

### JAX on GPU (ran on Google Colab - CUDA)

```shell
1.47 ms ± 1.26 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

### JAX (JaxVectorStore) on CPU (ran on MAC M1 PRO)

```shell
Run statistics for func - profile_search:
    Average: 119.901629 ms
    Standard deviation: 12.727818 ms
    Longest run: 185.600958 ms
    Shortest run: 113.682 ms
```

### NumPy (NumPyVectorStore) on CPU (ran on MAC M1 PRO)

```shell
Run statistics for func - profile_search:
    Average: 280.259775 ms
    Standard deviation: 74.657506 ms
    Longest run: 426.638959 ms
    Shortest run: 226.190125 ms
```

Note: While GPU Jax for ARM arch is available as `experimental` (using
`jax-metal`, refer to <https://github.com/google/jax?tab=readme-ov-file#installation>),
the current implementation crashes for larger size arrays (e.g. crashes on 100K).

## Future Work

- Support SVM training on vector search for improved similarity scoring beyond KNN
- Further research on JAX optimizations to improve baseline
