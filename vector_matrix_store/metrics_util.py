from functools import wraps
import numpy as np
import time
import timeit
from tqdm import tqdm


class TimerContext:
    """Context that helps time nested code.

    Example Usage:
    ```
        # do something
        with TimerContext():
            fn_to_time()
            # do more operations.
        # continue after timing
    ```
    """

    def __init__(self, name=None):
        self._name = name or f"'{name}'"

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start) * 1000.0
        print("Code block" + self._name + " took: " + str(self.took) + " ms")


def timer(func):
    @wraps(func)
    def wrap_fn(*args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print("{} completed in {} secs".format(repr(func.__name__), round(elapsed, 3)))
        return value

    return wrap_fn


def eval_timer(nruns=10):
    def timer(func):
        @wraps(func)
        def wrap_fn(*args, **kwargs):

            runs = []
            print(f"Profiling {func.__name__} {nruns} times")
            for _ in tqdm(range(nruns)):
                s = time.perf_counter()
                func(*args, **kwargs)
                runs.append((time.perf_counter() - s))

            avg_run_time_ms = float(np.mean(runs)) * 1000
            longest_ms = np.max(runs) * 1000
            shortest_ms = np.min(runs) * 1000
            std_ms = float(np.std(runs)) * 1000

            print(f"Run statistics for func - {func.__name__}:")
            print(f"    Average: {round(avg_run_time_ms, 6)} ms")
            print(f"    Standard deviation: {round(std_ms, 6)} ms")
            print(f"    Longest run: {round(longest_ms, 6)} ms")
            print(f"    Shortest run: {round(shortest_ms, 6)} ms")

        return wrap_fn

    return timer
