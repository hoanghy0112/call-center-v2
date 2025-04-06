import time
import functools


def timing_decorator(func):
    """Decorator that wraps a generator and yields (value, elapsed_time_since_start)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        gen = func(*args, **kwargs)

        for item in gen:
            current_time = time.perf_counter()
            elapsed_time = current_time - start_time
            yield (item, elapsed_time)
            start_time = time.perf_counter()

    return wrapper
