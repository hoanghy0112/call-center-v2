import time
import functools


def timing_decorator(func):
    """Decorator to measure the execution time of a function and return both
    the result and the elapsed time."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start timing
        result = func(*args, **kwargs)
        end_time = time.perf_counter()  # End timing
        elapsed_time = end_time - start_time
        return result, elapsed_time

    return wrapper
