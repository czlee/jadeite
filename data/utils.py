"""Utilities for data processing."""


def divide_integer_evenly(n, m):
    """Returns a list of `m` integers summing to `n`, with elements as even as
    possible. For example:
    ```
        divide_integer_evenly(10, 4)  ->  [3, 3, 2, 2]
        divide_integer_evenly(20, 3)  ->  [7, 6, 6]
    ```
    """
    lengths = [n // m] * m
    for i in range(n - sum(lengths)):
        lengths[i] += 1
    return lengths
