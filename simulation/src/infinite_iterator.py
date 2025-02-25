"""
Module Name: infinite_iterator.py
Author: taken from https://github.com/brendel-group/cl-ica
Description: Create a iterator that we can infinitely loop over
"""

from typing import Iterable

class InfiniteIterator:
    """Infinitely repeat the iterable."""

    def __init__(self, iterable: Iterable):
        self._iterable = iterable
        self.iterator = iter(self._iterable)

    def __iter__(self):
        return self

    def __next__(self):
        for _ in range(2):
            try:
                return next(self.iterator)
            except StopIteration:
                # reset iterator
                del self.iterator
                self.iterator = iter(self._iterable)
