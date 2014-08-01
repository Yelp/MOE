# -*- coding: utf-8 -*-
"""Simple context manager for logging timing information.

TODO(GH-299): Make this part of a more complete monitoring setup, flesh out timing tools.
TODO(GH-299): Add a decorator for timing functions.

"""
import contextlib
import logging
import time


@contextlib.contextmanager
def timing_context(name):
    """Context manager that logs the runtime of the body of the with-statement.

    Uses time.clock() for measurement; not appropriate for fast-running code.
    Consider the ``timeit`` library for such situations.

    :param name: name to log with this timing information
    :type name: str

    """
    if "log" not in timing_context.__dict__:
        timing_context.log = logging.getLogger(__name__)

    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    timing_context.log.info("{0:s}: {1:f} secs".format(name, elapsed_time))
