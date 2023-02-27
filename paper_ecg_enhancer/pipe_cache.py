import cv2
import os
import shutil
from pathlib import Path
from uuid import uuid4
import numpy as np

def pipe_cache(log_prefix='', debug_logs=False):
  cache = dict()
  log_counter = 0
  log_dir = None

  if debug_logs:
    # Import the output dir only if we are going to log something.
    # Otherwise, we don't need it to be initialized.
    from .config import OUTPUT_DIR

    log_dir = Path(os.path.join(OUTPUT_DIR, log_prefix)) if log_prefix else str(uuid4())
    shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir()

  def generate_log_id(name):
    nonlocal log_counter
    id = '_'.join(filter(bool, [str(log_counter), name]))
    log_counter += 1
    return id

  def log_value(id, value):
    if isinstance(value, np.ndarray) and value.dtype == np.uint8:
      cv2.imwrite(str(log_dir.joinpath('{}.png'.format(id))), value)
    else:
      print('{}: {}'.format(id, value))

  def load_any(key):
    if isinstance(key, tuple):
      return tuple([load_any(x) for x in key])
    return key.resolve() if isinstance(key, K) else key

  class K:
    """
    A wrapper for a cache key. When an instance of this class is encountered as
    an argument to a piped function, it is replaced during that function's invocation
    with the cached value stored under that key.
    """
    def __init__(self, value, expression = lambda x: x):
      self.value = value
      self.expression = expression

    def resolve(self):
      resolved_value = cache[self.value] if isinstance(self.value, str) else self.value.resolve()
      return self.expression(resolved_value)

    def __getitem__(self, key):
      return K(self, lambda x: x[key])

    def __add__(self, other):
      return K(self, lambda x: x + other)

    def __sub__(self, other):
      return K(self, lambda x: x - other)

    def __mul__(self, other):
      return K(self, lambda x: x * other)

    def __truediv__(self, other):
      return K(self, lambda x: x / other)

    def __floordiv__(self, other):
      return K(self, lambda x: x // other)

    def __pow__(self, other):
      return K(self, lambda x: x ** other)

  def load(key):
    """
    Loads a value from the cache. Use this function inside piped lambdas
    to load cached values dynamically.
    """
    return cache[key]

  def _(fn, *args, K='', **kwargs):
    """
    Creates a pipe-able wrapper of the given function, passing the given args and kwargs
    along with any additional (kw)args passed to the wrapper itself.

    Parameters
    ----------
    fn : function
      The function to wrap
    K : str
      If a nonempty string is passed, it will be used as a key to store the return
      value of the function in the pipe cache.
    """
    def wrapper(*other_args, **other_kwargs):
      nonlocal args, kwargs
      args = [load_any(arg) for arg in args]
      kwargs = { k: load_any(v) for k, v in kwargs.items() }
      result = fn(*(args + list(other_args)), **{ **kwargs, **other_kwargs })

      if K:
        cache[K] = result

      if debug_logs:
        log_value(generate_log_id(K or fn.__name__), result)

      return result

    return wrapper

  def tap(fn, *args, K=None, **kwargs):
    """
    Same as `_()`, but passes the input value down along the pipeline, instead of
    the return value of the wrapped function. The purpose of this function is to
    perform intemediate operations "on the side" without disrupting the pipeline.
    However, piped functions can also store values in the pipe cache.

    See Also
    --------
    _ : Apply a piped function
    """
    apply = _(fn, *args, K=K, **kwargs)
    def wrapper(pipe_result, *other_args, **other_kwargs):
      apply(pipe_result, *other_args, **other_kwargs)
      return pipe_result

    return wrapper

  return K, load, _, tap
