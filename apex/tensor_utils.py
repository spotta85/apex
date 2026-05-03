"""Tensor-shape-agnostic helpers used across the cache + backends.

The system passes around several different "tensor-ish" objects:
  - np.ndarray and np.memmap (npy / mmap / safetensors / gguf stores)
  - mx.array (after MLXBackend.prepare_expert)
  - ExpertWeights bundles (post-step-4: 3 tensors per expert)

These helpers normalize "how many bytes does this thing occupy" across
all of them so the cache's memory budget tracking stays in one place.
"""

from apex.types import ExpertWeights


def tensor_nbytes(x) -> int:
    """Return the byte footprint of any tensor-like or bundle.

    - ExpertWeights: sum of the bundle's three projections.
    - np.ndarray / np.memmap: native `.nbytes` attribute.
    - mx.array: no `.nbytes`, so fall back to size * itemsize.
    - Anything else: best-effort via duck-typed `.nbytes` attribute,
      raises TypeError if the object exposes neither.

    Centralizing this means the cache's memory accounting stays correct
    when a backend's `prepare_expert` swaps out the array type, or when
    step 4 turns each cache entry into a 3-tensor bundle.
    """
    if isinstance(x, ExpertWeights):
        # Sum the three projections so the cache budget reflects the
        # whole bundle as a single accounting unit.
        return (
            tensor_nbytes(x.gate_proj)
            + tensor_nbytes(x.up_proj)
            + tensor_nbytes(x.down_proj)
        )
    nbytes = getattr(x, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    # mx.array path: no .nbytes attribute, but .size and .itemsize work
    # (size = total element count, itemsize = bytes per element).
    size = getattr(x, "size", None)
    itemsize = getattr(x, "itemsize", None)
    if size is not None and itemsize is not None:
        return int(size) * int(itemsize)
    raise TypeError(f"Cannot determine nbytes for object of type {type(x).__name__}")
