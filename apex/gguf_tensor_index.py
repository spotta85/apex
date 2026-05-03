"""Read tensor metadata + data from a .gguf file by name.

GGUF is the single-file, mmap-friendly format llama.cpp uses for model
weights. This index wraps `gguf.GGUFReader` and exposes a small
name-keyed API the rest of APEX can use without knowing GGUF internals:

  - has(name) / names()         -- inventory
  - metadata(name)              -- shape / dtype / element_count
  - tensor(name)                -- mmap-backed numpy array

Notes about the underlying gguf-py API:
  - Each ReaderTensor's `.data` is an np.memmap with the *correct*
    numpy shape (so we use it directly).
  - Each ReaderTensor's `.shape` is reversed (ggml convention: ne[0]
    is the fastest-varying axis). We report shape via `data.shape`
    instead so callers see the natural numpy ordering.
  - GGUFReader opens the file with mmap, so loaded tensors are pulled
    in lazily — same memory profile as MMapExpertStore.

This sprint only handles unquantized fp32 2D tensors. Quantized types
(Q4_0, Q8_0, etc.) need ggml-side dequantization and are deferred.
"""

from gguf import GGUFReader


class GGUFTensorIndex:
    def __init__(self, path):
        self.path = str(path)
        # Open once; mmap'd file handle lives for the index's lifetime.
        self._reader = GGUFReader(self.path)
        # Build a name -> ReaderTensor map so lookups are O(1) instead
        # of scanning the tensor list per call.
        self._by_name = {t.name: t for t in self._reader.tensors}

    def has(self, name: str) -> bool:
        return name in self._by_name

    def names(self):
        return list(self._by_name.keys())

    def metadata(self, name: str) -> dict:
        # Returns shape in numpy order (not ggml order) and a string
        # name for the dtype enum so callers don't depend on gguf types.
        t = self._require(name)
        return {
            "name": name,
            "shape": tuple(int(d) for d in t.data.shape),
            "dtype": str(t.data.dtype),
            "element_count": int(t.n_elements),
        }

    def tensor(self, name: str):
        # Returns the mmap-backed numpy array directly. The caller can
        # read / matmul against it; pages get pulled in as touched.
        return self._require(name).data

    def _require(self, name: str):
        if name not in self._by_name:
            # KeyError with the missing name + file path is the same
            # contract as the safetensors store's "missing tensor" path.
            raise KeyError(f"Tensor {name!r} not found in GGUF file {self.path!r}")
        return self._by_name[name]
