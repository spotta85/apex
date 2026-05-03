"""Demo with MMapExpertStore in place of the eager npy store.

Cache holds np.memmap views; OS pages in lazily as the matmul touches
them. Final type check at the end confirms the swap landed correctly.
"""

import numpy as np

from apex.demo_modes import make_mmap_runtime

bundle = make_mmap_runtime()

print("First run:")
bundle.runtime.run()

print("\nSecond run:")
final = bundle.runtime.run()

print("\nFinal hidden state:")
print(final)

# Visible signal that swapping the storage backend left the rest of
# the system intact: cached experts are ExpertWeights bundles whose
# projections are np.memmap, not fully-loaded ndarrays. Inspect the
# gate_proj as a representative projection.
sample_key = next(iter(bundle.cache.cache))
sample = bundle.cache.cache[sample_key]
print(f"\nCached expert at {sample_key}:")
print(f"  bundle type:           {type(sample).__name__}")
print(f"  gate_proj shape:       {sample.gate_proj.shape}")
print(f"  gate_proj dtype:       {sample.gate_proj.dtype}")
print(f"  gate_proj is np.memmap: {isinstance(sample.gate_proj, np.memmap)}")

bundle.prefetcher.shutdown()
