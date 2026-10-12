# Experiments

Every optimization tried, with numbers. Failures included.

## 2026-10-11 — A4 baseline: core.bin cold vs warm page-in

`apex bench packed/` maps core.bin (1296.7 MiB, 82,990 pages of 16 KB) and touches one byte per page.

| run | mmap_core | touch_core |
|---|---|---|
| cold (first run after pack) | 0.4 ms | 1609 ms (~800 MB/s) |
| warm (page cache) | 5 ms | 36 ms |

Memory: tracked 1296.7 MB, OS resident 1302.8 MB, OS footprint 2.5 MB. Mapped file pages are resident but not charged to the process. Budget enforcement (C10) must use footprint; resident tells whether the page cache still holds the weights.

## 2026-10-12 — B5: GPU reads the mmap'd core.bin through a no-copy buffer

`apex bench`: checksum of embed_tokens.weight (352 MB) on GPU and CPU match.

| phase | warm ms |
|---|---|
| touch_core | 36 |
| cpu_checksum | 18 |
| checkSumU32 (GPU) | 256 |

GPU kernel is slow because each thread reads 1024 consecutive words (uncoalesced across threads). Throwaway test kernel; lesson for B6: thread i reads word i, i+N, i+2N.

**Memory, unconfirmed:** after the GPU read the 352 MB tensor, os footprint went from 2.5 MB to 388 MB and resident dropped from 1303 MB to 366 MB. Pages the GPU touches through a no-copy buffer appear to be charged to the process. If so, the A4 conclusion ("mapped weights are free for the budget") only holds for CPU reads, and core weights cost ~1.3 GB of footprint once kernels run over all of them. To confirm: checksum every core tensor and watch footprint.
