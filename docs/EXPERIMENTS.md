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

**Memory, confirmed (scratch program: wrap, CPU touch all, GPU checksum every core tensor, CPU re-touch, sampled with vm_stat and `footprint -p`):**

| step | resident | footprint | system wired |
|---|---|---|---|
| mmap + wrapNoCopy | 14 MB | 6 MB | baseline |
| CPU touch all pages | 1310 MB | 7 MB | baseline |
| GPU reads all 1291 MB | 15 MB | 36 MB | **+1.26 GB** |
| CPU re-touch all pages | 1311 MB | 1333 MB | unchanged |

Three effects:
1. CPU-only reads of the mapping: resident, not charged. (The A4 conclusion, still true.)
2. First GPU use of the no-copy buffer: the driver **wires the entire 1.3 GB** (vm_stat wired pages +80k x 16 KB). Those pages leave the process's resident/footprint accounting but are real, non-evictable RAM. `task_info` cannot see them; the tracker's `core_weights` entry is the truth.
3. CPU touching pages the GPU has wired reclassifies them as "Dirty mapped file" in `footprint`, so footprint jumps to 1.3 GB. No extra RAM is used (anonymous and wired counts unchanged). This is what the earlier bench run showed (388 MB = CPU re-read of the 352 MB tensor after the GPU had it).

Consequences: the memory budget must count core weights as fully used once any kernel runs; OS footprint undercounts unless the CPU touches the pages; avoid CPU reads of core weights after wrapping (embedding gather should be a GPU kernel); try MAP_SHARED instead of MAP_PRIVATE at some point to see if effect 3 disappears.
