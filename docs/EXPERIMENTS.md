# Experiments

Every optimization tried, with numbers. Failures included.

## 2026-10-11 — A4 baseline: core.bin cold vs warm page-in

`apex bench packed/` maps core.bin (1296.7 MiB, 82,990 pages of 16 KB) and touches one byte per page.

| run | mmap_core | touch_core |
|---|---|---|
| cold (first run after pack) | 0.4 ms | 1609 ms (~800 MB/s) |
| warm (page cache) | 5 ms | 36 ms |

Memory: tracked 1296.7 MB, OS resident 1302.8 MB, OS footprint 2.5 MB. Mapped file pages are resident but not charged to the process. Budget enforcement (C10) must use footprint; resident tells whether the page cache still holds the weights.
