# Gpu-Atomic-Ops-Lock-Free-Histogram

256 bin grayscale histogram benchmark comparing three GPU atomic strategies:

1. **Global atomics**: all threads hit global memory directly
2. **Shared mem atomics**: per-block histogram in shared memory, then merge
3. **Warp-aggregated atomics**: pre-aggregate with `__match_any_sync` before touching shared mem

## Build & Run

```bash
./run_benchmark.sh
```

Or manually:

```bash
nvcc -O3 -arch=native -allow-unsupported-compiler main.cu -o hist_bench
./hist_bench
```
![benchmark](https://github.com/user-attachments/assets/5a2d02bd-45fa-4c02-983c-4074c502f445)


| Strategy | Time (ms) | Speedup |
|---|---|---|
| Global Atomics | 99.97 | 1x |
| Shared Memory | 14.86 | 6.7x |
| Warp Aggregated | 14.86 | 6.7x |

Tested on ~256 MiB of 8-bit image data.
