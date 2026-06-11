# Generic latency bench for Core AI vision/audio .aimodel exports.
# Measures median single-inference latency per compute-unit preference
# (gpu / neural_engine / cpu) with synthetic inputs derived from the
# function's input descriptors.
#
# Usage: .venv/bin/python _apple_bench_cv.py <path.aimodel> [units...]
import asyncio
import statistics
import sys
import time

import numpy as np
import coreai.runtime as rt

WARMUP = 3
TRIALS = 20


def synth_input(name: str, desc) -> np.ndarray:
    shape = [int(s) for s in desc.shape]
    dt = str(desc.dtype).lower()
    lname = name.lower()
    if "int" in dt:
        np_dt = np.int32 if "32" in dt else np.int64
        if "mask" in lname:
            return np.ones(shape, dtype=np_dt)
        return np.random.randint(0, 1000, size=shape).astype(np_dt)
    np_dt = np.float16 if "16" in dt else np.float32
    return np.random.randn(*shape).astype(np_dt)


async def bench(path: str, unit_name: str) -> tuple[float, float, float]:
    unit = getattr(rt.ComputeUnitKind, unit_name)()
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(unit)
    t0 = time.perf_counter()
    m = await rt.AIModel.load(path, opts)
    fn = m.load_function("main")
    load_s = time.perf_counter() - t0

    d = fn.desc
    feed = {}
    for n in d.input_names:
        feed[n] = rt.NDArray(synth_input(n, d.input_descriptor(n)))

    for _ in range(WARMUP):
        await fn(feed)
    times = []
    for _ in range(TRIALS):
        t = time.perf_counter()
        await fn(feed)
        times.append((time.perf_counter() - t) * 1000)
    return load_s, statistics.median(times), min(times)


async def main():
    path = sys.argv[1]
    units = sys.argv[2:] or ["gpu", "neural_engine"]
    print(f"model: {path}")
    for u in units:
        try:
            load_s, med, best = await bench(path, u)
            print(f"  {u:14s} load {load_s:6.2f}s   median {med:8.2f} ms   best {best:8.2f} ms")
        except Exception as e:
            print(f"  {u:14s} FAILED: {e}")


asyncio.run(main())
