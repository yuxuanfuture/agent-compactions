from __future__ import annotations

from benchmarks.swe_bench import SweBenchAdapter
from benchmarks.tau_bench import TauBenchAdapter


_SWE_BENCH = SweBenchAdapter()
_TAU_BENCH = TauBenchAdapter()

_ADAPTERS = {
    "swe_bench": _SWE_BENCH,
    "swe-bench": _SWE_BENCH,
    "tau_bench": _TAU_BENCH,
    "tau-bench": _TAU_BENCH,
}


def benchmark_names() -> list[str]:
    return sorted(_ADAPTERS)


def get_benchmark(name: str):
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        known = ", ".join(benchmark_names())
        raise SystemExit(f"unknown benchmark {name!r}; known benchmarks: {known}") from exc
