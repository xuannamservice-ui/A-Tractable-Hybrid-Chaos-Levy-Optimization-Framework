"""
platform_spec.py -- Phase 1 of the single-platform re-measurement.

Captures a reproducible record of the machine that every subsequent timing
claim in the manuscript is measured on, and establishes the measurement
methodology those later phases use:

  1.  Full static specification: CPU model, physical/logical core counts,
      cache topology, RAM size and configured speed, OS build, Python build,
      numpy / scipy / mpmath versions, and the BLAS numpy is actually linked
      against (it changes float64 throughput materially).

  2.  Core pinning.  This is a heterogeneous P-core/E-core CPU.  An unpinned
      latency measurement mixes two physically different processors and its
      tail is meaningless.  We pin with processor affinity, VERIFY the pin
      took effect by asking the kernel which processor we are actually
      executing on (GetCurrentProcessorNumber), and we do NOT assume that
      logical processor indices are ordered P-first: we MEASURE which indices
      are fast and which are slow by running one identical fixed workload
      pinned to each logical processor in turn and clustering the results.
      The OS-reported EfficiencyClass is read separately and used only as an
      independent cross-check of the measured clustering.

  3.  The empirical resolution of time.perf_counter_ns on this machine: the
      smallest non-zero increment observed in a tight calling loop.

Writes  data/10_platform/platform.json.

Nothing here is modelled, fitted, smoothed or assumed.  Every number that is
not read directly out of the OS or a library __version__ is produced by a
clock on this machine during this run.  Fields that could not be obtained are
written as null rather than filled in from a datasheet; two vendor datasheet
values (nominal turbo clocks) are carried in a separate `vendor_datasheet`
block that is explicitly labelled as NOT measured.

Run:   python platform_spec.py
Portable: runs on Linux/macOS too (the Windows-only blocks degrade to null),
so a reviewer can produce the equivalent record on their own machine.
"""

# --- BLAS threading must be pinned to 1 BEFORE numpy is imported, otherwise
# --- OpenBLAS spawns a pool sized to the machine and the per-core benchmark
# --- stops measuring one core.
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import ctypes
import json
import platform
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "data" / "10_platform"
OUT_JSON = OUT_DIR / "platform.json"

IS_WINDOWS = sys.platform.startswith("win")


# =====================================================================
# 0.  small helpers
# =====================================================================
def _ps(cmd):
    """Run a PowerShell snippet, return stdout, or None on any failure."""
    if not IS_WINDOWS:
        return None
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True, text=True, timeout=90,
        )
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def _ps_json(cmd):
    out = _ps(cmd + " | ConvertTo-Json -Compress -Depth 4")
    if not out:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


# =====================================================================
# 1.  clock resolution -- measured, not assumed
# =====================================================================
def measure_clock_resolution(n_samples=2_000_000):
    """
    Smallest non-zero increment of time.perf_counter_ns() observed in a tight
    loop.  This is the granularity below which no later timing claim in the
    paper can be trusted.  Also reports the cost of one call to the clock,
    which is the floor on how short an interval can be timed at all.
    """
    pc = time.perf_counter_ns
    deltas = []
    prev = pc()
    for _ in range(n_samples):
        cur = pc()
        d = cur - prev
        if d > 0:
            deltas.append(d)
            prev = cur
    # cost of a back-to-back call pair (includes the loop overhead)
    t0 = pc()
    for _ in range(200_000):
        pc()
    t1 = pc()

    deltas.sort()
    return {
        "min_nonzero_increment_ns": deltas[0] if deltas else None,
        "median_nonzero_increment_ns": statistics.median(deltas) if deltas else None,
        "n_nonzero_increments": len(deltas),
        "n_calls": n_samples,
        "fraction_of_calls_returning_same_value": round(
            1.0 - len(deltas) / n_samples, 6),
        "mean_call_cost_ns": round((t1 - t0) / 200_000, 2),
        "clock": "time.perf_counter_ns",
        "monotonic": time.get_clock_info("perf_counter").monotonic,
        "reported_resolution_s": time.get_clock_info("perf_counter").resolution,
        "implementation": time.get_clock_info("perf_counter").implementation,
    }


# =====================================================================
# 2.  Windows logical-processor topology (EfficiencyClass) -- cross-check only
# =====================================================================
RelationProcessorCore = 0
RelationCache = 2

def win_topology():
    """
    Parse GetLogicalProcessorInformationEx.  Returns, per physical core, the
    kernel's EfficiencyClass (higher = higher-performance core class) and the
    logical processors that belong to it.  Used ONLY to cross-check the
    measured P/E clustering -- it is never substituted for the measurement.
    """
    if not IS_WINDOWS:
        return None
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = k32.GetLogicalProcessorInformationEx
        length = ctypes.c_ulong(0)
        RelationAll = 0xFFFF
        fn(RelationAll, None, ctypes.byref(length))
        buf = (ctypes.c_byte * length.value)()
        if not fn(RelationAll, buf, ctypes.byref(length)):
            return None
        raw = bytes(buf)
    except Exception:
        return None

    cores, caches = [], []
    off = 0
    while off < len(raw):
        rel, size = struct.unpack_from("<II", raw, off)
        if size == 0:
            break
        if rel == RelationProcessorCore:
            flags, eff = struct.unpack_from("<BB", raw, off + 8)
            group_count = struct.unpack_from("<H", raw, off + 30)[0]
            lps = []
            for g in range(max(group_count, 1)):
                mask, grp = struct.unpack_from("<QH", raw, off + 32 + 16 * g)
                for bit in range(64):
                    if mask & (1 << bit):
                        lps.append(grp * 64 + bit)
            cores.append({
                "efficiency_class": eff,
                "smt": bool(flags & 1),
                "logical_processors": sorted(lps),
            })
        elif rel == RelationCache:
            lvl, assoc, line = struct.unpack_from("<BBH", raw, off + 8)
            csize, ctype = struct.unpack_from("<II", raw, off + 12)
            caches.append({"level": lvl, "size_bytes": csize,
                           "line_bytes": line, "assoc": assoc, "type": ctype})
        off += size

    by_class = {}
    for c in cores:
        by_class.setdefault(c["efficiency_class"], []).extend(
            c["logical_processors"])
    cache_summary = {}
    ctype_name = {0: "unified", 1: "instruction", 2: "data", 3: "trace"}
    for c in caches:
        key = f"L{c['level']}_{ctype_name.get(c['type'], c['type'])}"
        cache_summary.setdefault(key, {"instances": 0, "sizes_bytes": set(),
                                       "line_bytes": c["line_bytes"]})
        cache_summary[key]["instances"] += 1
        cache_summary[key]["sizes_bytes"].add(c["size_bytes"])
    for k in cache_summary:
        cache_summary[k]["sizes_bytes"] = sorted(cache_summary[k]["sizes_bytes"])

    return {
        "physical_cores": len(cores),
        "cores": cores,
        "logical_by_efficiency_class": {str(k): sorted(v)
                                        for k, v in sorted(by_class.items())},
        "caches": cache_summary,
        "note": ("EfficiencyClass is the Windows kernel's own label; higher "
                 "value = higher-performance core class. Reported here only "
                 "as an independent cross-check of the measured clustering."),
    }


# =====================================================================
# 3.  affinity: set + VERIFY
# =====================================================================
try:
    import psutil
except Exception:
    psutil = None

if IS_WINDOWS:
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.GetCurrentProcessorNumber.restype = ctypes.c_ulong


def current_processor_number():
    """Which logical processor is this thread executing on RIGHT NOW."""
    if IS_WINDOWS:
        return int(_k32.GetCurrentProcessorNumber())
    if hasattr(os, "sched_getcpu"):
        return os.sched_getcpu()
    return None


def get_affinity():
    if psutil is not None:
        return sorted(psutil.Process().cpu_affinity())
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return None


def set_affinity(cpus):
    """Set process affinity. Returns True if the API call succeeded."""
    if psutil is not None:
        psutil.Process().cpu_affinity(list(cpus))
        return True
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpus))
        return True
    if IS_WINDOWS:  # last-resort ctypes path if psutil is unavailable
        mask = 0
        for c in cpus:
            mask |= (1 << c)
        h = _k32.GetCurrentProcess()
        return bool(_k32.SetProcessAffinityMask(ctypes.c_void_p(h),
                                                ctypes.c_size_t(mask)))
    return False


def verify_pin(cpu, n_probes=2000):
    """
    Verify a pin actually took effect, two independent ways:
      (a) read the affinity mask back from the OS;
      (b) ask the kernel, repeatedly and while doing real work, which logical
          processor we are on.  (a) alone is not proof of execution placement.
    """
    mask = get_affinity()
    seen = {}
    x = 0.0
    for i in range(n_probes):
        c = current_processor_number()
        seen[c] = seen.get(c, 0) + 1
        x += i * 1.000001  # keep the thread runnable between probes
    return {
        "requested_cpu": cpu,
        "affinity_mask_readback": mask,
        "mask_matches_request": mask == [cpu],
        "observed_processor_numbers": {str(k): v for k, v in sorted(seen.items())},
        "all_probes_on_requested_cpu": set(seen) == {cpu},
        "n_probes": n_probes,
        "_sink": x,
    }


# =====================================================================
# 4.  the per-core workload
# =====================================================================
# A fixed, deterministic, compute-bound float64 kernel of the same shape as the
# manuscript's inner loop (exp_timing.py:kernel_exact): a swarm of 30
# candidates against a K=10 truncation, i.e. broadcast (30, 11) power / divide
# / reduce over ~2.6 kB of working set, which stays in L1 on every core class.
# Identical inputs on every core, so the only thing that varies is the core.
_N_CAND, _K = 30, 10
_rng = np.random.default_rng(20260827)
_XI = _rng.uniform(0.30, 0.95, _N_CAND)
_A0 = _rng.uniform(0.20, 0.90, _N_CAND)
_KC1 = _rng.uniform(-1.0, 1.0, _K + 1)
_KC2 = _rng.uniform(-1.0, 1.0, _K + 1)
_CB = _rng.uniform(0.10, 1.0, _K + 1)
_CA = _rng.uniform(0.10, 1.0, _K + 1)
_KARR = np.arange(_K + 1, dtype=np.float64)
_ALPHA, _BETA = 1.2, 1.1


def workload(reps):
    """`reps` evaluations of the kernel. Returns a checksum so it cannot be
    optimised away and so we can prove every core computed the same thing."""
    acc = 0.0
    x2 = _XI * _XI
    for _ in range(reps):
        powB = _A0[:, None] ** (_BETA + _KARR)[None, :]
        powA = _A0[:, None] ** (_ALPHA + _KARR)[None, :]
        t1 = _KC1[None, :] * x2[:, None] / ((x2[:, None] - _BETA - _KARR[None, :]) * powB)
        t2 = _KC2[None, :] * x2[:, None] / ((x2[:, None] - _ALPHA - _KARR[None, :]) * powA)
        tot = (t1 * _CB[None, :]).sum(1) + (t2 * _CA[None, :]).sum(1)
        acc = float(tot.sum())
    return acc


def bench_cpu(cpu, reps=400, repeats=31, warmup=3):
    """
    Pin to `cpu`, verify the pin, then time `repeats` identical batches of
    `reps` kernel evaluations.  Report the MINIMUM batch time: the minimum is
    the estimator least contaminated by preemption and interrupt noise, which
    is what we want when the question is 'how fast is this core'.  The median
    and max are reported too so the reader can see the noise we discarded.
    """
    ok = set_affinity([cpu])
    v = verify_pin(cpu)
    for _ in range(warmup):
        workload(reps)
    times, chk = [], None
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        chk = workload(reps)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    times.sort()
    return {
        "cpu": cpu,
        "affinity_api_ok": bool(ok),
        "pin_verified": v["mask_matches_request"] and v["all_probes_on_requested_cpu"],
        "pin_evidence": {k: v[k] for k in
                         ("affinity_mask_readback", "mask_matches_request",
                          "observed_processor_numbers",
                          "all_probes_on_requested_cpu", "n_probes")},
        "reps_per_batch": reps,
        "n_batches": repeats,
        "min_batch_ns": times[0],
        "median_batch_ns": int(statistics.median(times)),
        "max_batch_ns": times[-1],
        "ns_per_eval_min": round(times[0] / reps, 1),
        "evals_per_second_min": round(reps / (times[0] * 1e-9), 1),
        "checksum": chk,
    }


def cluster_two(values):
    """
    Split a 1-D set of per-core costs into two groups at the largest gap in the
    sorted order (1-D 2-means has the same optimum for a clean bimodal set;
    the largest-gap split is deterministic and needs no iteration).  Also
    reports the separation so the reader can judge whether the split is real.
    """
    idx = sorted(range(len(values)), key=lambda i: values[i])
    s = [values[i] for i in idx]
    gaps = [(s[i + 1] - s[i], i) for i in range(len(s) - 1)]
    gap, at = max(gaps)
    fast_idx = sorted(idx[:at + 1])
    slow_idx = sorted(idx[at + 1:])
    fast = [values[i] for i in fast_idx]
    slow = [values[i] for i in slow_idx]
    within = max(max(fast) - min(fast), max(slow) - min(slow))
    return {
        "fast_group": fast_idx,
        "slow_group": slow_idx,
        "fast_mean_ns_per_eval": round(statistics.mean(fast), 1),
        "slow_mean_ns_per_eval": round(statistics.mean(slow), 1),
        "fast_min_ns_per_eval": round(min(fast), 1),
        "fast_median_ns_per_eval": round(statistics.median(fast), 1),
        "fast_max_ns_per_eval": round(max(fast), 1),
        "slow_min_ns_per_eval": round(min(slow), 1),
        "slow_median_ns_per_eval": round(statistics.median(slow), 1),
        "slow_max_ns_per_eval": round(max(slow), 1),
        "slow_over_fast_ratio_of_medians": round(
            statistics.median(slow) / statistics.median(fast), 4),
        "slow_over_fast_ratio_of_minima": round(min(slow) / min(fast), 4),
        "split_gap_ns_per_eval": round(gap, 1),
        "max_within_group_spread_ns_per_eval": round(within, 1),
        "gap_to_spread_ratio": round(gap / within, 2) if within > 0 else None,
        "slow_over_fast_ratio": round(statistics.mean(slow) / statistics.mean(fast), 4),
    }


# =====================================================================
# 5.  achieved clock, measured (Windows): % Processor Performance under load
# =====================================================================
def measure_achieved_clock(cpu, base_mhz, seconds=3.0):
    """
    Windows exposes '% Processor Performance' per logical processor, defined
    against the nominal (base) frequency.  Load one core and sample it: this
    gives the ACHIEVED clock of that core under our workload, which is the
    clock the paper's timings actually ran at -- not the datasheet turbo bin.
    """
    if not IS_WINDOWS or base_mhz is None:
        return None
    import threading
    stop = threading.Event()

    def spin():
        set_affinity([cpu])
        while not stop.is_set():
            workload(200)

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    time.sleep(0.8)  # let the core ramp
    inst = f"0,{cpu}"
    out = _ps(
        f"$s=(Get-Counter '\\Processor Information({inst})\\% Processor "
        f"Performance' -SampleInterval 1 -MaxSamples 3).CounterSamples "
        f"| ForEach-Object {{ $_.CookedValue }}; $s -join ','")
    stop.set()
    t.join(timeout=5)
    if not out:
        return None
    try:
        vals = [float(v) for v in out.strip().split(",") if v.strip()]
    except Exception:
        return None
    if not vals:
        return None
    pct = max(vals)
    return {
        "cpu": cpu,
        "counter": f"\\Processor Information({inst})\\% Processor Performance",
        "samples_pct_of_base": [round(v, 2) for v in vals],
        "base_mhz": base_mhz,
        "achieved_mhz_max_sample": round(base_mhz * pct / 100.0, 1),
        "achieved_mhz_median_sample": round(
            base_mhz * statistics.median(vals) / 100.0, 1),
        "method": ("one thread pinned to this logical processor running the "
                   "benchmark kernel; counter is relative to nominal/base clock"),
    }


# =====================================================================
# 6.  static specification
# =====================================================================
def blas_info():
    """Which BLAS numpy is actually linked against, and the SIMD level it
    dispatches to.  Both change float64 throughput materially."""
    info = {"numpy_version": np.__version__}
    try:
        cfg = np.__config__._check_pyyaml  # noqa: F841  (probe availability)
    except Exception:
        pass
    try:
        d = np.__config__.CONFIG if hasattr(np.__config__, "CONFIG") else {}
        bd = d.get("Build Dependencies", {})
        blas = bd.get("blas", {})
        lapack = bd.get("lapack", {})
        info["blas_name"] = blas.get("name")
        info["blas_version"] = blas.get("version")
        info["blas_configuration"] = blas.get("openblas configuration")
        info["lapack_name"] = lapack.get("name")
        info["lapack_version"] = lapack.get("version")
        simd = d.get("SIMD Extensions", {})
        info["simd_baseline"] = simd.get("baseline")
        info["simd_found"] = simd.get("found")
        comp = d.get("Compilers", {}).get("c", {})
        info["compiler"] = f"{comp.get('name')} {comp.get('version')}"
    except Exception as e:
        info["error"] = repr(e)
    try:
        info["threading_layer_env"] = {
            v: os.environ.get(v) for v in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    except Exception:
        pass
    return info


def library_versions():
    out = {}
    for mod in ("numpy", "scipy", "mpmath", "matplotlib", "psutil"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = None
    return out


def cpu_static():
    d = {
        "platform_processor": platform.processor(),
        "machine": platform.machine(),
        "logical_processors_os": os.cpu_count(),
    }
    if psutil is not None:
        try:
            d["physical_cores_psutil"] = psutil.cpu_count(logical=False)
            d["logical_cores_psutil"] = psutil.cpu_count(logical=True)
        except Exception:
            pass
    w = _ps_json("Get-CimInstance Win32_Processor | Select-Object Name,"
                 "Description,Manufacturer,NumberOfCores,"
                 "NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,"
                 "L3CacheSize,SocketDesignation,ProcessorId")
    if w:
        if isinstance(w, list):
            w = w[0]
        d["model_name"] = w.get("Name", "").strip()
        d["cpuid_description"] = w.get("Description")
        d["vendor"] = w.get("Manufacturer")
        d["physical_cores_os"] = w.get("NumberOfCores")
        d["logical_processors_cim"] = w.get("NumberOfLogicalProcessors")
        d["base_clock_mhz_nominal"] = w.get("MaxClockSpeed")
        d["l2_total_kb_cim"] = w.get("L2CacheSize")
        d["l3_total_kb_cim"] = w.get("L3CacheSize")
        d["socket"] = w.get("SocketDesignation")
        d["cpuid_signature"] = w.get("ProcessorId")
    d["_base_clock_note"] = (
        "Win32_Processor.MaxClockSpeed reports the NOMINAL/base clock, not the "
        "turbo ceiling. The clock actually achieved under our workload is "
        "measured separately in achieved_clock_measured.")
    return d


def memory_static():
    d = {}
    if psutil is not None:
        try:
            d["total_bytes"] = psutil.virtual_memory().total
            d["total_gib"] = round(psutil.virtual_memory().total / 2**30, 2)
        except Exception:
            pass
    mods = _ps_json("Get-CimInstance Win32_PhysicalMemory | Select-Object "
                    "BankLabel,DeviceLocator,Capacity,Speed,"
                    "ConfiguredClockSpeed,Manufacturer,SMBIOSMemoryType")
    if mods:
        if isinstance(mods, dict):
            mods = [mods]
        smbios = {26: "DDR4", 34: "DDR5", 24: "DDR3", 20: "DDR"}
        d["modules"] = [{
            "slot": m.get("DeviceLocator"),
            "capacity_bytes": m.get("Capacity"),
            "capacity_gib": round(int(m.get("Capacity", 0)) / 2**30, 2),
            "rated_speed_mts": m.get("Speed"),
            "configured_speed_mts": m.get("ConfiguredClockSpeed"),
            "type": smbios.get(m.get("SMBIOSMemoryType"),
                               m.get("SMBIOSMemoryType")),
            "manufacturer": (m.get("Manufacturer") or "").strip(),
        } for m in mods]
        d["n_modules"] = len(d["modules"])
        d["installed_gib"] = round(
            sum(int(m.get("Capacity", 0)) for m in mods) / 2**30, 2)
        speeds = {m.get("ConfiguredClockSpeed") for m in mods}
        d["configured_speed_mts"] = (speeds.pop() if len(speeds) == 1
                                     else sorted(speeds))
    return d


def os_static():
    d = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "platform": platform.platform(),
        "node_hash": None,
    }
    w = _ps_json("Get-CimInstance Win32_OperatingSystem | Select-Object "
                 "Caption,Version,BuildNumber,OSArchitecture")
    if w:
        if isinstance(w, list):
            w = w[0]
        d["caption"] = w.get("Caption")
        d["version_full"] = w.get("Version")
        d["build"] = w.get("BuildNumber")
        d["architecture"] = w.get("OSArchitecture")
    r = _ps_json("Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\"
                 "CurrentVersion' | Select-Object DisplayVersion,CurrentBuild,UBR")
    if r:
        if isinstance(r, list):
            r = r[0]
        d["display_version"] = r.get("DisplayVersion")
        d["build_ubr"] = f"{r.get('CurrentBuild')}.{r.get('UBR')}"
    b = _ps_json("Get-CimInstance Win32_BaseBoard | Select-Object "
                 "Manufacturer,Product")
    if b:
        if isinstance(b, list):
            b = b[0]
        d["motherboard"] = f"{(b.get('Manufacturer') or '').strip()} " \
                           f"{(b.get('Product') or '').strip()}".strip()
    g = _ps_json("Get-CimInstance Win32_VideoController | Select-Object "
                 "Name,DriverVersion")
    if g:
        if isinstance(g, dict):
            g = [g]
        d["gpus"] = [{"name": x.get("Name"), "driver": x.get("DriverVersion")}
                     for x in g]
    return d


def python_static():
    return {
        "version": sys.version,
        "version_info": list(sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "executable": sys.executable,
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
        "64bit": sys.maxsize > 2**32,
    }


def process_static():
    d = {"pid": os.getpid()}
    if psutil is not None:
        try:
            p = psutil.Process()
            d["priority_class"] = str(p.nice())
            d["priority_class_note"] = (
                "NORMAL_PRIORITY_CLASS unless stated otherwise. Windows "
                "priority classes are the nearest analogue to Linux "
                "SCHED_OTHER/chrt, but they are NOT the same mechanism and "
                "must not be presented as a reproduction of them.")
        except Exception:
            pass
    return d


# =====================================================================
# main
# =====================================================================
def main():
    t_start = time.time()
    print("=" * 72)
    print("platform_spec.py -- measuring this machine")
    print("=" * 72)

    original_affinity = get_affinity()
    print(f"[i] original process affinity: {original_affinity}")
    print(f"[i] psutil available: {psutil is not None}")

    # ---- 1. clock resolution -------------------------------------------
    print("\n[1/5] measuring time.perf_counter_ns resolution ...")
    clock = measure_clock_resolution()
    print(f"      smallest non-zero increment : {clock['min_nonzero_increment_ns']} ns")
    print(f"      median non-zero increment   : {clock['median_nonzero_increment_ns']} ns")
    print(f"      calls returning same value  : "
          f"{clock['fraction_of_calls_returning_same_value']*100:.2f} %")
    print(f"      mean cost of one call       : {clock['mean_call_cost_ns']} ns")

    # ---- 2. static spec -------------------------------------------------
    print("\n[2/5] reading static specification ...")
    cpu = cpu_static()
    mem = memory_static()
    osd = os_static()
    topo = win_topology()
    print(f"      cpu   : {cpu.get('model_name')}")
    print(f"      cores : {cpu.get('physical_cores_os')} physical / "
          f"{cpu.get('logical_processors_os')} logical")
    print(f"      ram   : {mem.get('installed_gib')} GiB @ "
          f"{mem.get('configured_speed_mts')} MT/s")
    if topo:
        print(f"      kernel EfficiencyClass groups: "
              f"{topo['logical_by_efficiency_class']}")

    # ---- 3. per-logical-processor benchmark -----------------------------
    n_lp = os.cpu_count()
    n_passes = 3
    print(f"\n[3/5] benchmarking each of {n_lp} logical processors "
          f"(pinned + verified), {n_passes} independent passes ...")
    # Background interference on this machine is bursty: a single contiguous
    # measurement window can catch one core mid-interruption and inflate its
    # cost by 70%, which is enough to misclassify it. Interference can only
    # ever make a core look SLOWER than it is, so the minimum across several
    # passes separated in time is a strictly better estimator of the intrinsic
    # per-core cost than a longer single window. Every pass is retained in the
    # JSON so the reader can see the dispersion that was reduced away.
    passes = []
    for p in range(n_passes):
        row = [bench_cpu(c) for c in range(n_lp)]
        passes.append(row)
        print(f"      pass {p+1}/{n_passes} done "
              f"(min ns/eval {min(r['ns_per_eval_min'] for r in row):.0f} .. "
              f"{max(r['ns_per_eval_min'] for r in row):.0f})")

    set_affinity(original_affinity)

    # elementwise best-of-passes
    per_cpu = []
    for c in range(n_lp):
        cand = [passes[p][c] for p in range(n_passes)]
        best = min(cand, key=lambda r: r["ns_per_eval_min"])
        best = dict(best)
        best["ns_per_eval_by_pass"] = [r["ns_per_eval_min"] for r in cand]
        best["pass_spread_pct"] = round(
            100.0 * (max(best["ns_per_eval_by_pass"])
                     - min(best["ns_per_eval_by_pass"]))
            / min(best["ns_per_eval_by_pass"]), 1)
        best["pin_verified_all_passes"] = all(r["pin_verified"] for r in cand)
        per_cpu.append(best)
        flag = "OK " if best["pin_verified_all_passes"] else "!! "
        print(f"      {flag}cpu {c:>2}: {best['ns_per_eval_min']:>8.1f} ns/eval "
              f"(best of {best['ns_per_eval_by_pass']}, "
              f"spread {best['pass_spread_pct']}%)")

    costs = [r["ns_per_eval_min"] for r in per_cpu]
    cl = cluster_two(costs)
    cl["n_passes"] = n_passes
    cl["clustering_clean"] = (cl["gap_to_spread_ratio"] is not None
                              and cl["gap_to_spread_ratio"] >= 5.0)
    if not cl["clustering_clean"]:
        print("      !! WARNING: the two groups are not cleanly separated "
              "(gap/spread < 5). Treat the P/E assignment as unconfirmed.")
    fast, slow = cl["fast_group"], cl["slow_group"]
    print(f"\n      measured FAST group (P-cores): {fast}")
    print(f"      measured SLOW group (E-cores): {slow}")
    print(f"      slow/fast cost ratio         : {cl['slow_over_fast_ratio']}")
    print(f"      split gap / within spread    : {cl['gap_to_spread_ratio']}")

    # cross-check against the kernel's own EfficiencyClass labels
    crosscheck = None
    if topo:
        classes = topo["logical_by_efficiency_class"]
        hi = classes.get(str(max(int(k) for k in classes)), [])
        lo = sorted(set(range(n_lp)) - set(hi))
        crosscheck = {
            "kernel_high_efficiency_class_lps": hi,
            "kernel_low_efficiency_class_lps": lo,
            "agrees_with_measurement": (sorted(hi) == sorted(fast)
                                        and sorted(lo) == sorted(slow)),
        }
        print(f"      kernel EfficiencyClass agrees with measurement: "
              f"{crosscheck['agrees_with_measurement']}")

    all_pinned = all(r["pin_verified_all_passes"] for r in per_cpu)
    print(f"      every pin verified by GetCurrentProcessorNumber: {all_pinned}")

    # ---- 3b. clock resolution / call cost, now PINNED --------------------
    # The step-1 measurement ran unpinned, so its call cost mixes core classes
    # and varies run to run. Repeat it pinned to one representative core of
    # each class: the tick granularity is a property of the timer source and
    # should be identical, while the per-call cost is a property of the core
    # and is the instrumentation floor for every later phase.
    print("\n[3b/5] re-measuring the clock pinned to one core of each class ...")
    clock_pinned = {}
    for label, group in (("p_core", fast), ("e_core", slow)):
        if not group:
            continue
        c = group[0]
        set_affinity([c])
        v = verify_pin(c, n_probes=500)
        r = measure_clock_resolution(n_samples=1_000_000)
        r["cpu"] = c
        r["pin_verified"] = (v["mask_matches_request"]
                             and v["all_probes_on_requested_cpu"])
        clock_pinned[label] = r
        print(f"      {label} cpu{c}: min increment "
              f"{r['min_nonzero_increment_ns']} ns, "
              f"call cost {r['mean_call_cost_ns']} ns, "
              f"pin_verified={r['pin_verified']}")
    set_affinity(original_affinity)

    # ---- 4. achieved clock under load -----------------------------------
    print("\n[4/5] measuring achieved clock under load ...")
    base = cpu.get("base_clock_mhz_nominal")
    ach = {}
    if fast:
        a = measure_achieved_clock(fast[0], base)
        if a:
            ach["fast_core_representative"] = a
            print(f"      P-core cpu{fast[0]}: {a['achieved_mhz_max_sample']} MHz "
                  f"(max sample)")
    if slow:
        a = measure_achieved_clock(slow[0], base)
        if a:
            ach["slow_core_representative"] = a
            print(f"      E-core cpu{slow[0]}: {a['achieved_mhz_max_sample']} MHz "
                  f"(max sample)")
    set_affinity(original_affinity)

    # ---- 5. assemble + write --------------------------------------------
    checksums = {r["checksum"] for r in per_cpu}
    record = {
        "schema": "platform_spec/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "platform_spec.py",
        "elapsed_s": round(time.time() - t_start, 1),
        "provenance": (
            "Every quantity below is either read directly from the operating "
            "system / a library __version__, or measured on this machine "
            "during this run with time.perf_counter_ns. Nothing is modelled, "
            "fitted, smoothed, or copied from a datasheet, except the clearly "
            "labelled vendor_datasheet block."),
        "cpu": cpu,
        "memory": mem,
        "os": osd,
        "python": python_static(),
        "libraries": library_versions(),
        "blas": blas_info(),
        "process": process_static(),
        "clock_resolution": clock,
        "clock_resolution_pinned": {
            "note": ("Same measurement as clock_resolution, but pinned to one "
                     "representative core of each measured class. The tick "
                     "granularity is a property of the timer source and is "
                     "identical on both; the per-call cost is a property of the "
                     "core and sets the instrumentation floor for later phases."),
            **clock_pinned,
        },
        "topology_kernel_reported": topo,
        "core_pinning": {
            "mechanism": ("Windows processor affinity: psutil.Process()."
                          "cpu_affinity([i]), which calls SetProcessAffinityMask. "
                          "Fallback path in this script uses SetProcessAffinityMask "
                          "via ctypes directly if psutil is absent."),
            "verification": ("Two independent checks per pin: (a) the affinity "
                             "mask is read back from the OS and compared to the "
                             "request; (b) kernel32.GetCurrentProcessorNumber() is "
                             "sampled 2000x while the thread is runnable, and every "
                             "sample must equal the requested logical processor."),
            "all_pins_verified": all_pinned,
            "note_not_linux": ("The manuscript's SCHED_OTHER / chrt / isolcpus arms "
                               "are Linux mechanisms and do not exist on Windows. "
                               "Processor affinity plus Windows priority classes are "
                               "the nearest analogue and are NOT a reproduction of "
                               "them."),
        },
        "per_logical_processor_benchmark": {
            "workload": ("fixed deterministic float64 kernel of the same shape as "
                         "the manuscript inner loop (30-candidate swarm, K=10 "
                         "truncation, broadcast (30,11) pow/divide/reduce, ~2.6 kB "
                         "working set, L1-resident); identical inputs on every core"),
            "estimator": ("3 independent passes over all logical processors; each "
                          "pass times 31 batches of 400 evaluations after 3 warm-up "
                          "batches and reports the batch minimum; the per-core value "
                          "is the minimum over the 3 passes. Interference can only "
                          "inflate a measured cost, never deflate it, so the minimum "
                          "is the least-contaminated estimator of the intrinsic core "
                          "cost. Per-pass values and the median/max of the retained "
                          "pass are kept so nothing is hidden."),
            "blas_threads": 1,
            "identical_checksum_on_all_cores": len(checksums) == 1,
            "checksum": sorted(checksums)[0] if len(checksums) == 1 else sorted(checksums),
            "results": per_cpu,
        },
        "pcore_ecore_clustering": {
            **cl,
            "method": ("largest-gap split of the sorted per-logical-processor "
                       "ns/eval costs; no assumption is made about index ordering"),
            "p_cores_logical": fast,
            "e_cores_logical": slow,
            "kernel_crosscheck": crosscheck,
        },
        "achieved_clock_measured": ach,
        "vendor_datasheet": {
            "_warning": "NOT MEASURED. Vendor-published figures, for reader "
                        "orientation only. Do not cite as a measurement.",
            "model": "Intel Core i5-14600KF (Raptor Lake Refresh, LGA1700)",
            "p_core_base_ghz": 3.5,
            "p_core_max_turbo_ghz": 5.3,
            "e_core_base_ghz": 2.6,
            "e_core_max_turbo_ghz": 4.0,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n[5/5] wrote {OUT_JSON}")
    print(f"      ({OUT_JSON.stat().st_size} bytes)")
    return record


if __name__ == "__main__":
    main()
