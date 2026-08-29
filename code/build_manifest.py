"""Assemble MANIFEST.json with a provenance entry for EVERY shipped data block.

WHY THIS EXISTS
    The released MANIFEST accounted for 4 of the 10 blocks actually shipped in
    data/.  It carried entries for 01_admissibility, 02_z_map,
    03_coefficient_tensors and 05_eq22_validation, and said nothing at all
    about 04_offgrid_error, 06_system_aber, 07_reference_campaign,
    08_landscape_probe, 09_ablation_faithful or 09_ablation_legacy -- so for
    six of ten blocks a reader had no record of what produced the files, when,
    how long it took, or whether the run that produced them finished.  Two
    separate causes:

    (1) generate.py rewrote the whole `blocks` dict from scratch on every run,
        so a block still in progress when the driver was interrupted never got
        an entry at all.  Block 04 is last in the block order and open-ended,
        so it always was; block 06 is the longest-running fixed-size block and
        was interrupted too.
    (2) Blocks 07, 08 and 09 are not produced by generate.py.  They come from
        run_campaign.py and landscape_probe.py, which write their own
        self-describing JSON but were never reflected in the manifest.

    This script reads provenance from the place that actually knows it in each
    case -- generate.py's per-block sidecars in logs/, and the campaign
    artefacts' own embedded metadata -- and never invents a figure.  A block it
    cannot source provenance for is written with status "unknown" and an
    explicit `provenance_gap` note, rather than left out.

Usage:  python code/build_manifest.py [--package DIR] [--check]

    --check  exit non-zero if any shipped block lacks a provenance entry, so
             this can be run as a release gate.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

# --------------------------------------------------------------------------
# What produced each block.  Only blocks NOT produced by generate.py need to be
# described here; generate.py records its own via logs/provenance_<block>.json.
# --------------------------------------------------------------------------
EXTERNAL = {
    "07_reference_campaign": {
        "produced_by": "code/run_campaign.py",
        "command": "python code/run_campaign.py --realizations 300 "
                   "--objective faithful --out ../data/07_reference_campaign",
        "metadata_file": "reference_campaign.json",
        "describes": "A/B reference-implementation campaign: ablation arms x "
                     "guard forms, per-branch surrogate AND post-EGC system "
                     "success rate, plus a brute-force feasibility ceiling.",
    },
    "08_landscape_probe": {
        "produced_by": "code/landscape_probe.py",
        "command": "python code/landscape_probe.py --starts 120 "
                   "--out ../data/08_landscape_probe",
        "metadata_file": "landscape_probe.json",
        "describes": "Objective-landscape diagnostics: finite fraction, local "
                     "minima and argmin of the receding-horizon cost under "
                     "each objective variant.",
    },
    "09_ablation_faithful": {
        "produced_by": "code/run_campaign.py",
        "command": "python code/run_campaign.py --realizations 60 "
                   "--objective faithful --out ../data/09_ablation_faithful",
        "metadata_file": "reference_campaign.json",
        "describes": "Ablation arms scored on the faithful objective "
                     "(steering, manuscript box, strict admissibility, h in "
                     "ABER).  Does NOT reproduce the Table 11 ordering.",
    },
    "09_ablation_legacy": {
        "produced_by": "code/run_campaign.py",
        "command": "python code/run_campaign.py --realizations 60 "
                   "--objective legacy --out ../data/09_ablation_legacy",
        "metadata_file": "reference_campaign.json",
        "describes": "The same ablation on the legacy objective, kept so the "
                     "two objective definitions can be compared directly.",
    },
}

GENERATE_BLOCKS = {
    "01_admissibility": "truncation and round-off behaviour over the full "
                        "(regime, sigma_s, xi, SNR, K) grid -- Table 7",
    "02_z_map": "conditioning parameter z, A_0, ladder order and admissible "
                "flag on the same grid",
    "03_coefficient_tensors": "a_k(alpha,beta), a_k(beta,alpha), D and A_0 on "
                              "the pole-free node grid up to K_max = 20",
    "04_offgrid_error": "deployed float64 kernel (rtodt_fast.pe_series_f64) "
                        "vs 200-digit mpmath reference at random off-grid xi. "
                        "OPEN-ENDED: it samples until the run's deadline, so "
                        "on a full run its status is always 'partial' and the "
                        "record count is whatever the clock allowed -- that is "
                        "the designed behaviour, not a failed run",
    "05_eq22_validation": "eq. (22) convolved series vs two independent "
                          "MN-fold convolution references, with the "
                          "admissibility band recorded per row",
    "06_system_aber": "post-EGC system ABER curves on the corrected "
                      "system_metric cell-mass machinery, two independent "
                      "branch-density constructions per point",
}


def sha256(path: str, blocksize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(blocksize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def count_records(path: str) -> int | None:
    """Rows for a CSV, arrays for an NPZ, top-level results for a JSON."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            with open(path, newline="", encoding="utf-8") as f:
                return max(sum(1 for _ in csv.reader(f)) - 1, 0)
        if ext == ".npz":
            import numpy as np
            with np.load(path, allow_pickle=False) as z:
                return len(z.files)
        if ext == ".json":
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict) and isinstance(d.get("results"), list):
                return len(d["results"])
            return len(d) if isinstance(d, list) else None
    except Exception:
        return None
    return None


def self_declared(block_dir: str, pkg: str) -> dict | None:
    """Provenance a block's own artefacts declare about themselves.

    Last resort, used only when there is no generate.py sidecar and no
    EXTERNAL entry.  Several scripts in this package stamp their output with
    the script that wrote it (`generated_by` or `generator`) and how long it
    took (`elapsed_s`).  Reading that back is SOURCING provenance, not
    inventing it -- the same thing the EXTERNAL branch does for blocks 07-09,
    just discovered rather than hard-coded.

    Two rules keep it honest:
      * a declared producer is accepted only if that script actually exists in
        code/.  A file claiming to come from a script that does not ship is
        not provenance, and is reported as a gap.
      * files in the block that declare nothing are listed by name in
        `files_without_declared_producer`, so "this block has provenance" can
        never quietly mean "one file in it did".
    """
    code_dir = os.path.join(pkg, "code")
    producers: dict[str, list] = {}
    silent, seconds, stamps = [], None, []
    for fn in sorted(os.listdir(block_dir)):
        p = os.path.join(block_dir, fn)
        if not os.path.isfile(p):
            continue
        low = fn.lower()
        if low.endswith(".npz"):
            # an .npz is a zip of named arrays and can carry its own provenance;
            # only the one key is materialised, so the 83-array timing archives
            # stay cheap to scan
            try:
                import numpy as _np
                with _np.load(p, allow_pickle=False) as _z:
                    if "generated_by" not in _z.files:
                        silent.append(fn)
                        continue
                    d = {"generated_by": str(_z["generated_by"])}
                    if "command" in _z.files:
                        d["command"] = str(_z["command"])
                    for _k in ("elapsed_s", "seconds"):
                        if _k in _z.files:
                            try:
                                d[_k] = float(_z[_k])
                            except Exception:
                                pass
            except Exception:
                silent.append(fn)
                continue
        elif low.endswith(".tex"):
            # a generated LaTeX fragment declares itself in a leading comment,
            # which does not disturb what it typesets
            try:
                d = {}
                with open(p, encoding="utf-8", errors="replace") as f:
                    for _line in range(8):
                        _s = f.readline()
                        if not _s:
                            break
                        _s = _s.strip()
                        if _s.startswith("%") and "generated_by:" in _s:
                            d["generated_by"] = _s.split("generated_by:", 1)[1].strip()
                            break
                if not d:
                    silent.append(fn)
                    continue
            except Exception:
                silent.append(fn)
                continue
        elif low.endswith(".json"):
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                silent.append(fn)
                continue
        else:
            silent.append(fn)
            continue
        if not isinstance(d, dict):
            silent.append(fn)
            continue
        who = d.get("generated_by") or d.get("generator")
        if not (isinstance(who, str) and who
                and os.path.exists(os.path.join(code_dir, os.path.basename(who)))):
            silent.append(fn)
            continue
        producers.setdefault(who, []).append(fn)
        for k in ("elapsed_s", "seconds"):
            if isinstance(d.get(k), (int, float)):
                seconds = (seconds or 0.0) + float(d[k])
        if isinstance(d.get("generated_utc"), str):
            stamps.append(d["generated_utc"])
    if not producers:
        return None
    return {
        "produced_by": ", ".join(sorted({"code/%s" % os.path.basename(w)
                                        for w in producers})),
        "seconds": round(seconds, 1) if seconds is not None else None,
        "timing_source": ("summed 'elapsed_s' of the block's own JSON artefacts"
                          if seconds is not None
                          else "not recorded by the producing script"),
        "provenance_source": "self-declared by the artefacts in this block "
                             "('generated_by'/'generator'), cross-checked "
                             "against code/",
        "declared_by_file": {w: v for w, v in sorted(producers.items())},
        "files_without_declared_producer": silent,
        "generated_utc_stamps": sorted(stamps) or None,
    }


def describe_files(block_dir: str) -> list:
    out = []
    for fn in sorted(os.listdir(block_dir)):
        p = os.path.join(block_dir, fn)
        if not os.path.isfile(p):
            continue
        out.append({"name": fn, "bytes": os.path.getsize(p),
                    "records": count_records(p),
                    "modified": iso(os.path.getmtime(p)),
                    "sha256": sha256(p)})
    return out


def build(pkg: str) -> dict:
    data_dir = os.path.join(pkg, "data")
    logs_dir = os.path.join(pkg, "logs")
    blocks = {}

    for name in sorted(os.listdir(data_dir)):
        bdir = os.path.join(data_dir, name)
        if not os.path.isdir(bdir):
            continue
        files = describe_files(bdir)
        entry = {"files": files,
                 "bytes_total": sum(f["bytes"] for f in files)}

        sidecar = os.path.join(logs_dir, "provenance_%s.json" % name)
        if os.path.exists(sidecar):
            # generate.py block, run by this driver: authoritative timing
            with open(sidecar, encoding="utf-8") as f:
                side = json.load(f)
            entry.update({
                "status": side.get("status", "unknown"),
                "produced_by": side.get("produced_by", "generate.py"),
                "command": "python generate.py --only %s" % name,
                "generated": side.get("finished"),
                "seconds": side.get("seconds"),
                "records": side.get("records"),
                "scope": side.get("scope", "full"),
                "describes": GENERATE_BLOCKS.get(name),
            })
        elif name in EXTERNAL:
            spec = EXTERNAL[name]
            meta_path = os.path.join(bdir, spec["metadata_file"])
            secs = None
            extra = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    secs = meta.get("seconds")
                    for k in ("realizations", "objective_mode", "gbar_db",
                              "sigma_s_swept", "starts", "horizon", "sigma_s"):
                        if k in meta:
                            extra[k] = meta[k]
                except Exception:
                    pass
            newest = max((f["modified"] for f in files), default=None)
            entry.update({
                # These artefacts are written in a single pass at the end of
                # their run; the run either produced the file or it did not.
                "status": "complete" if files else "missing",
                "produced_by": spec["produced_by"],
                "command": spec["command"],
                "generated": newest,
                "seconds": secs,
                "records": max((f["records"] or 0 for f in files), default=0),
                "scope": "full",
                "describes": spec["describes"],
                "parameters": extra or None,
                "timing_source": ("embedded 'seconds' field of %s"
                                  % spec["metadata_file"]) if secs is not None
                                 else "not recorded by the producing script",
            })
        else:
            newest = max((f["modified"] for f in files), default=None)
            decl = self_declared(bdir, pkg)
            if decl:
                entry.update({
                    # The artefacts name their own producer, but nothing here
                    # observed the run, so completeness is not knowable and is
                    # not asserted.
                    "status": "unknown",
                    "generated": newest,
                    "records": max((f["records"] or 0 for f in files), default=0),
                    "scope": "unknown",
                    "describes": None,
                })
                entry.update(decl)
                if decl["files_without_declared_producer"]:
                    entry["provenance_gap"] = (
                        "Partial: %d of %d file(s) in this block declare no "
                        "producer (%s). The rest are attributed from their own "
                        "'generated_by'/'generator' field."
                        % (len(decl["files_without_declared_producer"]),
                           len(files),
                           ", ".join(decl["files_without_declared_producer"])))
            else:
                entry.update({
                    "status": "unknown",
                    "produced_by": None,
                    "generated": newest,
                    "seconds": None,
                    "records": max((f["records"] or 0 for f in files), default=0),
                    "provenance_gap": "No sidecar in logs/, no entry in "
                                      "build_manifest.EXTERNAL, and no artefact "
                                      "in the block names its own producer. The "
                                      "files exist but nothing records what "
                                      "produced them.",
                })
        blocks[name] = entry

    return {
        "generated_by": "code/build_manifest.py",
        "built": _dt.datetime.now().isoformat(timespec="seconds"),
        "package_root": os.path.basename(pkg),
        "block_count": len(blocks),
        "blocks_with_provenance": sum(1 for b in blocks.values()
                                      if b.get("produced_by")),
        "blocks_with_provenance_gaps": sorted(
            n for n, b in blocks.items() if b.get("provenance_gap")),
        "provenance_note":
            "Every directory under data/ has an entry. `seconds` is wall-clock "
            "for the run that produced the files: for generate.py blocks it is "
            "measured by the driver and recorded in logs/provenance_<block>.json "
            "as the block finishes; for the campaign blocks it is the "
            "producing script's own embedded figure; for a block that is "
            "neither, it is whatever that block's own artefacts declare, and "
            "`provenance_source` says so. `status` is complete, partial (the "
            "run hit its deadline mid-block and the file holds what was "
            "finished), or unknown (nothing observed the run, so completeness "
            "cannot be asserted). Block 04 is open-ended by design and is "
            "therefore 'partial' on every full run; see its `describes` field. "
            "A block carrying `provenance_gap` is only partly accounted for "
            "and --check fails on it.",
        "blocks": blocks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=PKG)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any block lacks provenance or is "
                         "only partly accounted for")
    a = ap.parse_args()

    man = build(os.path.abspath(a.package))
    path = os.path.join(os.path.abspath(a.package), "MANIFEST.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)

    print("MANIFEST.json: %d blocks, %d with provenance"
          % (man["block_count"], man["blocks_with_provenance"]))
    print("%-26s %-9s %-21s %10s %9s  %s"
          % ("block", "status", "generated", "seconds", "records", "produced_by"))
    print("-" * 118)
    missing, partial = [], []
    for name, b in man["blocks"].items():
        print("%-26s %-9s %-21s %10s %9s  %s"
              % (name, b["status"], b.get("generated") or "-",
                 "%.1f" % b["seconds"] if b.get("seconds") is not None else "-",
                 b.get("records") if b.get("records") is not None else "-",
                 b.get("produced_by") or "-"))
        if not b.get("produced_by"):
            missing.append(name)
        elif b.get("provenance_gap"):
            partial.append(name)
    if missing:
        print("\nBLOCKS WITHOUT PROVENANCE: %s" % ", ".join(missing))
    for name in partial:
        print("\nPARTIAL PROVENANCE  %s: %s"
              % (name, man["blocks"][name]["provenance_gap"]))
    if a.check and (missing or partial):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
