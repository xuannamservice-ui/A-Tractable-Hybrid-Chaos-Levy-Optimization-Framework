import csv

path = "data/05_eq22_validation/eq22_vs_reference_sigma0p2_0p3.csv"
rows = list(csv.DictReader(open(path)))
print("total rows:", len(rows))
ib = [r for r in rows if r["admissible"] == "1" and r["comparison_valid"] == "1"]
print("in-band valid:", len(ib))
print()
print("=== In-band valid rows sorted by |rel%| desc ===")
for r in sorted(ib, key=lambda r: -abs(float(r["rel_diff_percent"]))):
    print(
        "{:>8} s={} xi={} K={:>2} snr={:>2} z={:>6} spread={:>8}%  "
        "eq22={:.3e} ref_quad={:.3e} rel={}%".format(
            r["regime"], r["sigma_s"], r["xi"], r["K"], r["snr_db"], r["z"],
            r["ref_spread_percent"], float(r["eq22"]), float(r["ref_quad"]),
            r["rel_diff_percent"]))
print()
print("=== admissible=1 but comparison_valid=0 ===")
for r in [r for r in rows if r["admissible"] == "1" and r["comparison_valid"] == "0"]:
    print(
        "{:>8} s={} xi={} K={:>2} snr={:>2} z={:>6} spread={:>8}%  "
        "ref_quad={:.3e} ref_log={:.3e}".format(
            r["regime"], r["sigma_s"], r["xi"], r["K"], r["snr_db"], r["z"],
            r["ref_spread_percent"], float(r["ref_quad"]),
            float(r["ref_logdomain"])))
print()
print("=== admissible=0 sample (out-of-band) ===")
for r in [r for r in rows if r["admissible"] == "0"][:8]:
    print(
        "{:>8} s={} xi={} K={:>2} snr={:>2} z={:>6} ladder_K={:>3}  rel={}%".format(
            r["regime"], r["sigma_s"], r["xi"], r["K"], r["snr_db"], r["z"],
            r["ladder_K_at_snr"], r["rel_diff_percent"]))
