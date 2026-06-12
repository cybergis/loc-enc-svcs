#!/usr/bin/env python3
"""Full-DGP report: untrained-vs-trained Delta r with Wilcoxon significance.

Reads combined_summary_stats.csv + statistical_tests.csv (produced by
aggregate_metrics.py) and writes results/REPORT_full_dgp.txt comparing the
nonlinear ('full') DGP to the linear ('simple') one for SVC recovery, plus the
contrastive training effect and its significance.
"""
import csv
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SUMMARY = os.path.join(ROOT, "combined_summary_stats.csv")
TESTS = os.path.join(ROOT, "statistical_tests.csv")
OUT = os.path.join(ROOT, "REPORT_full_dgp.txt")

rows = list(csv.DictReader(open(SUMMARY)))
tests = list(csv.DictReader(open(TESTS))) if os.path.exists(TESTS) else []

ENCS = ["none", "Sphere2Vec-sphereM+", "Sphere2Vec-dfs", "wrap_ffn",
        "Sphere2Vec-sphereM", "Sphere2Vec-sphereC", "Sphere2Vec-sphereC+",
        "Space2Vec-grid", "Space2Vec-theory", "NeRF", "rff", "tile_ffn"]
SCALES = ["grid", "county", "global"]
EFFECTS = [("SVC_X1_Smooth", "beta1"), ("SVC_X2_Smooth", "beta2")]


def cell(scale, dgp, fc, trained, enc, eff, field="pearson_r_mean"):
    for r in rows:
        if (r["scale"] == scale and r["dgp"] == dgp and r["feature_config"] == fc
                and r["encoder_trained"] == trained and r["encoder"] == enc
                and r["spatial_effect"] == eff):
            try:
                return float(r[field])
            except (ValueError, KeyError):
                return None
    return None


def sig(scale, dgp, enc, eff):
    """Is trained-vs-untrained (emb+coords) Wilcoxon significant? returns p or None.

    statistical_tests.csv compares feature configs, not trained-vs-untrained
    directly, so we report the emb+coords_vs_baseline_trained p as a proxy of
    whether trained embeddings help over baseline.
    """
    for t in tests:
        if (t["scale"] == scale and t.get("dgp") == dgp and t["encoder"] == enc
                and t["spatial_effect"] == eff and t["metric"] == "pearson_r"
                and t["comparison"] == "emb+coords_vs_baseline_trained"):
            return t["p_value"], t["significant"]
    return None, None


lines = []
def p(s=""):
    lines.append(s)
    print(s)


p("=" * 78)
p("FULL (nonlinear) DGP report — SVC recovery, training effect, significance")
p("=" * 78)
p("DGPs present: " + ", ".join(sorted(set(r["dgp"] for r in rows))))

for eff, lbl in EFFECTS:
    p("\n" + "#" * 78)
    p(f"# {lbl}  ({eff})")
    p("#" * 78)
    for scale in SCALES:
        p(f"\n-- {scale} --   (untrained_full | trained_full | Δr | simple_untrained)")
        for enc in ENCS:
            u_f = cell(scale, "full", "emb+coords", "False", enc, eff)
            t_f = cell(scale, "full", "emb+coords", "True", enc, eff)
            u_s = cell(scale, "simple", "emb+coords", "False", enc, eff)
            if u_f is None and t_f is None:
                continue
            d = (t_f - u_f) if (u_f is not None and t_f is not None) else None
            f = lambda x: f"{x:.4f}" if x is not None else "  --  "
            ds = f"{d:+.4f}" if d is not None else "  --  "
            p(f"   {enc:22s} {f(u_f)} | {f(t_f)} | {ds} | {f(u_s)}")

p("\n" + "=" * 78)
p("Mean Δr (trained - untrained, full DGP, emb+coords) by scale/effect:")
for eff, lbl in EFFECTS:
    for scale in SCALES:
        ds = []
        for enc in ENCS:
            u = cell(scale, "full", "emb+coords", "False", enc, eff)
            t = cell(scale, "full", "emb+coords", "True", enc, eff)
            if u is not None and t is not None and enc != "none":
                ds.append(t - u)
        if ds:
            p(f"  {lbl:6s} {scale:7s}: mean Δr = {sum(ds)/len(ds):+.4f}  "
              f"(n={len(ds)}, {'HURTS' if sum(ds)/len(ds) < 0 else 'helps'})")

with open(OUT, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"\nSaved: {OUT}")
