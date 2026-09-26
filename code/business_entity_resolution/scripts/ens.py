"""Ensemble finished ber5 runs: average per-pair probabilities, tune the threshold on aligned validation
pairs (runs that saved them), then exclusivity + threshold on test.

python ens.py --data DATASET_DIR --out OUT_DIR --runs WORK_DIR [WORK_DIR ...] [--validator PATH]

Each WORK_DIR is a ber5 --work directory (or a folder fetched with aws_ber5.py fetch). All runs must come
from ber5 versions that group records by country (cfg has "wide").
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd


def excl(pt, p):
    order = np.lexsort((-p, pt))
    first = np.ones(len(order), bool)
    first[1:] = pt[order][1:] != pt[order][:-1]
    keep = np.zeros(len(pt), bool)
    keep[order[first]] = True
    return keep


def f05(qi, y, pred, nt):
    m = len(nt)
    npred = np.bincount(qi, weights=pred, minlength=m)
    tp = np.bincount(qi, weights=pred & y, minlength=m)
    with np.errstate(divide="ignore", invalid="ignore"):
        P, R = tp / npred, tp / nt
        f = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
    return float(np.where(nt == 0, (npred == 0).astype(float), f).mean())


def average(key_list, p_list):
    uk, inv = np.unique(np.concatenate(key_list), return_inverse=True)
    s = np.bincount(inv, weights=np.concatenate(p_list), minlength=len(uk))
    c = np.bincount(inv, minlength=len(uk))
    return uk, (s / c).astype(np.float32), c


ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--validator", default=None)
args = ap.parse_args()
runs, data = args.runs, args.data
print("runs:", runs, flush=True)
frames = [pd.read_csv(os.path.join(data, "test", f"test_source{s}.tsv"), sep="\t", dtype=str, keep_default_na=False,
                      na_filter=False, quoting=csv.QUOTE_NONE) for s in (1, 2, 3)]
# same record order as ber5.load_split (rows grouped by country inside each source)
frames = [f.sort_values("country", kind="stable", ignore_index=True) for f in frames]
ids = np.concatenate([f["entity_id"].to_numpy(object) for f in frames])
n1 = len(frames[0])
del frames

tkeys, tps, taus, vkeys, vps, vy, vfold, vs1 = [], [], [], [], [], [], [], None
for r in runs:
    meta = json.load(open(os.path.join(r, "model", "meta.json")))
    if "wide" not in meta.get("cfg", {}):  # older code: records not grouped by country, indices incompatible
        print("skip (older record order):", r, flush=True)
        continue
    use2 = meta.get("stage2") and os.path.exists(os.path.join(r, "test_p2.npy"))
    tag = "2" if use2 else "1"
    tau = meta["tau2"] if use2 else meta["tau"]
    rep = meta.get("report2") if use2 else meta.get("report", {})
    print(r, "stage", tag, "tau", tau, "val", (rep or {}).get("val_f05"), flush=True)
    pq, pt = np.load(os.path.join(r, "test_pq.npy")), np.load(os.path.join(r, "test_pt.npy"))
    tkeys.append((pq.astype(np.int64) << 32) | pt.astype(np.int64))
    tps.append(np.load(os.path.join(r, f"test_p{tag}.npy")).astype(np.float64))
    taus.append(tau)
    vk = os.path.join(r, "model", "val_keys.npy")
    if os.path.exists(vk):
        vkeys.append(np.load(vk))
        vps.append(np.load(os.path.join(r, "model", f"val_p{tag}.npy")).astype(np.float64))
        vy.append(np.load(os.path.join(r, "model", "val_y.npy")))
        vfold.append(np.load(os.path.join(r, "model", "val_fold.npy")))
        vs1 = np.load(os.path.join(r, "model", "val_s1.npy"))

tau = float(np.mean(taus))
if vkeys:
    uk, p, _ = average(vkeys, vps)
    y = np.zeros(len(uk), bool)
    fold = np.full(len(uk), -1, np.int8)
    for k, yy, ff in zip(vkeys, vy, vfold):
        pos = np.searchsorted(uk, k)
        y[pos] |= yy
        fold[pos] = ff
    vq, vt = uk >> 32, uk & 0xFFFFFFFF
    s1, s1fold, s1nt = vs1
    res = {}
    for name, fs in (("oof", (0, 1)), ("val", (2,))):
        sel_s1 = np.isin(s1fold, fs)
        dense = np.full(int(s1.max()) + 1, -1, np.int64)
        dense[s1[sel_s1]] = np.arange(int(sel_s1.sum()))
        m = np.isin(fold, fs)
        res[name] = (dense[vq[m]], y[m], p[m], excl(vt[m], p[m]), s1nt[sel_s1].astype(float))
    qi, yy, pp, kx, nt = res["oof"]
    sweep = {round(float(t), 2): f05(qi, yy, kx & (pp >= t), nt) for t in np.arange(0.4, 0.96, 0.01)}
    tau = max(sweep, key=sweep.get)
    qi, yy, pp, kx, nt = res["val"]
    print(f"ENSEMBLE of {len(vkeys)} runs with validation: tau {tau} oof {sweep[tau]:.5f} "
          f"val {f05(qi, yy, kx & (pp >= tau), nt):.5f}", flush=True)
print(f"test tau {tau:.3f}", flush=True)

uk, p, c = average(tkeys, tps)
pq = (uk >> 32).astype(np.int64)
pt = (uk & 0xFFFFFFFF).astype(np.int64)
print(f"test pairs {len(uk):,}; in all runs {(c == len(runs)).mean():.4f}", flush=True)
best = excl(pt, p)
out = args.out
os.makedirs(out, exist_ok=True)
starts = np.zeros(n1 + 1, np.int64)
starts[1:] = np.cumsum(np.bincount(pq, minlength=n1))
tid = ids[pt]
for t in sorted({round(tau + d, 2) for d in (-0.05, 0.0, 0.05, 0.1)}):
    keep = best & (p >= t)
    name = "matching_results.tsv" if abs(t - round(tau, 2)) < 1e-9 else f"matching_results_t{t:.2f}.tsv"
    with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for q in range(n1):
            a, b = starts[q], starts[q + 1]
            fm.write(ids[q] + "\t" + ",".join(tid[a:b][keep[a:b]]) + "\n")
    npred = np.bincount(pq, weights=keep, minlength=n1)
    print(f"{name}: tau {t}, matches {int(keep.sum()):,}, empty {(npred == 0).mean():.4f}", flush=True)
with open(os.path.join(out, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="\n") as fc:
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    for q in range(n1):
        a, b = starts[q], starts[q + 1]
        fc.write(ids[q] + "\t" + ",".join(tid[a:b]) + "\n")
if args.validator:
    r = subprocess.run([sys.executable, args.validator, "--matching", os.path.join(out, "matching_results.tsv"),
                        "--candidate", os.path.join(out, "candidate_pairs.tsv"), "--test-dir",
                        os.path.join(data, "test"), "--check-ids"], capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-1000:], flush=True)
