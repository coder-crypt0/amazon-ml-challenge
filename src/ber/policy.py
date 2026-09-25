"""Tune the match-selection policy on saved experiment probabilities (no retraining).

Reads the experiment arrays the pipeline already checkpointed (probabilities.npy,
pairs.npy, y.npy, queries.json), chooses a policy on the TUNE split only, and
reports the untouched HOLDOUT split.  The chosen policy is written to
policy.json for `ber.predict assemble --policy`.

Policy per query, candidates sorted by probability:
  select p >= t, plus the top-ranked candidate when p >= t1 (t1 <= t),
  then drop any selected candidate below r * (best probability of that query).
Chosen globally, then separately per country (unseen countries use global).

Usage (Colab, second CPU runtime while prediction runs):
  python -m ber.policy /content/drive/MyDrive/amazon-ml-challenge/runs/ber_v1/checkpoints/experiment [policy.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

T_GRID = np.round(np.arange(0.30, 0.9501, 0.025), 3)
T1_GRID = np.round(np.arange(0.05, 0.9501, 0.05), 3)
R_GRID = (0.0, 0.2, 0.35, 0.5, 0.65)


def load(experiment):
    experiment = Path(experiment)
    probabilities = np.load(experiment / "probabilities.npy").astype(np.float64)
    pairs = np.load(experiment / "pairs.npy")
    labels = np.load(experiment / "y.npy").astype(bool)
    queries = json.loads((experiment / "queries.json").read_text(encoding="utf-8"))
    return probabilities, pairs, labels, queries


class Evaluator:
    def __init__(self, probabilities, pairs, labels, queries):
        self.p, self.y = probabilities, labels
        self.q = pairs[:, 0].astype(np.int64)
        self.n = len(queries)
        self.true_counts = np.array([len(x["truth"]) for x in queries], dtype=np.float64)
        self.country = np.array([x["country"] for x in queries])
        order = np.lexsort((-self.p, self.q))
        qs = self.q[order]
        starts = np.r_[0, np.flatnonzero(np.diff(qs)) + 1]
        lengths = np.diff(np.r_[starts, len(qs)])
        self.rank = np.empty(len(self.p), dtype=np.int64)
        self.rank[order] = np.arange(len(qs)) - np.repeat(starts, lengths)
        best = np.zeros(self.n)
        best[qs[starts]] = self.p[order][starts]
        self.relative = self.p / np.maximum(best[self.q], 1e-12)
        fit_end, tune_end = int(self.n * .7), int(self.n * .85)
        self.tune = np.arange(fit_end, tune_end)
        self.holdout = np.arange(tune_end, self.n)

    def select(self, t, t1, r):
        chosen = (self.p >= t) | ((self.rank == 0) & (self.p >= t1))
        return chosen & (self.relative >= r) if r else chosen

    def per_query_f05(self, selected):
        """Identical to pipeline.metric's macro-F0.5 per query."""
        counts = np.bincount(self.q[selected], minlength=self.n)
        correct = np.bincount(self.q[selected & self.y], minlength=self.n)
        denominator = .25 * self.true_counts + counts
        return np.divide(1.25 * correct, denominator, out=np.ones(self.n), where=denominator > 0)


def search(ev):
    """Return per-query F vectors' tune means for every policy, globally and per country."""
    countries = sorted(set(ev.country[ev.tune]))
    masks = {c: ev.tune[ev.country[ev.tune] == c] for c in countries}
    results = []
    for t in T_GRID:
        for t1 in [x for x in T1_GRID if x <= t]:
            for r in R_GRID:
                f = ev.per_query_f05(ev.select(t, t1, r))
                results.append({"t": float(t), "t1": float(t1), "r": float(r),
                                "tune": float(f[ev.tune].mean()),
                                "by_country": {c: float(f[m].mean()) for c, m in masks.items()}})
    return results, countries


def apply_policy(ev, policy):
    selected = np.zeros(len(ev.p), dtype=bool)
    pair_country = ev.country[ev.q]
    for country in set(ev.country):
        rule = policy["by_country"].get(country, policy["global"])
        rows = pair_country == country
        selected |= rows & ev.select(rule["t"], rule["t1"], rule["r"])
    return selected


def report(ev, name, selected):
    f = ev.per_query_f05(selected)
    out = {"policy": name, "tune": round(float(f[ev.tune].mean()), 6), "holdout": round(float(f[ev.holdout].mean()), 6)}
    for c in sorted(set(ev.country[ev.holdout])):
        idx = ev.holdout[ev.country[ev.holdout] == c]
        out[f"holdout_{c}"] = round(float(f[idx].mean()), 6)
    zero = (ev.true_counts[ev.holdout] > 0) & (f[ev.holdout] == 0)
    out["holdout_zero_f_queries"] = int(zero.sum())
    return out


def main(experiment, output="policy.json", current_threshold=None):
    ev = Evaluator(*load(experiment))
    calibration = Path(experiment) / "calibration.json"
    if current_threshold is None and calibration.exists():
        current_threshold = json.loads(calibration.read_text())["threshold"]
    rows = []
    if current_threshold is not None:
        rows.append(report(ev, f"CURRENT p>={current_threshold}", ev.p >= current_threshold))
    results, countries = search(ev)
    plain = max((x for x in results if x["t1"] == x["t"] and x["r"] == 0), key=lambda x: x["tune"])
    rows.append(report(ev, f"retuned p>={plain['t']}", ev.select(plain["t"], plain["t1"], 0)))
    best = max(results, key=lambda x: x["tune"])
    glob = {"t": best["t"], "t1": best["t1"], "r": best["r"]}
    rows.append(report(ev, f"global t={glob['t']} top1>={glob['t1']} rel>={glob['r']}", ev.select(**glob)))
    by_country = {}
    for c in countries:
        b = max(results, key=lambda x: x["by_country"][c])
        by_country[c] = {"t": b["t"], "t1": b["t1"], "r": b["r"]}
    policy = {"global": glob, "by_country": by_country}
    rows.append(report(ev, "per-country " + json.dumps(by_country), apply_policy(ev, policy)))
    for row in rows:
        print(json.dumps(row))
    # Keep per-country only if it beats global on TUNE (holdout is never used to choose).
    tune_global = rows[-2]["tune"]
    tune_country = rows[-1]["tune"]
    chosen = policy if tune_country > tune_global else {"global": glob, "by_country": {}}
    Path(output).write_text(json.dumps(chosen, indent=2))
    print("chosen policy (by TUNE):", json.dumps(chosen), "->", output)
    return chosen


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or []))
