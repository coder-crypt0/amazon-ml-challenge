"""Per-country decision profile for spotting domain shift (France has no labels).

Test accuracy for an unlabeled country cannot be measured.  Instead, the same unlabeled
statistics (top-1 probability, share of references with any selected match, selected
links per reference) are computed per country on the test predictions and on the labeled
holdout, where they sit next to the true values.  A country far from the labeled ones is a
warning that the model is under- or over-confident there; it is not a score.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from .predict import select_matches


def _rule(policy, country, threshold):
    if policy is None:
        return {"t": threshold, "t1": threshold, "r": 0.0}
    return policy["by_country"].get(country, policy["global"])


class _Tally:
    def __init__(self):
        self.top1, self.picked, self.candidates = [], [], []

    def add(self, probabilities, chosen):
        self.candidates.append(len(probabilities))
        self.top1.append(float(probabilities.max()) if len(probabilities) else 0.)
        self.picked.append(int(chosen.sum()))

    def summary(self):
        top1, picked = np.asarray(self.top1), np.asarray(self.picked)
        return {"queries": len(top1), "mean_candidates": round(float(np.mean(self.candidates)), 2),
                "top1_p25": round(float(np.percentile(top1, 25)), 4), "top1_median": round(float(np.median(top1)), 4),
                "share_with_selected_match": round(float((picked > 0).mean()), 4),
                "selected_links_per_query": round(float(picked.mean()), 4)}


def holdout_profile(experiment, policy=None):
    """Unlabeled statistics plus the true link counts and macro-F0.5 on the holdout split."""
    experiment = Path(experiment)
    probabilities = np.load(experiment / "probabilities.npy")
    pairs = np.load(experiment / "pairs.npy")
    labels = np.load(experiment / "y.npy").astype(bool)
    queries = json.loads((experiment / "queries.json").read_text(encoding="utf-8"))
    threshold = json.loads((experiment / "calibration.json").read_text())["threshold"]
    n = len(queries)
    order = np.argsort(pairs[:, 0], kind="stable")
    bounds = np.searchsorted(pairs[order, 0], np.arange(n + 1))
    tallies, truth, scores = {}, {}, {}
    for qi in range(int(n * .85), n):
        country = queries[qi]["country"]
        rows = order[bounds[qi]:bounds[qi + 1]]
        p = probabilities[rows]
        chosen = select_matches(p, _rule(policy, country, threshold))
        tallies.setdefault(country, _Tally()).add(p, chosen)
        true = len(queries[qi]["truth"])
        truth.setdefault(country, []).append(true)
        picked, correct = int(chosen.sum()), int((chosen & labels[rows]).sum())
        scores.setdefault(country, []).append(1. if true + picked == 0 else 1.25 * correct / (.25 * true + picked))
    out = {}
    for country, tally in sorted(tallies.items()):
        t = np.asarray(truth[country])
        out[country] = {**tally.summary(), "true_links_per_query": round(float(t.mean()), 4),
                        "true_share_with_match": round(float((t > 0).mean()), 4),
                        "macro_f05": round(float(np.mean(scores[country])), 6)}
    return out


def test_profile(artifacts, data, policy=None):
    artifacts = Path(artifacts)
    threshold = json.loads((artifacts / "experiment/calibration.json").read_text())["threshold"]
    with (Path(data) / "test/test_source1.tsv").open(encoding="utf-8-sig", newline="") as handle:
        countries = {r["entity_id"]: r["country"] for r in csv.DictReader(handle, delimiter="\t")}
    directory = artifacts / "predictions"
    summary = json.loads((directory / "COMPLETE").read_text())
    tallies = {}
    for i in range(summary["batches"]):
        with np.load(directory / f"batch_{i:06d}.npz") as chunk:
            offsets, probabilities = chunk["offsets"], chunk["probabilities"]
            for j, qid in enumerate(chunk["query_ids"]):
                country = countries.get(str(qid), "")
                p = probabilities[offsets[j]:offsets[j + 1]]
                tallies.setdefault(country, _Tally()).add(p, select_matches(p, _rule(policy, country, threshold)))
    return {country: tally.summary() for country, tally in sorted(tallies.items())}


def profile(artifacts, data, policy_path=None):
    policy = json.loads(Path(policy_path).read_text()) if policy_path else None
    result = {"policy": policy, "holdout": holdout_profile(Path(artifacts) / "experiment", policy),
              "test": test_profile(artifacts, data, policy)}
    columns = ("queries", "mean_candidates", "top1_p25", "top1_median", "share_with_selected_match",
               "selected_links_per_query", "true_share_with_match", "true_links_per_query", "macro_f05")
    print(f"{'split':8}{'country':10}" + "".join(f"{c:>26}" for c in columns))
    for split in ("holdout", "test"):
        for country, row in result[split].items():
            print(f"{split:8}{country or '?':10}" + "".join(f"{row.get(c, ''):>26}" for c in columns))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--policy", default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    result = profile(a.artifacts, a.data, a.policy)
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=2))
