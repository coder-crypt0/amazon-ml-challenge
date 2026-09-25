"""v3 training stages: learned transliteration, IDF token retrieval, retrieval-aware features.

Same reservoir sample, 70/15/15 entity-disjoint fit/tune/holdout split, metric, report
layout and Drive checkpointing as ``ber.pipeline``; candidate generation and features
change.  Name-word odds are target statistics, so fit pairs receive out-of-fold values
and tune/holdout pairs use a table learned on fit pairs only.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from .checkpoint import ensure_config, persist_file, restore_file
from .features3 import FEATURE_NAMES, MODEL_NAMES, TokenOdds, prepare, query_features, word_keys
from .model import PairClassifier, train_xgboost
from .pipeline import metric, sample_queries
from .retrieval import TokenIndex
from .store import RecordStore
from .tokens import configure
from .translit import learn

_STORE = _INDEX = None
INDEX_OPTIONS = ("df_max", "k_all", "k_name")
ODDS_FOLDS = 5


def index_options(args):
    return {k: int(getattr(args, k)) for k in INDEX_OPTIONS}


def worker_init(artifact_root, split, translit, options):
    global _STORE, _INDEX
    configure(translit)
    _STORE = RecordStore(Path(artifact_root) / split / "store")
    _INDEX = TokenIndex(Path(artifact_root) / split / "tokens", **options)
    prepared_target.cache_clear()


@lru_cache(maxsize=20000)
def prepared_target(row):
    return prepare(_STORE[row], _INDEX)


def retrieve(query):
    """The exact candidates the model scores: rows, prepared targets, features, one-sided words."""
    rows, score_all, score_name, channel = _INDEX.query(query)
    targets = [prepared_target(int(r)) for r in rows]
    features, extra, missing = query_features(prepare(query, _INDEX), targets, score_all, score_name, channel)
    return rows, targets, features, extra, missing


def candidate_batch(batch):
    pairs, blocks, labels, raw, extra, missing = [], [], [], [], [], []
    for qi, query in batch:
        rows, targets, features, e, m = retrieve(query)
        truth = set(query.get("truth", ()))
        pairs.extend((qi, int(r)) for r in rows)
        blocks.append(features)
        labels.extend(t["entity_id"] in truth for t in targets)
        raw.append((qi, len(rows)))
        extra.extend(e)
        missing.extend(m)
    X = np.vstack(blocks) if blocks else np.empty((0, len(FEATURE_NAMES)), np.float32)
    e_keys, e_counts = word_keys(extra, "e")
    m_keys, m_counts = word_keys(missing, "m")
    return (np.asarray(pairs, np.int32).reshape(-1, 2), X, np.asarray(labels, np.uint8), raw,
            e_keys, e_counts, m_keys, m_counts)


def build_split(data, artifacts, split, workers):
    """Record store plus token index for one split's Source 2/3 targets."""
    root = Path(artifacts) / split
    store = RecordStore.build([Path(data) / split / f"{split}_source{i}.tsv" for i in (2, 3)], root / "store")
    index = TokenIndex.build(store, root / "tokens", Path(artifacts) / "translit.json", workers=workers)
    print(f"Ready: {split}: {len(store):,} targets; {len(index.rows):,} token postings", flush=True)
    store.close()


def create_training_candidates(args):
    destination = Path(args.artifacts) / "experiment"
    destination.mkdir(parents=True, exist_ok=True)
    ensure_config(destination / "candidate_config.json", {"version": 3, "samples": args.samples, "seed": args.seed,
                  "retrieval": index_options(args), "feature_names": list(FEATURE_NAMES)})
    restore_file(destination / "queries.json")
    if (destination / "queries.json").exists():
        queries = json.loads((destination / "queries.json").read_text(encoding="utf-8"))
    else:
        queries = sample_queries(args.data, args.samples, args.seed)
        (destination / "queries.json").write_text(json.dumps(queries, ensure_ascii=False), encoding="utf-8")
        persist_file(destination / "queries.json")
    translit = Path(args.artifacts) / "translit.json"
    # Sampled reference entities (including tune/holdout) never inform the table.
    learn(args.data, translit, exclude=[q["entity_id"] for q in queries], seed=args.seed)
    configure(translit)
    build_split(args.data, args.artifacts, "train", args.workers)
    enumerated = list(enumerate(queries))
    batches = [enumerated[s:s + 64] for s in range(0, len(queries), 64)]
    chunks = destination / "chunks"
    chunks.mkdir(exist_ok=True)
    missing = [(i, b) for i, b in enumerate(batches) if not restore_file(chunks / f"{i:06d}.npz")]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init,
                             initargs=(args.artifacts, "train", str(translit), index_options(args))) as pool:
        for j, ((i, _), result) in enumerate(zip(missing, pool.map(candidate_batch, [b for _, b in missing])), 1):
            pairs, X, y, raw, e_keys, e_counts, m_keys, m_counts = result
            path = chunks / f"{i:06d}.npz"
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, pairs=pairs, X=X, y=y, raw=np.asarray(raw), e_keys=e_keys,
                                e_counts=e_counts, m_keys=m_keys, m_counts=m_counts)
            temporary.replace(path)
            persist_file(path)
            if j % 10 == 0 or j == len(missing):
                print(f"Candidate features: {len(batches)-len(missing)+j}/{len(batches)} checkpoint batches; "
                      f"{time.monotonic()-started:.0f}s", flush=True)
    total = 0
    for i in range(len(batches)):
        with np.load(chunks / f"{i:06d}.npz") as chunk:
            total += len(chunk["y"])
    shapes = {"pairs": ((total, 2), np.int32), "X": ((total, len(FEATURE_NAMES)), np.float32), "y": ((total,), np.uint8)}
    arrays = {k: np.lib.format.open_memmap(destination / f"{k}.npy", mode="w+", dtype=d, shape=s)
              for k, (s, d) in shapes.items()}
    pos, raw_sizes, words = 0, [], {k: [] for k in ("e_keys", "e_counts", "m_keys", "m_counts")}
    for i in range(len(batches)):
        with np.load(chunks / f"{i:06d}.npz") as chunk:
            n = len(chunk["y"])
            for k, a in arrays.items():
                a[pos:pos + n] = chunk[k]
            for k in words:
                words[k].append(chunk[k])
            raw_sizes.extend(chunk["raw"].tolist())
            pos += n
    for k, a in arrays.items():
        a.flush()
        persist_file(destination / f"{k}.npy")
    np.savez(destination / "words.npz", **{k: np.concatenate(v) if v else np.empty(0) for k, v in words.items()})
    persist_file(destination / "words.npz")
    (destination / "retrieval.json").write_text(json.dumps({"version": 3, **index_options(args), "samples": len(queries),
        "seed": args.seed, "candidates_mean": float(np.mean([v for _, v in raw_sizes])) if raw_sizes else 0.,
        "feature_names": list(FEATURE_NAMES), "elapsed_seconds": time.monotonic() - started}, indent=2))
    persist_file(destination / "retrieval.json")


def _word_rows(words, rows):
    """Subset the CSR word arrays to pair ``rows``, keeping row order."""
    out = {}
    rows = np.asarray(rows, np.int64)
    for side in ("e", "m"):
        counts = words[f"{side}_counts"].astype(np.int64)
        starts = np.r_[0, np.cumsum(counts)[:-1]][rows] if len(counts) else np.empty(0, np.int64)
        taken = counts[rows]
        begin = np.r_[0, np.cumsum(taken)[:-1]] if len(taken) else taken
        index = np.repeat(starts - begin, taken) + np.arange(int(taken.sum()))
        out[f"{side}_keys"] = words[f"{side}_keys"][index]
        out[f"{side}_counts"] = taken
    return out


def _odds(words, rows, labels):
    w = _word_rows(words, rows)
    return TokenOdds.fit(w["e_keys"], w["e_counts"], w["m_keys"], w["m_counts"], labels[rows])


def odds_features(words, pairs, labels, fit_end):
    """Out-of-fold odds for fit pairs; fit-only table for tune/holdout pairs."""
    out = np.zeros((len(labels), 5), np.float32)
    fit = np.flatnonzero(pairs[:, 0] < fit_end)
    other = np.flatnonzero(pairs[:, 0] >= fit_end)
    fold = pairs[fit, 0] % ODDS_FOLDS
    for k in range(ODDS_FOLDS):
        inside, outside = fit[fold == k], fit[fold != k]
        if len(inside):
            w = _word_rows(words, inside)
            out[inside] = _odds(words, outside, labels).features(w["e_keys"], w["e_counts"], w["m_keys"], w["m_counts"])
    if len(other):
        w = _word_rows(words, other)
        out[other] = _odds(words, fit, labels).features(w["e_keys"], w["e_counts"], w["m_keys"], w["m_counts"])
    return out


def train_and_evaluate(args):
    destination = Path(args.artifacts) / "experiment"
    queries = json.loads((destination / "queries.json").read_text(encoding="utf-8"))
    pairs = np.load(destination / "pairs.npy")
    labels = np.load(destination / "y.npy")
    with np.load(destination / "words.npz") as data:
        words = {k: data[k] for k in data.files}
    n = len(queries)
    train_end, tune_end = int(n * .7), int(n * .85)
    X = np.hstack([np.load(destination / "X.npy", mmap_mode="r"), odds_features(words, pairs, labels, train_end)])
    masks = {"fit": pairs[:, 0] < train_end}
    ensure_config(destination / "training_config.json", {"version": 3, "seed": args.seed, "trees": args.trees,
        "xgboost_device": args.xgboost_device, "fit_fraction": .7, "tune_fraction": .15, "features": list(MODEL_NAMES)})
    model = PairClassifier(n_estimators=args.trees).fit_classifier(X[masks["fit"]], labels[masks["fit"]], seed=args.seed,
        threads=args.workers, checkpoint=destination / "lgb_progress.txt", feature_names=MODEL_NAMES)
    model.save(destination / "model.txt")
    persist_file(destination / "model.txt")
    lgb_p = model.predict_proba(X)
    xgb_p = None
    if args.xgboost_device != "none":
        booster = train_xgboost(X[masks["fit"]], labels[masks["fit"]], destination / "xgboost.ubj", device=args.xgboost_device,
                                rounds=args.trees, seed=args.seed, threads=args.workers, feature_names=MODEL_NAMES)
        booster.set_param({"device": "cpu", "nthread": args.workers})
        xgb_p = booster.inplace_predict(np.asarray(X))
    tune = np.arange(train_end, tune_end)
    curve = []
    for weight in ([0., .25, .5, .75, 1.] if xgb_p is not None else [0.]):
        p = lgb_p if not weight else (1 - weight) * lgb_p + weight * xgb_p
        for threshold in np.r_[np.arange(.3, .96, .01), .97, .98, .99]:
            curve.append({"threshold": float(threshold), "xgboost_weight": weight,
                          **metric(p, pairs, labels, queries, tune, float(threshold))})
    best = max(curve, key=lambda r: (r["macro_f05"], r["threshold"]))
    threshold, weight = best["threshold"], best["xgboost_weight"]
    p = lgb_p if not weight else (1 - weight) * lgb_p + weight * xgb_p
    np.save(destination / "probabilities.npy", p)
    persist_file(destination / "probabilities.npy")
    report = {"version": 3, "split": {"fit": train_end, "tune": tune_end - train_end, "holdout": n - tune_end},
              "seed": args.seed, "threshold": threshold, "tuning": best, "threshold_curve": curve,
              "holdout": metric(p, pairs, labels, queries, np.arange(tune_end, n), threshold), "holdout_by_country": {},
              "model": "LightGBM + XGBoost" if weight else "LightGBM", "xgboost_weight": weight,
              "model_license": "MIT / Apache-2.0",
              "validation_scope": "Random entity-disjoint queries; complete training target corpus; settings chosen on tuning only"}
    for country in sorted({q["country"] for q in queries}):
        idx = [i for i in range(tune_end, n) if queries[i]["country"] == country]
        if idx:
            report["holdout_by_country"][country] = metric(p, pairs, labels, queries, idx, threshold)
    gain = model.model.feature_importance(importance_type="gain")
    report["feature_importance"] = {k: float(v) for k, v in sorted(zip(MODEL_NAMES, gain), key=lambda z: -z[1])}
    # Test-time table uses every labelled training pair.
    _odds(words, np.arange(len(labels)), labels).save(destination / "odds.npz")
    persist_file(destination / "odds.npz")
    Path("reports").mkdir(exist_ok=True)
    for path in (Path("reports/validation.json"), destination / "validation.json"):
        path.write_text(json.dumps(report, indent=2))
    persist_file(destination / "validation.json")
    (destination / "calibration.json").write_text(json.dumps({"version": 3, "threshold": threshold, "xgboost_weight": weight,
        **index_options(args), "exclusive": False}, indent=2))
    persist_file(destination / "calibration.json")
    print(json.dumps({"threshold": threshold, "tuning": best, "holdout": report["holdout"],
                      "holdout_by_country": report["holdout_by_country"]}, indent=2), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("candidates", "train"))
    p.add_argument("--data", default="student_resource/dataset")
    p.add_argument("--artifacts", default="artifacts/v3")
    p.add_argument("--samples", type=int, default=60000)
    p.add_argument("--df-max", type=int, default=50000)
    p.add_argument("--k-all", type=int, default=50)
    p.add_argument("--k-name", type=int, default=10)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--trees", type=int, default=500)
    p.add_argument("--xgboost-device", choices=("none", "cpu", "cuda"), default="none")
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    create_training_candidates(a) if a.stage == "candidates" else train_and_evaluate(a)
