"""v3 resumable full-test scoring.  Batches keep the ``ber.predict`` layout, so its
``assemble`` (threshold or policy, optional target exclusivity, both TSVs) is reused."""
import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from .checkpoint import ensure_config, file_digest, persist_file, remote_path, restore_directory, restore_file
from .features3 import MODEL_NAMES, TokenOdds, word_keys
from .model import InferenceModel
from .pipeline3 import build_split, retrieve, worker_init
from .predict import assemble
from .store import FIELDS, iter_records
from .tokens import configure

_MODEL = _ODDS = None


def initialize(artifacts, translit, options):
    global _MODEL, _ODDS
    worker_init(artifacts, "test", translit, options)
    _MODEL = InferenceModel(Path(artifacts) / "experiment")
    _ODDS = TokenOdds.load(Path(artifacts) / "experiment" / "odds.npz")


def score_batch(batch):
    rows, offsets, blocks, extra, missing = [], [0], [], [], []
    for query in batch:
        found, _, features, e, m = retrieve(query)
        rows.extend(int(r) for r in found)
        blocks.append(features)
        extra.extend(e)
        missing.extend(m)
        offsets.append(len(rows))
    probabilities = np.empty(0, np.float32)
    if rows:
        e_keys, e_counts = word_keys(extra, "e")
        m_keys, m_counts = word_keys(missing, "m")
        X = np.hstack([np.vstack(blocks), _ODDS.features(e_keys, e_counts, m_keys, m_counts)])
        assert X.shape[1] == len(MODEL_NAMES)
        probabilities = _MODEL.predict_proba(X)
    return dict(query_ids=np.asarray([q["entity_id"] for q in batch]), rows=np.asarray(rows, np.uint32),
                offsets=np.asarray(offsets, np.uint32), probabilities=np.asarray(probabilities, np.float32))


def predict(data, artifacts, workers=4, batch_size=512, reverse=False):
    """Score every Source 1 test record in checkpointed batches.

    ``reverse=True`` is a helper for a second runtime sharing the checkpoint root: it
    scores missing batches from the last one backwards and skips any batch already saved
    by either runtime, so the two meet in the middle.  Only the forward run restores all
    batches and writes COMPLETE.
    """
    artifacts = Path(artifacts)
    model_dir = artifacts / "experiment"
    translit = artifacts / "translit.json"
    for required in (translit, model_dir / "odds.npz", model_dir / "model.txt", model_dir / "calibration.json"):
        if not restore_file(required):
            raise ValueError(f"Missing {required.name}; run the v3 training stages first")
    config = json.loads((model_dir / "calibration.json").read_text())
    if config.get("xgboost_weight") and not restore_file(model_dir / "xgboost.ubj"):
        raise ValueError("Missing xgboost.ubj; run the v3 training stages first")
    options = {k: int(config[k]) for k in ("df_max", "k_all", "k_name")}
    configure(translit)
    if reverse and not restore_directory(artifacts / "test" / "tokens"):
        raise ValueError("Start the helper after the main runtime has built the test token index ('Ready: test')")
    build_split(data, artifacts, "test", workers)
    destination = artifacts / "predictions"
    destination.mkdir(exist_ok=True)
    hashes = {p.name: file_digest(p) for p in (model_dir / "model.txt", model_dir / "calibration.json",
                                                 model_dir / "odds.npz", translit)}
    if config.get("xgboost_weight"):
        hashes["xgboost.ubj"] = file_digest(model_dir / "xgboost.ubj")
    ensure_config(destination / "config.json", {"version": 3, "models": hashes, "batch_size": batch_size,
                  "test_source1_sha256": file_digest(Path(data) / "test/test_source1.tsv")})
    # Same parser as before, so batch membership is identical in both directions.
    records = [tuple(r[k] for k in FIELDS) for r in iter_records([Path(data) / "test/test_source1.tsv"])]
    order = list(enumerate(range(0, len(records), batch_size)))
    if reverse:
        order.reverse()

    def saved(path):
        if not reverse:
            return restore_file(path)
        remote = remote_path(path)
        return path.exists() or (remote is not None and remote.is_file())

    started, jobs, seen, fresh = time.monotonic(), {}, 0, 0
    pending = iter(order)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
                             initargs=(str(artifacts), str(translit), options)) as executor:
        exhausted = False
        while not exhausted or jobs:
            while not exhausted and len(jobs) < 2 * workers:
                item = next(pending, None)
                if item is None:
                    exhausted = True
                    break
                index, start = item
                path = destination / f"batch_{index:06d}.npz"
                size = min(batch_size, len(records) - start)
                if saved(path):
                    seen += size
                    continue
                batch = [dict(zip(FIELDS, r)) for r in records[start:start + size]]
                jobs[executor.submit(score_batch, batch)] = (path, size)
            if jobs:
                done, _ = wait(jobs, return_when=FIRST_COMPLETED)
                for future in done:
                    path, count = jobs.pop(future)
                    temporary = path.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, **future.result())
                    temporary.replace(path)
                    persist_file(path)
                    seen += count
                    fresh += count
                    rate = fresh / max(time.monotonic() - started, 1e-9)
                    left = (len(records) - seen) / max(rate, 1e-9) / 3600
                    print(f"Prediction checkpoints: {seen:,}/{len(records):,} queries; {rate:.1f} q/s; "
                          f"~{left:.1f} h left at this rate{' (helper, upper bound)' if reverse else ''}", flush=True)
    summary = {"queries": len(records), "batches": len(order), "batch_size": batch_size, "scored_here": fresh,
               "seconds": time.monotonic() - started}
    if reverse:
        print(f"Helper finished: scored {fresh:,} queries; the forward run restores them and writes COMPLETE.", flush=True)
        return summary
    (destination / "COMPLETE").write_text(json.dumps(summary))
    persist_file(destination / "COMPLETE")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("score", "assemble"))
    p.add_argument("--data", default="student_resource/dataset")
    p.add_argument("--artifacts", default="artifacts/v3")
    p.add_argument("--output", default="output")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--exclusive", action="store_true", default=None)
    p.add_argument("--policy", default=None)
    p.add_argument("--reverse", action="store_true", help="helper runtime: score missing batches from the end")
    a = p.parse_args()
    result = predict(a.data, a.artifacts, a.workers, a.batch_size, a.reverse) if a.stage == "score" else \
        assemble(a.artifacts, a.output, a.exclusive, a.policy, a.data)
    print(json.dumps(result, indent=2))
