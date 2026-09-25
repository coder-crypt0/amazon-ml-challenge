"""IDF-scored token retrieval over a sorted, memory-mapped CSR posting index.

A candidate's score is the summed IDF of the tokens it shares with the query, so rare
shared evidence dominates and no single key has to match exactly.  Tokens more frequent
than ``df_max`` still supply IDF to the features but do not generate candidates.  Two
channels: the top ``k_all`` by all tokens, plus the top ``k_name`` by name tokens only,
which recovers targets whose address is missing or rewritten.

Postings are packed as ``(40-bit token hash << 24) | row`` and sorted in place, which
bounds the build to ~8 bytes per posting plus the output arrays.
"""
from array import array
from pathlib import Path
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from .blocking import key_hash
from .checkpoint import restore_directory, persist_directory, restore_file, persist_file, remote_path
from .tokens import configure, record_tokens

VERSION = 1
PART_ROWS = 250000
ROW_BITS = 24
ROW_MASK = (1 << ROW_BITS) - 1
_NAME_KINDS = ("n", "c")


def token_keys(tokens):
    """40-bit keys of token strings; collisions are negligible at this vocabulary size."""
    return np.fromiter((key_hash(t) >> ROW_BITS for t in tokens), dtype=np.uint64, count=len(tokens))


def _make_part(args):
    store_path, start, stop, part_path, translit = args
    marker = Path(part_path).with_suffix(".json")
    if marker.exists() and Path(part_path).exists():
        count = json.loads(marker.read_text())["count"]
        if Path(part_path).stat().st_size == count * 8:
            return part_path, count
    from .store import RecordStore
    configure(translit)
    store = RecordStore(store_path)
    packed = array("Q")
    for row in range(start, stop):
        packed.extend(((key_hash(t) >> ROW_BITS) << ROW_BITS) | row for t in record_tokens(store[row]))
    store.close()
    np.frombuffer(packed, dtype=np.uint64).tofile(part_path)
    marker.write_text(json.dumps({"count": len(packed)}))
    return part_path, len(packed)


class TokenIndex:
    def __init__(self, path, df_max=50000, k_all=50, k_name=10):
        self.path = Path(path)
        if not (self.path / "COMPLETE").exists():
            raise ValueError(f"Incomplete token index: {path}")
        meta = json.loads((self.path / "metadata.json").read_text())
        if meta["version"] != VERSION:
            raise ValueError("Token index version changed; rebuild in a new directory")
        self.keys = np.load(self.path / "keys.npy", mmap_mode="r")
        self.offsets = np.load(self.path / "offsets.npy", mmap_mode="r")
        self.rows = np.load(self.path / "rows.npy", mmap_mode="r")
        self.n_records = max(1, int(meta["rows"]))
        self.df_max, self.k_all, self.k_name = int(df_max), int(k_all), int(k_name)

    @classmethod
    def build(cls, store, path, translit, workers=4, **options):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        restore_directory(path)
        if (path / "COMPLETE").exists():
            return cls(path, **options)
        if len(store) > ROW_MASK:
            raise ValueError(f"TokenIndex supports at most {ROW_MASK:,} target rows")
        started = time.monotonic()
        jobs = [(str(store.path), s, min(s + PART_ROWS, len(store)), str(path / f"part_{s:09d}.bin"), str(translit))
                for s in range(0, len(store), PART_ROWS)]
        for job in jobs:
            if restore_file(Path(job[3]).with_suffix(".json")):
                restore_file(job[3])
        parts = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(_make_part, jobs), 1):
                parts.append(result)
                persist_file(result[0])
                persist_file(Path(result[0]).with_suffix(".json"))
                print(f"Token postings: {i}/{len(jobs)} chunks, {time.monotonic()-started:.0f}s", flush=True)
        total = sum(n for _, n in parts)
        packed = np.empty(total, dtype=np.uint64)
        pos = 0
        for part, n in parts:
            packed[pos:pos + n] = np.fromfile(part, dtype=np.uint64)
            pos += n
        print(f"Sorting {total:,} token postings for {len(store):,} records", flush=True)
        packed.sort()
        rows = np.lib.format.open_memmap(path / "rows.npy", mode="w+", dtype="<u4", shape=(total,))
        keys, starts, previous = [], [], None
        step = 8000000
        for s in range(0, total, step):
            block = packed[s:s + step]
            rows[s:s + len(block)] = (block & np.uint64(ROW_MASK)).astype(np.uint32)
            k = block >> np.uint64(ROW_BITS)
            boundary = np.r_[True, k[1:] != k[:-1]] if previous is None else np.r_[k[0] != previous, k[1:] != k[:-1]]
            keys.append(k[boundary])
            starts.append(np.flatnonzero(boundary) + s)
            previous = k[-1]
        rows.flush()
        del rows, packed
        keys = np.concatenate(keys) if keys else np.empty(0, np.uint64)
        starts = np.concatenate(starts) if starts else np.empty(0, np.int64)
        np.save(path / "keys.npy", keys.astype("<u8"))
        np.save(path / "offsets.npy", np.r_[starts, total].astype("<u8"))
        for part, _ in parts:
            Path(part).unlink()
            Path(part).with_suffix(".json").unlink(missing_ok=True)
        (path / "metadata.json").write_text(json.dumps({"version": VERSION, "rows": len(store), "postings": int(total),
                                                        "keys": int(len(keys)), "seconds": time.monotonic() - started}, indent=2))
        (path / "COMPLETE").write_text("complete\n")
        persist_directory(path)
        for part, _ in parts:
            for obsolete in (Path(part), Path(part).with_suffix(".json")):
                remote = remote_path(obsolete)
                if remote is not None:
                    remote.unlink(missing_ok=True)
        return cls(path, **options)

    def document_frequency(self, keys):
        """Posting counts and CSR positions for 40-bit ``keys`` (0 when absent)."""
        keys = np.asarray(keys, dtype=np.uint64)
        df = np.zeros(len(keys), dtype=np.int64)
        if not len(self.keys) or not len(keys):
            return df, np.zeros(len(keys), dtype=np.int64)
        pos = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        found = np.asarray(self.keys[pos]) == keys
        df[found] = (np.asarray(self.offsets[pos[found] + 1]) - np.asarray(self.offsets[pos[found]])).astype(np.int64)
        return df, pos

    def idf(self, df):
        return np.log(self.n_records / (np.asarray(df, dtype=np.float64) + 1.0))

    def query(self, record):
        """Return (rows, all-token score, name-token score, channel) with channel 0=all, 1=name-only."""
        tokens = record_tokens(record)
        is_name = np.fromiter((t.split("|", 2)[1] in _NAME_KINDS for t in tokens), dtype=bool, count=len(tokens))
        df, pos = self.document_frequency(token_keys(tokens))
        use = (df > 0) & (df <= self.df_max)
        if not use.any():
            return np.empty(0, np.uint32), np.empty(0), np.empty(0), np.empty(0, np.uint8)
        weight = self.idf(df[use])
        spans = [(int(self.offsets[p]), int(self.offsets[p + 1])) for p in pos[use]]
        rows = np.concatenate([np.asarray(self.rows[s:e]) for s, e in spans])
        lengths = np.fromiter((e - s for s, e in spans), dtype=np.int64, count=len(spans))
        unique, inverse = np.unique(rows, return_inverse=True)
        score = np.bincount(inverse, weights=np.repeat(weight, lengths), minlength=len(unique))
        name = np.bincount(inverse, weights=np.repeat(weight * is_name[use], lengths), minlength=len(unique))
        top = _top(score, unique, self.k_all)
        rest = np.ones(len(unique), dtype=bool)
        rest[top] = False
        extra = np.flatnonzero(rest & (name > 0))
        if len(extra):
            extra = extra[_top(name[extra], unique[extra], self.k_name)]
        chosen = np.r_[top, extra].astype(np.int64)
        channel = np.r_[np.zeros(len(top), np.uint8), np.ones(len(extra), np.uint8)]
        return unique[chosen].astype(np.uint32), score[chosen], name[chosen], channel


def _top(score, rows, k):
    """Indices of the k best scores; ties broken by row so results are deterministic."""
    if len(score) > k:
        cut = np.partition(score, len(score) - k)[len(score) - k]
        candidates = np.flatnonzero(score >= cut)
    else:
        candidates = np.arange(len(score))
    order = np.lexsort((rows[candidates], -score[candidates]))
    return candidates[order[:k]]
