"""Deterministic blocking with a sorted, memory-mapped posting index."""
from pathlib import Path
import hashlib
import json
import time
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from .text import canonical, address, latin, salient_tokens, ADDRESS_COMMON
from .checkpoint import restore_directory, persist_directory, restore_file, persist_file, remote_path

DTYPE = np.dtype([("hash", "<u8"), ("row", "<u4")])
INDEX_VERSION = 3


def key_hash(key):
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little")


def blocking_keys(record, country=""):
    name = record.get("business_name", record.get("name", ""))
    addr = record.get("business_address", record.get("address", ""))
    country = latin(country or record.get("country", ""))
    n, a = canonical(name), address(addr)
    out = []
    def add(kind, value):
        if value:
            out.append(f"{country}|{kind}|{value}")
    add("n", n)
    add("ns", " ".join(sorted(n.split())))
    add("a", a)
    add("as", " ".join(sorted(a.split())))
    nt = salient_tokens(n)[:5]
    at = a.split()
    numbers = [t for t in at if any(c.isdigit() for c in t)]
    words = [t for t in at if len(t) >= 3 and not any(c.isdigit() for c in t) and t not in ADDRESS_COMMON]
    # Number + independent street/locality words survive component reordering.
    for number in numbers[:2]:
        for word in words[:4]:
            add("h", number + " " + word)
    for token in nt:
        add("t", token)
    for x, y in list(combinations(sorted(nt), 2))[:6]:
        add("p", x + " " + y)
    for token in nt[:3]:
        if len(token) >= 5:
            add("f", token[:5])
    for word in words[:2]:
        add("w", word)
    return tuple(dict.fromkeys(out))[:30]


def _make_part(args):
    store_path, start, stop, part_path = args
    marker = Path(part_path).with_suffix(".json")
    if marker.exists() and Path(part_path).exists():
        count = json.loads(marker.read_text())["count"]
        if Path(part_path).stat().st_size == count * DTYPE.itemsize:
            return part_path, count
    from .store import RecordStore
    store = RecordStore(store_path)
    buffer = np.empty(250000, dtype=DTYPE)
    count = total = 0
    with open(part_path, "wb") as handle:
        for row in range(start, stop):
            for key in blocking_keys(store[row]):
                buffer[count] = (key_hash(key), row)
                count += 1
                if count == len(buffer):
                    buffer.tofile(handle)
                    total += count
                    count = 0
        buffer[:count].tofile(handle)
        total += count
    store.close()
    marker.write_text(json.dumps({"count":total}))
    return part_path, total


class BlockingIndex:
    def __init__(self, path, max_postings=150):
        self.path = Path(path)
        self.max_postings = int(max_postings)
        if not (self.path / "COMPLETE").exists():
            raise ValueError("Incomplete blocking index")
        metadata = json.loads((self.path / "metadata.json").read_text())
        if metadata["version"] != INDEX_VERSION:
            raise ValueError("Blocking index version changed; rebuild in a new directory")
        self.hashes = np.load(self.path / "hashes.npy", mmap_mode="r")
        self.rows = np.load(self.path / "rows.npy", mmap_mode="r")

    @classmethod
    def build(cls, records, path, country="", chunk_size=250000, max_postings=150, workers=4):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        restore_directory(path)
        if (path / "COMPLETE").exists():
            return cls(path, max_postings)
        started = time.monotonic()
        parts = []
        if hasattr(records, "path") and hasattr(records, "__len__"):
            jobs = [(str(records.path), start, min(start + 250000, len(records)), str(path / f"part_{start:09d}.bin"))
                    for start in range(0, len(records), 250000)]
            for job in jobs:
                restore_file(Path(job[3]).with_suffix(".json"))
                if Path(job[3]).with_suffix(".json").exists():
                    restore_file(job[3])
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for i, result in enumerate(pool.map(_make_part, jobs), 1):
                    parts.append(result)
                    persist_file(result[0])
                    persist_file(Path(result[0]).with_suffix(".json"))
                    print(f"Blocking keys: {i}/{len(jobs)} chunks, {time.monotonic()-started:.0f}s", flush=True)
            n_records = len(records)
        else:
            buffer, n_records = [], 0
            for row, rec in enumerate(records):
                n_records = row + 1
                buffer.extend((key_hash(k), row) for k in blocking_keys(rec, country))
                if len(buffer) >= chunk_size:
                    part = path / f"part_{len(parts):09d}.bin"
                    np.array(buffer, dtype=DTYPE).tofile(part)
                    parts.append((str(part), len(buffer)))
                    buffer = []
            if buffer:
                part = path / f"part_{len(parts):09d}.bin"
                np.array(buffer, dtype=DTYPE).tofile(part)
                parts.append((str(part), len(buffer)))
        total = sum(size for _, size in parts)
        print(f"Sorting {total:,} postings for {n_records:,} records", flush=True)
        # Only contiguous hashes, row ids, and argsort indices are resident (~20 bytes/posting).
        hashes = np.empty(total, dtype=np.uint64)
        rowids = np.empty(total, dtype=np.uint32)
        pos = 0
        for part, count in parts:
            data = np.fromfile(part, dtype=DTYPE)
            hashes[pos:pos+count] = data["hash"]
            rowids[pos:pos+count] = data["row"]
            pos += count
        if parts:
            del data
        order = np.argsort(hashes, kind="stable")
        h_out = np.lib.format.open_memmap(path / "hashes.npy", mode="w+", dtype="<u8", shape=(total,))
        r_out = np.lib.format.open_memmap(path / "rows.npy", mode="w+", dtype="<u4", shape=(total,))
        for start in range(0, total, 1000000):
            idx = order[start:start+1000000]
            h_out[start:start+len(idx)] = hashes[idx]
            r_out[start:start+len(idx)] = rowids[idx]
        h_out.flush()
        r_out.flush()
        del h_out, r_out, hashes, rowids, order
        for part, _ in parts:
            Path(part).unlink()
            Path(part).with_suffix(".json").unlink(missing_ok=True)
        (path / "metadata.json").write_text(json.dumps({"version":INDEX_VERSION,"rows":n_records,"postings":total,
                                                       "elapsed_seconds":time.monotonic()-started},indent=2))
        (path / "COMPLETE").write_text("complete\n")
        persist_directory(path)
        # Canonical index is durable; obsolete raw shards no longer consume Drive quota.
        for part, _ in parts:
            for obsolete in (Path(part), Path(part).with_suffix(".json")):
                remote=remote_path(obsolete)
                if remote is not None:
                    remote.unlink(missing_ok=True)
        return cls(path, max_postings)

    def query(self, record, country="", limit=None):
        keys = np.asarray([key_hash(k) for k in blocking_keys(record, country)], dtype=np.uint64)
        starts = np.searchsorted(self.hashes, keys, side="left")
        ends = np.searchsorted(self.hashes, keys, side="right")
        hits = [self.rows[int(s):int(e)] for s, e in zip(starts, ends) if 0 < e-s <= self.max_postings]
        result = np.unique(np.concatenate(hits)) if hits else np.empty(0, np.uint32)
        # Candidate truncation belongs to text ranking, never physical row order.
        if limit is not None:
            raise ValueError("Rank candidates by text before applying a limit")
        return result
