"""Compact random-access record storage; original input never needs to fit in RAM."""
from array import array
import csv
import json
import mmap
from pathlib import Path

import numpy as np
from .checkpoint import restore_directory, persist_directory, file_digest

FIELDS = ("entity_id", "business_name", "business_address", "country")


def iter_records(paths):
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"Unexpected source schema: {path}: {reader.fieldnames}")
            for row in reader:
                if None in row or any(row[k] is None for k in FIELDS):
                    raise ValueError(f"Malformed source row in {path}")
                yield row


class RecordStore:
    def __init__(self, path):
        self.path = Path(path)
        if not (self.path / "COMPLETE").exists():
            raise ValueError(f"Incomplete record store: {path}")
        self.offsets = np.load(self.path / "offsets.npy", mmap_mode="r")
        self.handle = (self.path / "records.jsonl").open("rb")
        self.data = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)

    @classmethod
    def build(cls, paths, destination):
        destination = Path(destination)
        signatures = [{"name": Path(p).name, "bytes": Path(p).stat().st_size,
                       "sha256": file_digest(p)} for p in paths]
        restore_directory(destination)
        if (destination / "COMPLETE").exists():
            if json.loads((destination / "inputs.json").read_text()) != signatures:
                raise ValueError("Input data changed; use a fresh artifact directory")
            return cls(destination)
        destination.mkdir(parents=True, exist_ok=True)
        offsets = array("Q", [0])
        with (destination / "records.jsonl").open("wb") as handle:
            for i, row in enumerate(iter_records(paths), 1):
                payload = (json.dumps([row[k] for k in FIELDS], ensure_ascii=False,
                                      separators=(",", ":")) + "\n").encode("utf-8")
                handle.write(payload)
                offsets.append(offsets[-1] + len(payload))
                if i % 500000 == 0:
                    print(f"Record store: {i:,} records", flush=True)
        np.save(destination / "offsets.npy", np.asarray(offsets, dtype="<u8"))
        (destination / "inputs.json").write_text(json.dumps(signatures, indent=2))
        (destination / "COMPLETE").write_text("complete\n")
        persist_directory(destination)
        return cls(destination)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, row):
        if not 0 <= row < len(self):
            raise IndexError(row)
        start, end = int(self.offsets[row]), int(self.offsets[row + 1])
        return dict(zip(FIELDS, json.loads(self.data[start:end])))

    def __iter__(self):
        with (self.path / "records.jsonl").open("rb") as handle:
            for line in handle:
                yield dict(zip(FIELDS, json.loads(line)))

    def close(self):
        self.data.close()
        self.handle.close()
