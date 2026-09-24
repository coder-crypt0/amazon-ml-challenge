"""Build reusable on-disk stores and blocking indexes."""
import argparse
from pathlib import Path
from .store import RecordStore
from .blocking import BlockingIndex


def build(data, artifacts, split, workers=4):
    root = Path(artifacts) / split
    paths = [Path(data) / split / f"{split}_source{i}.tsv" for i in (2, 3)]
    store = RecordStore.build(paths, root / "store")
    index = BlockingIndex.build(store, root / "index", workers=workers)
    print(f"Ready: {split}: {len(store):,} targets; {len(index.hashes):,} postings", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="student_resource/dataset")
    parser.add_argument("--artifacts", default="artifacts/v1")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build(args.data, args.artifacts, args.split, args.workers)
