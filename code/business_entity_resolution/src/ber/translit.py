"""Learn a native-script -> Latin token table from TRAIN ground-truth pairs only.

Native-script names in Source 2/3 transliterate the reference name token for token, so
aligned pairs with equal token counts give direct evidence.  Native address tokens are
state names; they are mapped by co-occurrence with the reference address.  Reference
entities used for validation can be excluded so the table never sees them.
"""
import argparse
import collections
import json
import random
import re
from pathlib import Path

import numpy as np
from .blocking import key_hash
from .checkpoint import persist_file, restore_file

NATIVE = re.compile(r"[\u0900-\u0DFF]")
_TOKEN = re.compile(r"[\w\u0900-\u0DFF]+", re.UNICODE)


def _tokens(text):
    return _TOKEN.findall(str(text or "").lower())


def _key(entity_id):
    return key_hash(entity_id)


def learn(data, destination, exclude=(), fraction=0.25, seed=20260925):
    destination = Path(destination)
    if restore_file(destination):
        return json.loads(destination.read_text(encoding="utf-8"))
    train = Path(data) / "train"
    exclude, rng = set(exclude), random.Random(seed)
    kept, targets, owners = {}, [], []
    with (train / "train_ground_truth.tsv").open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            s1, _, rest = line.rstrip("\n").partition("\t")
            if rest and s1 not in exclude and rng.random() < fraction:
                kept[s1] = len(kept)
                for t in rest.split(","):
                    targets.append(_key(t))
                    owners.append(kept[s1])
    targets, owners = np.asarray(targets, np.uint64), np.asarray(owners, np.int32)
    order = np.argsort(targets)
    targets, owners = targets[order], owners[order]
    reference = [None] * len(kept)
    with (train / "train_source1.tsv").open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            p = line.rstrip("\n").split("\t")
            if p[0] in kept:
                reference[kept[p[0]]] = (p[1], p[2])
    names, co, latin_df, native_df = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    for source in ("train_source2.tsv", "train_source3.tsv"):
        with (train / source).open(encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                p = line.rstrip("\n").split("\t")
                if len(p) < 3 or not (NATIVE.search(p[1]) or NATIVE.search(p[2])):
                    continue
                k = np.uint64(_key(p[0]))
                j = np.searchsorted(targets, k)
                if j >= len(targets) or targets[j] != k:
                    continue
                name, address = reference[owners[j]]
                a, b = _tokens(p[1]), _tokens(name)
                if NATIVE.search(p[1]) and len(a) == len(b):
                    for x, y in zip(a, b):
                        if NATIVE.search(x) and not NATIVE.search(y):
                            names[(x, y)] += 1
                if NATIVE.search(p[2]):
                    latin = set(_tokens(address))
                    latin_df.update(latin)
                    for x in {t for t in _tokens(p[2]) if NATIVE.search(t)}:
                        native_df[x] += 1
                        for y in latin:
                            co[(x, y)] += 1
    table = {"name": {}, "addr": {}}
    by_source = collections.defaultdict(list)
    for (x, y), c in names.items():
        by_source[x].append((c, y))
    for x, options in by_source.items():
        options.sort(reverse=True)
        if options[0][0] >= 2 and options[0][0] >= 0.5 * sum(c for c, _ in options):
            table["name"][x] = options[0][1]
    by_native = collections.defaultdict(list)
    for (x, y), c in co.items():
        if c >= 3:
            by_native[x].append((c / native_df[x], c / latin_df[y] ** 0.5, y))
    for x, options in by_native.items():
        share, _, y = max(options)
        if share >= 0.6:
            table["addr"][x] = y
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(table, ensure_ascii=False, indent=0), encoding="utf-8")
    persist_file(destination)
    print(f"Transliteration table: {len(table['name'])} name tokens, {len(table['addr'])} address tokens", flush=True)
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="student_resource/dataset")
    p.add_argument("--output", default="artifacts/v3/translit.json")
    p.add_argument("--exclude", default=None, help="queries.json whose reference entities are held out")
    a = p.parse_args()
    held = [q["entity_id"] for q in json.loads(Path(a.exclude).read_text(encoding="utf-8"))] if a.exclude else ()
    learn(a.data, a.output, held)
