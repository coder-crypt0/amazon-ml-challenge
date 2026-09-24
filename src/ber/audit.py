"""Bounded-memory audit, including exact membership of every scored target ID."""
import argparse
import csv
from itertools import chain,zip_longest
import json
from pathlib import Path
import numpy as np


def _rows(path,expected_header=None):
    with Path(path).open(encoding="utf-8-sig",newline="") as handle:
        reader=csv.reader(handle,delimiter="\t")
        header=next(reader,None)
        if expected_header is not None and header!=expected_header:
            raise ValueError(f"Invalid header in {path}: {header}")
        if not header:
            raise ValueError(f"Empty file: {path}")
        yield from reader


def audit_outputs(matching_path,candidate_path,test_dir):
    test_dir=Path(test_dir)
    errors=[]
    counts=dict(s1=0,matching=0,candidate=0,matched_ids=0,candidate_ids=0)
    def error(message):
        if len(errors)<100:
            errors.append(message)
    def target_ids():
        for row in chain(_rows(test_dir/"test_source2.tsv"),_rows(test_dir/"test_source3.tsv")):
            encoded=row[0].encode("utf-8")
            if len(encoded)>64:
                raise ValueError("Target ID exceeds the audited 64-byte storage limit")
            yield encoded
    targets=np.fromiter(target_ids(),dtype="S64")
    targets.sort()
    pending=[]
    def check_pending():
        if not pending:
            return
        ids=np.asarray(pending,dtype="S64")
        positions=np.searchsorted(targets,ids)
        valid=positions<len(targets)
        valid[valid]&=targets[positions[valid]]==ids[valid]
        if not valid.all():
            error(f"Unknown target IDs, first examples: {ids[~valid][:5].tolist()}")
        pending.clear()
    expected=_rows(test_dir/"test_source1.tsv")
    matches=_rows(matching_path,["source1_entity_id","matched_entity_ids"])
    candidates=_rows(candidate_path,["source1_entity_id","candidate_entity_ids"])
    for index,(source,match,candidate) in enumerate(zip_longest(expected,matches,candidates),1):
        if source is None or match is None or candidate is None:
            error(f"Row count mismatch at row {index}")
            continue
        counts["s1"]+=1;counts["matching"]+=1;counts["candidate"]+=1
        if len(match)!=2 or len(candidate)!=2:
            error(f"Malformed output row {index}")
            continue
        if source[0]!=match[0] or source[0]!=candidate[0]:
            error(f"Source1 coverage/order mismatch at row {index}")
        mids=match[1].split(",") if match[1] else []
        cids=candidate[1].split(",") if candidate[1] else []
        counts["matched_ids"]+=len(mids);counts["candidate_ids"]+=len(cids)
        if len(mids)!=len(set(mids)) or len(cids)!=len(set(cids)):
            error(f"Duplicate IDs in row {index}")
        if not set(mids)<=set(cids):
            error(f"Match absent from candidates at row {index}")
        for entity_id in set(cids)|set(mids):
            encoded=entity_id.encode("utf-8")
            if not entity_id.startswith(("S2-","S3-")) or len(encoded)>64:
                error(f"Invalid target ID at row {index}")
            else:
                pending.append(encoded)
        if len(pending)>=65536:
            check_pending()
    check_pending()
    return dict(ok=not errors,errors=errors,warnings=[],counts=counts)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--matching",required=True);p.add_argument("--candidate",required=True)
    p.add_argument("--test-dir",required=True);p.add_argument("--report")
    args=p.parse_args()
    report=audit_outputs(args.matching,args.candidate,args.test_dir)
    if args.report:
        Path(args.report).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return 0 if report["ok"] else 1


if __name__=="__main__":
    raise SystemExit(main())
