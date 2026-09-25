"""Resumable full-test scoring. Saved chunks contain every scored candidate."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import time

import numpy as np
from .checkpoint import ensure_config, file_digest, persist_file, restore_file
from .features import FEATURE_NAMES, pair_features
from .model import InferenceModel
from .pipeline import ranked_candidates, prepared_target, worker_init
from .store import RecordStore, iter_records

_MODEL=None


def initialize(artifacts,max_postings):
    global _MODEL
    worker_init(artifacts,"test",max_postings)
    _MODEL=InferenceModel(Path(artifacts)/"experiment")


def score_batch(args):
    batch,top_k=args
    features,rows,offsets=[],[],[0]
    for query in batch:
        left,candidates,_=ranked_candidates(query,top_k)
        rows.extend(candidates)
        features.extend(pair_features(left,prepared_target(i)) for i in candidates)
        offsets.append(len(rows))
    X=np.asarray(features,dtype=np.float32).reshape(-1,len(FEATURE_NAMES))
    probabilities=_MODEL.predict_proba(X) if len(X) else np.empty(0,dtype=np.float32)
    return dict(query_ids=np.asarray([q["entity_id"] for q in batch]),
                rows=np.asarray(rows,dtype=np.uint32),offsets=np.asarray(offsets,dtype=np.uint32),
                probabilities=np.asarray(probabilities,dtype=np.float32))


def predict(data,artifacts,workers=4,batch_size=512):
    artifacts=Path(artifacts)
    model_dir=artifacts/"experiment"
    config=json.loads((model_dir/"calibration.json").read_text())
    destination=artifacts/"predictions"
    destination.mkdir(exist_ok=True)
    hashes={p.name:file_digest(p) for p in (model_dir/"model.txt",model_dir/"calibration.json")}
    if config.get("xgboost_weight"):
        hashes["xgboost.ubj"]=file_digest(model_dir/"xgboost.ubj")
    ensure_config(destination/"config.json",{"models":hashes,"batch_size":batch_size,
        "test_source1_sha256":file_digest(Path(data)/"test/test_source1.tsv")})
    iterator=iter_records([Path(data)/"test/test_source1.tsv"])
    started=time.monotonic()
    jobs={}
    batch_index=0
    completed=0
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,
                             initargs=(str(artifacts),config["max_postings"])) as executor:
        exhausted=False
        while not exhausted or jobs:
            while not exhausted and len(jobs)<2*workers:
                batch=[]
                for _ in range(batch_size):
                    row=next(iterator,None)
                    if row is None:
                        exhausted=True
                        break
                    batch.append(row)
                if not batch:
                    break
                path=destination/f"batch_{batch_index:06d}.npz"
                batch_index+=1
                if restore_file(path):
                    completed+=len(batch)
                    continue
                jobs[executor.submit(score_batch,(batch,config["top_k"]))]=(path,len(batch))
            if jobs:
                done,_=wait(jobs,return_when=FIRST_COMPLETED)
                for future in done:
                    path,count=jobs.pop(future)
                    result=future.result()
                    temp=path.with_suffix(".tmp.npz")
                    np.savez_compressed(temp,**result)
                    temp.replace(path)
                    persist_file(path)
                    completed+=count
                    print(f"Prediction checkpoints: {completed:,} queries; {time.monotonic()-started:.0f}s",flush=True)
    summary={"queries":completed,"batches":batch_index,"batch_size":batch_size,"seconds":time.monotonic()-started}
    (destination/"COMPLETE").write_text(json.dumps(summary))
    persist_file(destination/"COMPLETE")
    return summary


def select_matches(probabilities,rule):
    """Tuned policy for one query: p>=t, plus the top candidate when p>=t1, then p>=r*best."""
    p=np.asarray(probabilities,dtype=np.float64)
    chosen=p>=rule["t"]
    if len(p):
        top=int(np.argmax(p))
        if p[top]>=rule["t1"]:
            chosen[top]=True
        if rule.get("r"):
            chosen&=p>=rule["r"]*p[top]
    return chosen


def assemble(artifacts,output,exclusive=None,policy=None,data=None):
    artifacts,output=Path(artifacts),Path(output)
    policy=json.loads(Path(policy).read_text()) if policy else None
    directory=artifacts/"predictions"
    if not restore_file(directory/"COMPLETE"):
        raise ValueError("Prediction not complete; resume the prediction stage first")
    summary=json.loads((directory/"COMPLETE").read_text())
    paths=[directory/f"batch_{i:06d}.npz" for i in range(summary["batches"])]
    for p in paths:
        if not restore_file(p):
            raise ValueError(f"Missing prediction checkpoint: {p}")
    store=RecordStore(artifacts/"test/store")
    config=json.loads((artifacts/"experiment/calibration.json").read_text())
    threshold=config["threshold"]
    if exclusive is None:
        exclusive=config.get("exclusive",False) and policy is None
    if policy is not None and exclusive:
        raise ValueError("A tuned policy and target exclusivity are separate experiments")
    countries={}
    if policy is not None and policy.get("by_country"):
        countries={r["entity_id"]:r["country"] for r in iter_records([Path(data)/"test/test_source1.tsv"])}
    owners=None
    conflicts=0
    if exclusive:
        best=np.full(len(store),-1.,dtype=np.float32)
        owners=np.full(len(store),-1,dtype=np.int32)
        q_offset=0
        for path in paths:
            with np.load(path) as chunk:
                for i in range(len(chunk["query_ids"])):
                    s,e=chunk["offsets"][i:i+2]
                    for target,p in zip(chunk["rows"][s:e],chunk["probabilities"][s:e]):
                        if p<threshold:
                            continue
                        if best[target]>=threshold:
                            conflicts+=1
                        if p>best[target]:
                            best[target]=p
                            owners[target]=q_offset+i
                        elif p==best[target]:
                            owners[target]=-1
                q_offset+=len(chunk["query_ids"])
    output.mkdir(parents=True,exist_ok=True)
    total=matched=0
    with (output/"matching_results.tsv.tmp").open("w",encoding="utf-8",newline="") as mf, (output/"candidate_pairs.tsv.tmp").open("w",encoding="utf-8",newline="") as cf:
        mw,cw=csv.writer(mf,delimiter="\t",lineterminator="\n"),csv.writer(cf,delimiter="\t",lineterminator="\n")
        mw.writerow(["source1_entity_id","matched_entity_ids"])
        cw.writerow(["source1_entity_id","candidate_entity_ids"])
        for path in paths:
            with np.load(path) as chunk:
                for i,qid in enumerate(chunk["query_ids"]):
                    start,end=chunk["offsets"][i:i+2]
                    candidates=[]; matches=[]
                    probabilities=chunk["probabilities"][start:end]
                    if policy is None:
                        chosen=probabilities>=threshold
                    else:
                        chosen=select_matches(probabilities,policy["by_country"].get(countries.get(str(qid),""),policy["global"]))
                    for ri,keep in zip(chunk["rows"][start:end],chosen):
                        entity_id=store[int(ri)]["entity_id"]
                        candidates.append(entity_id)
                        if keep and (owners is None or owners[ri]==total):
                            matches.append(entity_id)
                    cw.writerow([qid,",".join(candidates)])
                    mw.writerow([qid,",".join(matches)])
                    total+=1; matched+=len(matches)
    for name in ("matching_results.tsv","candidate_pairs.tsv"):
        (output/(name+".tmp")).replace(output/name)
    store.close()
    result={"queries":total,"matched_links":matched,"exclusive":exclusive,"conflicts":conflicts,"policy":policy}
    (output/"prediction_summary.json").write_text(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("score","assemble"))
    p.add_argument("--data",default="student_resource/dataset")
    p.add_argument("--artifacts",default="artifacts/v1")
    p.add_argument("--output",default="output")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--exclusive",action="store_true",default=None)
    p.add_argument("--policy",default=None,help="policy.json from scripts/tune_decision.py")
    a=p.parse_args()
    print(json.dumps(predict(a.data,a.artifacts,a.workers,a.batch_size) if a.stage=="score" else assemble(a.artifacts,a.output,a.exclusive,a.policy,a.data),indent=2))
