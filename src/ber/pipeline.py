"""Sample, retrieve, train, and evaluate against the complete target corpus."""
import argparse
import csv
from functools import lru_cache
import heapq
import json
from pathlib import Path
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from .blocking import BlockingIndex
from .features import FEATURE_NAMES, pair_features, preliminary_similarity, prepare_record
from .model import PairClassifier, train_xgboost
from .store import RecordStore, iter_records
from .checkpoint import ensure_config, persist_file, restore_file

_STORE = _INDEX = _MODEL = None


def worker_init(artifact_root, split, max_postings=150, model_path=None):
    global _STORE, _INDEX, _MODEL
    _STORE = RecordStore(Path(artifact_root) / split / "store")
    _INDEX = BlockingIndex(Path(artifact_root) / split / "index", max_postings=max_postings)
    _MODEL = PairClassifier.load(model_path) if model_path else None


@lru_cache(maxsize=12000)
def prepared_target(index):
    return prepare_record(_STORE[index])


def ranked_candidates(query, top_k):
    raw = _INDEX.query(query)
    left = prepare_record(query)
    ranks = [(preliminary_similarity(left, prepared_target(int(i))), int(i)) for i in raw]
    # Similarity only; row index is a deterministic tiebreak, never a model feature.
    ranked = heapq.nlargest(top_k, ranks)
    return left, [i for _, i in ranked], len(raw)


def candidate_batch(args):
    batch, top_k = args
    rows, features, labels, raw_sizes = [], [], [], []
    for qi, query in batch:
        left, candidates, raw_size = ranked_candidates(query, top_k)
        truth = set(query.get("truth", ()))
        raw_sizes.append((qi, raw_size))
        for ri in candidates:
            right = prepared_target(ri)
            rows.append((qi, ri))
            features.append(pair_features(left, right))
            labels.append(right["entity_id"] in truth)
    return (np.asarray(rows,dtype=np.int32).reshape(-1,2),
            np.asarray(features,dtype=np.float32).reshape(-1,len(FEATURE_NAMES)),
            np.asarray(labels,dtype=np.uint8), raw_sizes)


def sample_queries(data, sample_size, seed):
    rng = random.Random(seed)
    reservoir = []
    for i, row in enumerate(iter_records([Path(data)/"train/train_source1.tsv"])):
        if i < sample_size:
            reservoir.append(row)
        else:
            j = rng.randrange(i+1)
            if j < sample_size:
                reservoir[j] = row
    rng.shuffle(reservoir)
    lookup = {r["entity_id"]:r for r in reservoir}
    with (Path(data)/"train/train_ground_truth.tsv").open(encoding="utf-8",newline="") as handle:
        for row in csv.DictReader(handle,delimiter="\t"):
            if row["source1_entity_id"] in lookup:
                lookup[row["source1_entity_id"]]["truth"] = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
    if any("truth" not in row for row in reservoir):
        raise ValueError("Training source is not fully covered by ground truth")
    return reservoir


def create_training_candidates(args):
    destination = Path(args.artifacts)/"experiment"
    destination.mkdir(parents=True,exist_ok=True)
    ensure_config(destination/"candidate_config.json",{"samples":args.samples,"seed":args.seed,
        "top_k":args.top_k,"max_postings":args.max_postings,"feature_names":list(FEATURE_NAMES)})
    restore_file(destination/"queries.json")
    if (destination/"queries.json").exists():
        queries=json.loads((destination/"queries.json").read_text(encoding="utf-8"))
    else:
        queries=sample_queries(args.data,args.samples,args.seed)
        (destination/"queries.json").write_text(json.dumps(queries,ensure_ascii=False),encoding="utf-8")
        persist_file(destination/"queries.json")
    enumerated=list(enumerate(queries))
    batches = [(enumerated[start:start+64],args.top_k) for start in range(0,len(queries),64)]
    chunks=destination/"chunks"
    chunks.mkdir(exist_ok=True)
    missing=[]
    for i,batch in enumerate(batches):
        path=chunks/f"{i:06d}.npz"
        if not restore_file(path):
            missing.append((i,batch))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,
                             initargs=(args.artifacts,"train",args.max_postings)) as pool:
        for j,((i,_),(pairs,x,y,raw)) in enumerate(zip(missing,pool.map(candidate_batch,[b for _,b in missing])),1):
            path=chunks/f"{i:06d}.npz"
            temporary=path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary,pairs=pairs,X=x,y=y,raw=np.asarray(raw))
            temporary.replace(path)
            persist_file(path)
            if j%10==0 or j==len(missing):
                print(f"Candidate features: {len(batches)-len(missing)+j}/{len(batches)} checkpoint batches; {time.monotonic()-started:.0f}s",flush=True)
    paths=[chunks/f"{i:06d}.npz" for i in range(len(batches))]
    total=0
    for path in paths:
        with np.load(path) as chunk:
            total+=len(chunk["y"])
    shapes={"pairs":((total,2),np.int32),"X":((total,len(FEATURE_NAMES)),np.float32),"y":((total,),np.uint8)}
    arrays={name:np.lib.format.open_memmap(destination/f"{name}.npy",mode="w+",dtype=dtype,shape=shape)
            for name,(shape,dtype) in shapes.items()}
    pos=0
    raw_sizes=[]
    for path in paths:
        with np.load(path) as chunk:
            count=len(chunk["y"])
            for name,array in arrays.items():
                array[pos:pos+count]=chunk[name]
            raw_sizes.extend(chunk["raw"].tolist())
            pos+=count
    for name,array in arrays.items():
        array.flush()
        persist_file(destination/f"{name}.npy")
    (destination/"retrieval.json").write_text(json.dumps({"top_k":args.top_k,"max_postings":args.max_postings,
        "samples":len(queries),"seed":args.seed,"elapsed_seconds":time.monotonic()-started,
        "raw_candidates_mean":float(np.mean([v for _,v in raw_sizes])),"feature_names":FEATURE_NAMES},indent=2))
    persist_file(destination/"retrieval.json")


def metric(probabilities, pairs, labels, queries, indices, threshold):
    n = len(queries)
    selected = probabilities >= threshold
    counts = np.bincount(pairs[selected,0],minlength=n)
    correct = np.bincount(pairs[selected & (labels==1),0],minlength=n)
    true_counts = np.asarray([len(q["truth"]) for q in queries])
    denominator = .25*true_counts + counts
    f = np.divide(1.25*correct,denominator,out=np.ones(n,dtype=float),where=denominator>0)
    indices = np.asarray(indices)
    singleton = indices[true_counts[indices]==0]
    candidate_correct = np.bincount(pairs[labels==1,0],minlength=n)
    tp, pred, actual = correct[indices].sum(), counts[indices].sum(), true_counts[indices].sum()
    return {"macro_f05":float(f[indices].mean()),"micro_precision":float(tp/pred) if pred else 0.,
            "micro_recall":float(tp/actual) if actual else 0.,
            "candidate_recall":float(candidate_correct[indices].sum()/actual) if actual else 1.,
            "singleton_accuracy":float((counts[singleton]==0).mean()) if len(singleton) else None,
            "queries":len(indices),"true_links":int(actual),"predicted_links":int(pred),
            "true_positive_links":int(tp),"singleton_queries":len(singleton)}


def train_and_evaluate(args):
    destination = Path(args.artifacts)/"experiment"
    queries = json.loads((destination/"queries.json").read_text(encoding="utf-8"))
    X = np.load(destination/"X.npy",mmap_mode="r")
    pairs = np.load(destination/"pairs.npy")
    labels = np.load(destination/"y.npy")
    n = len(queries)
    train_end, tune_end = int(n*.7),int(n*.85)
    masks = {"fit":pairs[:,0]<train_end,"tune":(pairs[:,0]>=train_end)&(pairs[:,0]<tune_end),"holdout":pairs[:,0]>=tune_end}
    ensure_config(destination/"training_config.json",{"seed":args.seed,"trees":args.trees,
        "xgboost_device":args.xgboost_device,"fit_fraction":.7,"tune_fraction":.15})
    model = PairClassifier(n_estimators=args.trees).fit_classifier(X[masks["fit"]],labels[masks["fit"]],
        seed=args.seed,threads=args.workers,checkpoint=destination/"lgb_progress.txt")
    model.save(destination/"model.txt")
    persist_file(destination/"model.txt")
    lgb_probabilities = model.predict_proba(X)
    xgb_probabilities = None
    if args.xgboost_device!="none":
        xgb_model=train_xgboost(X[masks["fit"]],labels[masks["fit"]],destination/"xgboost.ubj",
            device=args.xgboost_device,rounds=args.trees,seed=args.seed,threads=args.workers)
        xgb_model.set_param({"device":"cpu","nthread":args.workers})
        xgb_probabilities=xgb_model.inplace_predict(np.asarray(X))
    tune_indices=np.arange(train_end,tune_end)
    curve=[]
    for weight in ([0.,.25,.5,.75,1.] if xgb_probabilities is not None else [0.]):
        probabilities=lgb_probabilities if not weight else (1-weight)*lgb_probabilities+weight*xgb_probabilities
        for threshold in np.r_[np.arange(.1,.96,.025),.975,.99,.995]:
            stats=metric(probabilities,pairs,labels,queries,tune_indices,float(threshold))
            curve.append({"threshold":float(threshold),"xgboost_weight":weight,**stats})
    best=max(curve,key=lambda row:(row["macro_f05"],row["threshold"]))
    threshold=best["threshold"]
    weight=best["xgboost_weight"]
    probabilities=lgb_probabilities if not weight else (1-weight)*lgb_probabilities+weight*xgb_probabilities
    np.save(destination/"probabilities.npy",probabilities)
    persist_file(destination/"probabilities.npy")
    report={"split":{"fit":train_end,"tune":tune_end-train_end,"holdout":n-tune_end},"seed":args.seed,
            "threshold":threshold,"tuning":best,"threshold_curve":curve,
            "holdout":metric(probabilities,pairs,labels,queries,np.arange(tune_end,n),threshold),
            "holdout_by_country":{},"model":"LightGBM + XGBoost" if weight else "LightGBM",
            "xgboost_weight":weight,"model_license":"MIT / Apache-2.0",
            "validation_scope":"Random entity-disjoint queries; complete training target corpus; settings chosen on tuning only"}
    for country in sorted({q["country"] for q in queries}):
        indices=[i for i in range(tune_end,n) if queries[i]["country"]==country]
        if indices:
            report["holdout_by_country"][country]=metric(probabilities,pairs,labels,queries,indices,threshold)
    importance=model.model.feature_importance(importance_type="gain")
    report["feature_importance"]={name:float(value) for name,value in sorted(zip(FEATURE_NAMES,importance),key=lambda p:-p[1])}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/validation.json").write_text(json.dumps(report,indent=2))
    (destination/"validation.json").write_text(json.dumps(report,indent=2))
    persist_file(destination/"validation.json")
    (destination/"calibration.json").write_text(json.dumps({"threshold":threshold,"top_k":args.top_k,
        "max_postings":args.max_postings,"xgboost_weight":weight},indent=2))
    persist_file(destination/"calibration.json")
    print(json.dumps({"threshold":threshold,"tuning":best,"holdout":report["holdout"],
                      "holdout_by_country":report["holdout_by_country"]},indent=2),flush=True)


def parser():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("candidates","train"))
    p.add_argument("--data",default="student_resource/dataset")
    p.add_argument("--artifacts",default="artifacts/v1")
    p.add_argument("--samples",type=int,default=30000)
    p.add_argument("--top-k",type=int,default=32)
    p.add_argument("--max-postings",type=int,default=150)
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--seed",type=int,default=20260925)
    p.add_argument("--trees",type=int,default=450)
    p.add_argument("--xgboost-device",choices=("none","cpu","cuda"),default="none")
    return p


if __name__=="__main__":
    args=parser().parse_args()
    if args.stage=="candidates":
        create_training_candidates(args)
    else:
        train_and_evaluate(args)
