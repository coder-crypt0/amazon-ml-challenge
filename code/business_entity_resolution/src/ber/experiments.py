"""Predeclared structural and geographic generalization ablations."""
import argparse
import json
from pathlib import Path
import numpy as np
from .checkpoint import persist_file
from .model import PairClassifier
from .pipeline import metric


def exclusive_probabilities(probabilities,pairs,query_indices,threshold):
    eligible=np.flatnonzero(np.isin(pairs[:,0],query_indices)&(probabilities>=threshold))
    scores=np.zeros(len(probabilities),dtype=np.float32)
    if not len(eligible):
        return scores
    order=eligible[np.lexsort((-probabilities[eligible],pairs[eligible,1]))]
    target=pairs[order,1]
    starts=np.r_[0,np.flatnonzero(target[1:]!=target[:-1])+1]
    for i,start in enumerate(starts):
        end=starts[i+1] if i+1<len(starts) else len(order)
        best=order[start]
        if end-start==1 or probabilities[best]>probabilities[order[start+1]]:
            scores[best]=probabilities[best]
    return scores


def run(artifacts,threads=4,country_transfer=False):
    root=Path(artifacts)/"experiment"
    report=json.loads((root/"validation.json").read_text())
    queries=json.loads((root/"queries.json").read_text(encoding="utf-8"))
    pairs=np.load(root/"pairs.npy");labels=np.load(root/"y.npy")
    probabilities=np.load(root/"probabilities.npy")
    config=json.loads((root/"calibration.json").read_text())
    threshold=config["threshold"]
    fit_end=report["split"]["fit"];tune_end=fit_end+report["split"]["tune"]
    indices={"tune":np.arange(fit_end,tune_end),"holdout":np.arange(tune_end,len(queries))}
    ablations={}
    for split,qids in indices.items():
        constrained=exclusive_probabilities(probabilities,pairs,qids,threshold)
        ablations[split]={"independent":metric(probabilities,pairs,labels,queries,qids,threshold),
                          "exclusive":metric(constrained,pairs,labels,queries,qids,threshold)}
    config["exclusive"]=ablations["tune"]["exclusive"]["macro_f05"]>ablations["tune"]["independent"]["macro_f05"]+1e-6
    report["graph_ablation"]=ablations
    report["exclusive_selected_on_tune"]=config["exclusive"]
    # A diagnostic only: country-held-out performance does not estimate France directly.
    if country_transfer:
        X=np.load(root/"X.npy",mmap_mode="r")
        report["country_transfer"]={}
        for country in sorted({q["country"] for q in queries}):
            fit=[i for i in range(fit_end) if queries[i]["country"]!=country]
            validation=[i for i in range(tune_end,len(queries)) if queries[i]["country"]==country]
            if not fit or not validation:
                continue
            mask=np.isin(pairs[:,0],fit)
            model=PairClassifier(n_estimators=250).fit_classifier(X[mask],labels[mask],seed=report["seed"],threads=threads)
            scores=model.predict_proba(X)
            report["country_transfer"][country]=metric(scores,pairs,labels,queries,validation,threshold)
    (root/"calibration.json").write_text(json.dumps(config,indent=2))
    (root/"validation.json").write_text(json.dumps(report,indent=2))
    for name in ("calibration.json","validation.json"):
        persist_file(root/name)
    print(json.dumps({"graph_ablation":ablations,"exclusive_selected":config["exclusive"],
                      "country_transfer":report.get("country_transfer",{})},indent=2),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--artifacts",default="artifacts/v1")
    p.add_argument("--threads",type=int,default=4);p.add_argument("--country-transfer",action="store_true")
    a=p.parse_args();run(a.artifacts,a.threads,a.country_transfer)
