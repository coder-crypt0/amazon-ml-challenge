"""One conservative sibling-evidence pass over already-scored candidates."""
import json
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz
from .corpus import Corpus
from .reciprocal import policy_scores
from .setmodel import macro_f05


def propagate_batch(probabilities,rows,mask,targets,threshold,delta,high_confidence=.97):
    result=np.array(probabilities,dtype=np.float32,copy=True)
    if delta<=0:return result
    for q in range(len(rows)):
        valid=np.flatnonzero(mask[q])
        siblings=valid[result[q,valid]>=high_confidence]
        borderline=valid[(result[q,valid]>=threshold-.15)&(result[q,valid]<threshold)]
        if not len(siblings) or not len(borderline):continue
        siblings=siblings[np.argsort(-result[q,siblings])[:5]]
        for position in borderline:
            candidate=int(rows[q,position])
            name=targets.names[candidate];address=targets.addresses[candidate]
            if not name and not address:continue
            evidence=0.
            for s in siblings:
                if int(s)==int(position):continue
                related=int(rows[q,s])
                if (targets.house[candidate] and targets.house[related]
                        and targets.house[candidate]!=targets.house[related]):
                    continue
                if (targets.postcode[candidate] and targets.postcode[related]
                        and targets.postcode[candidate]!=targets.postcode[related]):
                    continue
                name_sim=fuzz.token_set_ratio(name,targets.names[related])/100 if name and targets.names[related] else 0.
                address_sim=fuzz.token_set_ratio(address,targets.addresses[related])/100 if address and targets.addresses[related] else 0.
                if min(name_sim,address_sim)<.5 or max(name_sim,address_sim)<.85:
                    continue
                evidence=max(evidence,.5*(name_sim+address_sim))
            if evidence:
                result[q,position]=min(1.,result[q,position]+delta*evidence)
    return result


def _metric(probs,labels,mask,true_count,threshold):
    picked=(probs>=threshold)&mask
    tp=(picked&(labels>0)).sum(1);counts=picked.sum(1)
    return {"macro_f05":macro_f05(true_count,counts,tp),
            "micro_precision":float(tp.sum()/max(1,counts.sum())),
            "false_positive_links":int(counts.sum()-tp.sum())}


def tune_collective(root):
    root=Path(root)
    with np.load(root/"validation_candidates.npz") as values:
        rows=values["target_rows"];mask=values["mask"];labels=values["labels"]
        true_count=values["true_counts"];scores=values["selected"]
        ranker=values["ranker"]
    calibration=json.loads((root/"calibration.json").read_text())
    split=json.loads((root/"split.json").read_text())
    threshold=calibration["threshold"]
    if calibration.get("ownership_policy","none")!="none":
        scope,rule=calibration["ownership_policy"].split(":",1)
        with np.load(root/"ownership_train.npz") as data:
            if scope=="final":
                from .reciprocal import owners_from_arrays
                best,owner,second=owners_from_arrays(rows,mask,scores,split["fit_end"],len(data["best"]))
                prior=scores
            else:
                best,owner,second=data["best"],data["owner"],data["second"]
                prior=ranker
            scores=policy_scores(scores,rows,mask,prior,split["fit_end"],best,owner,second,rule)
    tail_path=root/"tail_validation_overrides.npz"
    tail_config=json.loads((root/"tail_config.json").read_text()) if (root/"tail_config.json").exists() else {"active":False}
    if tail_config.get("active") and tail_path.exists():
        scores=scores.copy()
        with np.load(tail_path) as data:
            for (q,p),value in zip(data["pairs"],data["probabilities"]):
                scores[int(q),int(p)]=value
    targets=Corpus.load(root/"train_targets.arrow")
    tune_count=split["tune_end"]-split["fit_end"]
    trials=[]
    for delta in (0.,.05,.1,.15):
        changed=propagate_batch(scores,rows,mask,targets,threshold,delta)
        tune=_metric(changed[:tune_count],labels[:tune_count],mask[:tune_count],true_count[:tune_count],threshold)
        holdout=_metric(changed[tune_count:],labels[tune_count:],mask[tune_count:],true_count[tune_count:],threshold)
        trials.append({"delta":delta,"tune":tune,"holdout":holdout})
    baseline=trials[0]["tune"]
    viable=[trial for trial in trials if trial["tune"]["micro_precision"]>=baseline["micro_precision"]-.001]
    best=max(viable,key=lambda trial:(trial["tune"]["macro_f05"],-trial["delta"]))
    enabled=best["delta"]>0 and best["tune"]["macro_f05"]>baseline["macro_f05"]+.0002
    selected=best["delta"] if enabled else 0.
    report={"selected_delta":selected,"active":enabled,"trials":trials,
            "note":"Only candidates already scored by the final matcher receive one sibling pass"}
    (root/"collective_config.json").write_text(json.dumps(report,indent=2))
    validation=json.loads((root/"validation.json").read_text())
    for stage in validation["ablation_table"]:
        if stage["stage"]=="collective_propagation":stage.update(status="measured",result=report)
    validation["collective_propagation"]=report
    if enabled:
        validation["holdout_before_collective"]=validation["holdout_macro_f05"]
        validation["holdout_macro_f05"]=best["holdout"]["macro_f05"]
        validation["micro_precision"]=best["holdout"]["micro_precision"]
        validation["holdout_false_positive_links"]=best["holdout"]["false_positive_links"]
    (root/"validation.json").write_text(json.dumps(validation,indent=2))
    return report
