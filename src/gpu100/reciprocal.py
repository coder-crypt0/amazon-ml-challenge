"""Reverse ownership ranking over all generated S1-to-target candidate edges."""
from pathlib import Path
import numpy as np
from .setmodel import macro_f05


def owners_from_final_chunks(paths,n_targets,score_key="ranker_prob"):
    best=np.full(n_targets,-1.,dtype=np.float32)
    owner=np.full(n_targets,np.iinfo(np.int32).max,dtype=np.int32)
    second=np.full(n_targets,-1.,dtype=np.float32)
    for path in paths:
        with np.load(path) as part:
            np.maximum.at(best,part["rows"],part[score_key])
    query_offset=0
    for path in paths:
        with np.load(path) as part:
            rows=part["rows"]
            count=len(part["qrows"]) if "qrows" in part.files else len(part["offsets"])-1
            qids=query_offset+np.repeat(np.arange(count,dtype=np.int32),np.diff(part["offsets"]))
            values=part[score_key]
            winner=(values>=best[rows]-1e-7)
            np.minimum.at(owner,rows[winner],qids[winner])
            query_offset+=count
    query_offset=0
    for path in paths:
        with np.load(path) as part:
            rows=part["rows"]
            count=len(part["qrows"]) if "qrows" in part.files else len(part["offsets"])-1
            qids=query_offset+np.repeat(np.arange(count,dtype=np.int32),np.diff(part["offsets"]))
            rivals=qids!=owner[rows]
            np.maximum.at(second,rows[rivals],part[score_key][rivals])
            query_offset+=count
    return best,owner,second


def owners_from_arrays(rows,mask,scores,query_offset,n_targets):
    valid=np.asarray(mask,dtype=bool)
    target=rows[valid]
    values=scores[valid]
    qids=np.broadcast_to(np.arange(query_offset,query_offset+len(rows),dtype=np.int32)[:,None],rows.shape)[valid]
    best=np.full(n_targets,-1.,dtype=np.float32)
    owner=np.full(n_targets,np.iinfo(np.int32).max,dtype=np.int32)
    second=np.full(n_targets,-1.,dtype=np.float32)
    np.maximum.at(best,target,values)
    winners=values>=best[target]-1e-7
    np.minimum.at(owner,target[winners],qids[winners])
    rivals=qids!=owner[target]
    np.maximum.at(second,target[rivals],values[rivals])
    return best,owner,second


def reverse_features(rows,mask,stage1_scores,query_offset,best,owner,second):
    result=np.zeros((*rows.shape,4),dtype=np.float32)
    valid=np.asarray(mask,dtype=bool)
    target=rows[valid]
    qids=np.broadcast_to(np.arange(query_offset,query_offset+len(rows),dtype=np.int32)[:,None],rows.shape)[valid]
    current=stage1_scores[valid]
    champion=owner[target]==qids
    competitor=np.where(champion,second[target],best[target])
    reverse_rank=np.where(champion,1.,np.where(current>=second[target]-1e-7,2.,3.))
    forward_rank=np.broadcast_to(np.arange(rows.shape[1])[None,:],rows.shape)[valid]
    values=np.stack((reverse_rank,(forward_rank<16)&(reverse_rank<=2),
                     np.maximum(0,competitor),current-np.maximum(0,competitor)),axis=1)
    result[valid]=values
    return result


def policy_scores(probabilities,rows,mask,stage1_scores,query_offset,best,owner,second,policy):
    if policy=="none":return probabilities
    valid=np.asarray(mask,dtype=bool)
    target=rows[valid]
    qids=np.broadcast_to(np.arange(query_offset,query_offset+len(rows),dtype=np.int32)[:,None],rows.shape)[valid]
    champion=owner[target]==qids
    if policy=="owner":
        keep=champion
    elif policy.startswith("gap:"):
        gap=float(policy.partition(":")[2])
        keep=champion & (best[target]-second[target]>=gap)
    else:
        raise ValueError(policy)
    result=probabilities.copy()
    result[valid]=np.where(keep,probabilities[valid],0.)
    return result


def tuning_result(probabilities,rows,mask,labels,true_counts,stage1_scores,fit_end,tune_end,
                  threshold,best,owner,second):
    policies=("none","owner","gap:0.01","gap:0.03","gap:0.05")
    trials=[]
    for policy in policies:
        scores=policy_scores(probabilities,rows,mask,stage1_scores,fit_end,best,owner,second,policy)
        values=[]
        for begin,end in ((0,tune_end-fit_end),(tune_end-fit_end,len(scores))):
            selected=(scores[begin:end]>=threshold)&mask[begin:end]
            tp=(selected&(labels[begin:end]>0)).sum(1)
            predicted=selected.sum(1)
            values.append(macro_f05(true_counts[begin:end],predicted,tp))
        trials.append({"policy":policy,"tune_macro_f05":values[0],"holdout_macro_f05":values[1]})
    baseline=trials[0]["tune_macro_f05"]
    winner=max(trials,key=lambda entry:(entry["tune_macro_f05"],entry["policy"]=="none"))
    selected=winner["policy"] if winner["tune_macro_f05"]>baseline+.0002 else "none"
    return {"selected":selected,"trials":trials,"reverse_scope":"competing S1 edges in the generated candidate graph"}
