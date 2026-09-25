"""Fast prefilter features for the GPU candidate ranker."""
import numpy as np
from rapidfuzz import process,fuzz

CHEAP_NAMES=("name_cos","address_cos","best_cos","both_cos","cos_product",
             "name_len_ratio","address_len_ratio","name_exact","address_exact",
             "house_agree","house_conflict","postcode_agree","postcode_conflict",
             "query_name_missing","target_name_missing","query_address_missing","target_address_missing",
             "source3","name_fuzzy_ratio","address_fuzzy_token_set")


def flatten_retrieval(queries,targets,query_rows,scored,threads=4):
    offsets=[0];target_rows=[];name_score=[];addr_score=[]
    for candidates in scored:
        for target,score in candidates.items():
            target_rows.append(target);name_score.append(score[0]);addr_score.append(score[1])
        offsets.append(len(target_rows))
    rows=np.asarray(target_rows,dtype=np.uint32)
    offsets=np.asarray(offsets,dtype=np.int64)
    if not len(rows):
        return np.empty((0,len(CHEAP_NAMES)),dtype=np.float32),rows,offsets
    qidx=np.repeat(np.asarray(query_rows,dtype=np.uint32),np.diff(offsets))
    qnames=[queries.names[int(i)] for i in qidx];tnames=[targets.names[int(i)] for i in rows]
    qaddr=[queries.addresses[int(i)] for i in qidx];taddr=[targets.addresses[int(i)] for i in rows]
    name_fuzzy=process.cpdist(qnames,tnames,scorer=fuzz.ratio,workers=threads,dtype=np.uint8).astype(np.float32)/100
    address_fuzzy=process.cpdist(qaddr,taddr,scorer=fuzz.token_set_ratio,workers=threads,dtype=np.uint8).astype(np.float32)/100
    n=np.asarray(name_score,dtype=np.float32);a=np.asarray(addr_score,dtype=np.float32)
    qnl=queries.name_len[qidx];tnl=targets.name_len[rows]
    qal=queries.address_len[qidx];tal=targets.address_len[rows]
    qhouse=queries.house[qidx];thouse=targets.house[rows]
    qpost=queries.postcode[qidx];tpost=targets.postcode[rows]
    bothhouse=(qhouse!=0)&(thouse!=0)
    bothpost=(qpost!=0)&(tpost!=0)
    x=np.column_stack((n,a,np.maximum(n,a),np.minimum(n,a),n*a,
        np.minimum(qnl,tnl)/np.maximum(1,np.maximum(qnl,tnl)),
        np.minimum(qal,tal)/np.maximum(1,np.maximum(qal,tal)),
        (queries.name_hash[qidx]!=0)&(queries.name_hash[qidx]==targets.name_hash[rows]),
        (queries.address_hash[qidx]!=0)&(queries.address_hash[qidx]==targets.address_hash[rows]),
        bothhouse&(qhouse==thouse),bothhouse&(qhouse!=thouse),
        bothpost&(qpost==tpost),bothpost&(qpost!=tpost),
        qnl==0,tnl==0,qal==0,tal==0,targets.source[rows]==3,
        name_fuzzy,address_fuzzy)).astype(np.float32)
    return x,rows,offsets


def select_top_rows(scores,rows,offsets,k):
    """Return selected candidate rows and parent indices, sorted by score per reference."""
    selected=[];new_offsets=[0]
    for start,end in zip(offsets[:-1],offsets[1:]):
        start,end=int(start),int(end)
        if end>start:
            segment=np.arange(start,end)
            if len(segment)>k:
                segment=segment[np.argpartition(scores[start:end],-k)[-k:]]
            segment=segment[np.argsort(-scores[segment],kind="stable")]
            selected.extend(segment.tolist())
        new_offsets.append(len(selected))
    selected=np.asarray(selected,dtype=np.int64)
    return rows[selected],selected,np.asarray(new_offsets,dtype=np.int64)


def predict_ranker(booster,features,device="cpu"):
    if not len(features):
        return np.empty(0,dtype=np.float32)
    if device.startswith("cuda"):
        import cupy as cp
        return cp.asnumpy(booster.inplace_predict(cp.asarray(features)))
    return booster.inplace_predict(features)
