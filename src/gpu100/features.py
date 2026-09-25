"""Full candidate features, scored in C++ batches after GPU prefiltering."""
import numpy as np
from rapidfuzz import process,fuzz
from .cheap import CHEAP_NAMES
from .noisy_channel import CHANNEL_FEATURE_NAMES

BASE_FEATURE_NAMES=CHEAP_NAMES+("name_ratio","name_token_set","name_partial",
                                 "address_ratio","address_token_set","address_partial",
                                 "stage1_probability","anchor_expanded")
FEATURE_NAMES=BASE_FEATURE_NAMES+CHANNEL_FEATURE_NAMES


def enrich(queries,targets,query_rows,target_rows,offsets,cheap_features,stage1_scores,threads=4,anchor_flags=None,channel=None):
    n=len(target_rows)
    if not n:
        return np.empty((0,len(FEATURE_NAMES)),dtype=np.float32)
    left=np.repeat(np.asarray(query_rows,dtype=np.uint32),np.diff(offsets))
    right=np.asarray(target_rows,dtype=np.uint32)
    qn=[queries.names[int(i)] for i in left];tn=[targets.names[int(i)] for i in right]
    qa=[queries.addresses[int(i)] for i in left];ta=[targets.addresses[int(i)] for i in right]
    def compare(first,second,scorer):
        return process.cpdist(first,second,scorer=scorer,workers=threads,dtype=np.uint8).astype(np.float32)/100
    name_ratio=compare(qn,tn,fuzz.ratio)
    name_set=compare(qn,tn,fuzz.token_set_ratio)
    name_partial=compare(qn,tn,fuzz.partial_ratio)
    address_ratio=compare(qa,ta,fuzz.ratio)
    address_set=compare(qa,ta,fuzz.token_set_ratio)
    address_partial=compare(qa,ta,fuzz.partial_ratio)
    for values,first,second in ((name_ratio,qn,tn),(name_set,qn,tn),(name_partial,qn,tn),
                                (address_ratio,qa,ta),(address_set,qa,ta),(address_partial,qa,ta)):
        missing=np.fromiter((not a or not b for a,b in zip(first,second)),dtype=bool,count=n)
        values[missing]=0.
    base=np.column_stack((cheap_features,name_ratio,name_set,name_partial,
                       address_ratio,address_set,address_partial,stage1_scores,
                       np.zeros(n,dtype=np.float32) if anchor_flags is None else anchor_flags)).astype(np.float32)
    channel_scores=(channel.score_batch(queries,targets,query_rows,target_rows,offsets,
                    stage1_scores,np.zeros(n,dtype=np.float32) if anchor_flags is None else anchor_flags)
                    if channel is not None else np.zeros((n,len(CHANNEL_FEATURE_NAMES)),dtype=np.float32))
    x=np.column_stack((base,channel_scores)).astype(np.float32)
    assert x.shape==(n,len(FEATURE_NAMES))
    return x
