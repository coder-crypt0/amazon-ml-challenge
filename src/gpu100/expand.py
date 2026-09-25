"""Bounded second-hop retrieval from high-confidence target anchors."""
import numpy as np
from .retrieval import retrieve_batch
from .cheap import flatten_retrieval
from .features import FEATURE_NAMES,enrich


def expand(queries,targets,qrows,direct_rows,direct_offsets,direct_cheap,direct_probability,
           views,threads=4,anchor_count=2,anchor_threshold=.8,extra_limit=32,channel=None,
           name_k=8,address_k=16):
    """Return direct + second-hop pairs in query order and the exact rich features scored."""
    qrows=np.asarray(qrows,dtype=np.uint32)
    anchors=[];owners=[];confidence=[]
    for q,(start,end) in enumerate(zip(direct_offsets[:-1],direct_offsets[1:])):
        for pos in range(int(start),min(int(end),int(start)+anchor_count)):
            p=float(direct_probability[pos])
            if p>=anchor_threshold:
                anchors.append(int(direct_rows[pos]));owners.append(q);confidence.append(p)
    extras=[{} for _ in qrows]
    if anchors:
        found=retrieve_batch(targets,np.asarray(anchors,dtype=np.uint32),views,name_k,address_k,threads)
        direct_sets=[set(int(r) for r in direct_rows[int(s):int(e)]) for s,e in zip(direct_offsets[:-1],direct_offsets[1:])]
        for anchor_idx,neighbours in enumerate(found):
            q=owners[anchor_idx]
            for row,pair in neighbours.items():
                if row in direct_sets[q]:
                    continue
                proxy=[confidence[anchor_idx]*pair[0],confidence[anchor_idx]*pair[1]]
                current=extras[q].get(row)
                if current is None or max(proxy)>max(current):
                    extras[q][row]=proxy
    trimmed=[]
    for options in extras:
        best=sorted(options.items(),key=lambda item:(-max(item[1]),-sum(item[1]),item[0]))[:extra_limit]
        trimmed.append(dict(best))
    extra_cheap,extra_rows,extra_offsets=flatten_retrieval(queries,targets,qrows,trimmed)
    extra_prior=np.asarray([max(pair) for options in trimmed for pair in options.values()],dtype=np.float32)
    direct_rich=enrich(queries,targets,qrows,direct_rows,direct_offsets,direct_cheap,
                       direct_probability,threads,channel=channel)
    extra_rich=enrich(queries,targets,qrows,extra_rows,extra_offsets,extra_cheap,
                      extra_prior,threads,anchor_flags=np.ones(len(extra_rows),dtype=np.float32),channel=channel)
    total=len(direct_rows)+len(extra_rows)
    rows=np.empty(total,dtype=np.uint32)
    x=np.empty((total,len(FEATURE_NAMES)),dtype=np.float32)
    offset=np.zeros(len(qrows)+1,dtype=np.int64)
    pos=0
    for q in range(len(qrows)):
        ds,de=map(int,direct_offsets[q:q+2])
        es,ee=map(int,extra_offsets[q:q+2])
        first=de-ds;second=ee-es
        rows[pos:pos+first]=direct_rows[ds:de]
        rows[pos+first:pos+first+second]=extra_rows[es:ee]
        x[pos:pos+first]=direct_rich[ds:de]
        x[pos+first:pos+first+second]=extra_rich[es:ee]
        pos+=first+second
        offset[q+1]=pos
    return rows,offset,x
