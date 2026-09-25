"""Decision rules for precision-oriented entity resolution (ablation module)."""
from __future__ import annotations
import numpy as np

def _dist(ps):
    d=np.array([1.0])
    for p in ps:
        n=np.zeros(len(d)+1); n[:-1]+=d*(1-p); n[1:]+=d*p; d=n
    return d

def expected_f05_prefix(probabilities, hidden_unmatched=0.0):
    """Return expected F0.5 for every probability-sorted prefix, plus empty.

    Candidate probabilities are interpreted as independent match events.  The
    optional ``hidden_unmatched`` is the probability mass of an ungenerated
    true match and is treated as one additional Bernoulli true entity.
    """
    ps=np.asarray(probabilities,dtype=float).clip(0,1)
    n=len(ps); out=[float(np.prod(1-ps)*(1-hidden_unmatched))]
    if n==0:return np.asarray(out)
    # Leave-one-out distributions give exact expected utility in O(K^3).
    for m in range(1,n+1):
        val=0.0
        for i,p in enumerate(ps[:m]):
            q=np.delete(ps,i); di=_dist(q)
            # hidden unmatched increases true-count denominator only when it exists
            term=sum(prob/(m+0.25*(1+k)) for k,prob in enumerate(di))
            if hidden_unmatched:
                h=float(np.clip(hidden_unmatched,0,1))
                term=(1-h)*term+h*sum(prob/(m+0.25*(2+k)) for k,prob in enumerate(di))
            val += p*term
        out.append(1.25*val)
    return np.asarray(out)

def choose_prefix(probabilities, hidden_unmatched=0.0):
    """Return (number_selected, expected_score), preserving input order."""
    ps=np.asarray(probabilities,dtype=float)
    order=np.argsort(-ps, kind="stable"); scores=expected_f05_prefix(ps[order],hidden_unmatched)
    k=int(np.argmax(scores)); return k,float(scores[k])

def resolve_exclusive(pairs, probabilities, margin=0.0):
    """Keep at most one query pair per target, selecting strongest calibrated edge.

    ``pairs`` are [query_id,target_id] rows. Ties within margin are rejected to
    avoid arbitrary ownership; this is intentionally separate from calibration.
    """
    pairs=np.asarray(pairs); probs=np.asarray(probabilities,float)
    best={}; groups={}
    for i,(pair,p) in enumerate(zip(pairs,probs)):
        t=pair[1].item() if hasattr(pair[1],'item') else pair[1]
        groups.setdefault(t,[]).append((i,float(p)))
        if t not in best or p>best[t][1]: best[t]=(i,float(p))
    keep=[]
    for t,(i,p) in best.items():
        rivals=[x for j,x in groups[t] if j!=i]
        if not rivals or p-max(rivals)>margin: keep.append(i)
    return np.asarray(sorted(keep),dtype=int)
