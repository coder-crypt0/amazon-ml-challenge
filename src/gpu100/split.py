"""Entity-disjoint reference split anchored to the original 60k notebook run."""
import random
import numpy as np


def choose_reference_rows(total,samples,seed):
    if samples>total:
        raise ValueError("Sample size exceeds reference set")
    if samples<60000:
        chosen=np.random.default_rng(seed).choice(total,size=samples,replace=False).astype(np.uint32)
        fit_end=int(samples*.7);tune_end=int(samples*.85)
        return chosen,{"fit_end":fit_end,"tune_end":tune_end,"holdout_end":samples,
                       "method":"entity-disjoint random sample"}
    rng=random.Random(seed)
    original=list(range(60000))
    for i in range(60000,total):
        j=rng.randrange(i+1)
        if j<60000:
            original[j]=i
    rng.shuffle(original)
    extra_count=samples-60000
    original_set=set(original)
    remaining=np.fromiter((i for i in range(total) if i not in original_set),dtype=np.uint32,
                          count=total-60000)
    extra=np.random.default_rng(seed+1).choice(remaining,size=extra_count,replace=False)
    chosen=np.asarray([*original[:42000],*extra,*original[42000:]],dtype=np.uint32)
    fit_end=42000+extra_count
    return chosen,{"fit_end":fit_end,"tune_end":fit_end+9000,"holdout_end":samples,
                   "method":"original 60k reservoir split; additional fit-only entities",
                   "original_fit":42000,"original_tune":9000,"original_holdout":9000,
                   "extra_fit":extra_count}
