"""Fast, leakage-safe decoder audit for the saved Kaggle validation arrays."""
import argparse
import numpy as np


def macro_f05(predicted, labels, truth):
    tp=np.sum(predicted & labels,axis=1)
    count=np.sum(predicted,axis=1)
    score=np.zeros(len(truth),dtype=np.float64)
    singleton=truth==0
    score[singleton]=(count[singleton]==0)
    linked=~singleton
    score[linked]=1.25*tp[linked]/np.maximum(1,0.25*truth[linked]+count[linked])
    return float(score.mean())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("input")
    args=parser.parse_args()
    with np.load(args.input) as data:
        labels=data["labels"].astype(bool)
        mask=data["mask"].astype(bool)
        truth=data["true_counts"]
        models={key:data[key].astype(np.float32) for key in ("selected","rich_channel","ranker","tabm_hard")}
    tune=slice(0,9000);hold=slice(9000,None)
    print("Oracle candidate ceiling",macro_f05(labels&mask,labels,truth))
    for name,scores in models.items():
        best=None
        for threshold in np.arange(.20,.951,.025):
            pred=(scores>=threshold)&mask
            result=(macro_f05(pred[tune],labels[tune],truth[tune]),threshold,
                    macro_f05(pred[hold],labels[hold],truth[hold]))
            if best is None or result[0]>best[0]:best=result
        print(name,"global",best)
    scores=models["selected"]
    masked=np.where(mask,scores,-1)
    ranked=np.argsort(-masked,axis=1,kind="stable")
    rank=np.empty_like(ranked)
    np.put_along_axis(rank,ranked,np.arange(scores.shape[1])[None,:],axis=1)
    top=np.max(masked,axis=1)
    oracle_count=(rank<truth[:,None])&mask
    print("Oracle cardinality with learned ranking",macro_f05(oracle_count,labels,truth))
    for k in (1,2,3,4,5,6):
        pred=(rank<k)&mask
        print("Fixed top",k,"tune",macro_f05(pred[tune],labels[tune],truth[tune]),
              "holdout",macro_f05(pred[hold],labels[hold],truth[hold]))
    for k in (8,16,32,64,128):
        eligible=(rank<k)&mask
        tp=(eligible&labels).sum()
        print("Top",k,"positive recall",float(tp/np.maximum(1,truth.sum())))
    trials=[]
    for base in (.55,.60,.65,.675,.70,.75):
        for floor in (.20,.35,.50,.60):
            for count in (2,3,4):
                for gate in (.80,.90,.95):
                    pred=((scores>=base)|((rank<count)&(scores>=floor)&(top[:,None]>=gate)))&mask
                    trials.append((macro_f05(pred[tune],labels[tune],truth[tune]),
                                   (base,floor,count,gate),macro_f05(pred[hold],labels[hold],truth[hold])))
    trials.sort(reverse=True)
    print("Top floor-count decoders by tune")
    for row in trials[:10]:print(row)
    print("Best holdout among these (audit only)",max(trials,key=lambda row:row[2]))


if __name__=="__main__":main()
