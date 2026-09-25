"""Test sibling-aware reranking on the frozen CNER tune/holdout entities."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz
from xgboost import XGBClassifier
from gpu100.corpus import name_text,address_text


def score(pred,labels,truth):
    tp=(pred&labels).sum(1);count=pred.sum(1)
    values=np.where(truth==0,(count==0).astype(float),
                    1.25*tp/np.maximum(1,.25*truth+count))
    return float(values.mean())


def extract(path,needed,offset=0):
    result={}
    with Path(path).open(encoding="utf-8",newline="") as handle:
        next(handle)
        for i,line in enumerate(handle):
            index=offset+i
            if index not in needed:continue
            pieces=line.rstrip("\r\n").split("\t")
            if len(pieces)!=4:raise ValueError((path,i,len(pieces)))
            result[index]=(name_text(pieces[1]),address_text(pieces[2]))
    return result,i+1


def build_features(data,top=16):
    mask=data["mask"].astype(bool)
    scores=data["selected"].astype(np.float32)
    order=np.argsort(-np.where(mask,scores,-1),axis=1,kind="stable")[:,:top]
    rows=np.take_along_axis(data["target_rows"],order,axis=1)
    labels=np.take_along_axis(data["labels"],order,axis=1).astype(bool)
    active=np.take_along_axis(mask,order,axis=1)
    needed=set(rows[active].astype(int))
    base=Path("student_resource/dataset/train")
    targets,n2=extract(base/"train_source2.tsv",needed)
    second,_=extract(base/"train_source3.tsv",needed,n2)
    targets.update(second)
    print("Loaded candidate target records",len(targets),flush=True)
    needed_queries=set(map(int,data["query_rows"]))
    queries,_=extract(base/"train_source1.tsv",needed_queries)
    print("Loaded validation references",len(queries),flush=True)
    channels=[np.take_along_axis(data[key],order,axis=1).astype(np.float32)
              for key in ("selected","rich_channel","tabm_hard","ranker")]
    n=len(rows)
    features=np.zeros((n,top,24),dtype=np.float32)
    for q in range(n):
        if q%3000==0:print("Features",q,"/",n,flush=True)
        qname,qaddr=queries[int(data["query_rows"][q])]
        qnumbers=set(re.findall(r"\d+",qaddr))
        anchor_names=[];anchor_addrs=[]
        for j in range(min(3,top)):
            if active[q,j]:
                a,b=targets[int(rows[q,j])]
                anchor_names.append(a);anchor_addrs.append(b)
        top_score=channels[0][q,0]
        neighbors=[targets[int(rows[q,k])] if active[q,k] else ("","") for k in range(top)]
        for j in range(top):
            if not active[q,j]:continue
            name,addr=targets[int(rows[q,j])]
            f=features[q,j]
            f[:4]=[item[q,j] for item in channels]
            f[4]=j/top
            f[5]=top_score-channels[0][q,j]
            f[6]=fuzz.ratio(qname,name)/100 if qname and name else 0
            f[7]=fuzz.ratio(qaddr,addr)/100 if qaddr and addr else 0
            f[8]=fuzz.token_set_ratio(qname,name)/100 if qname and name else 0
            f[9]=fuzz.token_set_ratio(qaddr,addr)/100 if qaddr and addr else 0
            f[10]=max((fuzz.ratio(name,a)/100 for a in anchor_names if a),default=0)
            f[11]=max((fuzz.ratio(addr,a)/100 for a in anchor_addrs if a),default=0)
            f[12]=max((fuzz.token_set_ratio(name,a)/100 for a in anchor_names if a),default=0)
            f[13]=max((fuzz.token_set_ratio(addr,a)/100 for a in anchor_addrs if a),default=0)
            f[14]=1.0 if int(rows[q,j])>=n2 else 0.0
            f[15]=min(len(qname),len(name))/max(1,len(qname),len(name))
            f[16]=min(len(qaddr),len(addr))/max(1,len(qaddr),len(addr))
            f[17]=float(bool(qnumbers&set(re.findall(r"\d+",addr))))
            f[18]=fuzz.token_sort_ratio(qname,name)/100 if qname and name else 0
            f[19]=fuzz.token_sort_ratio(qaddr,addr)/100 if qaddr and addr else 0
            other=[(fuzz.ratio(name,a)/100 if name and a else 0,
                    fuzz.ratio(addr,b)/100 if addr and b else 0)
                   for k,(a,b) in enumerate(neighbors) if k!=j and active[q,k]
                   and channels[0][q,k]>=.45]
            if other:
                pairs=np.asarray(other,dtype=np.float32)
                f[20]=max(pairs[:,0])
                f[21]=max(pairs[:,1])
                f[22]=max(.5*(pairs[:,0]+pairs[:,1]))
                f[23]=np.sum((pairs[:,0]>=.75)&(pairs[:,1]>=.75))/top
    return features,labels,active


def tune_threshold(prob,labels,active,truth,start,stop):
    best=(-1,None)
    for threshold in np.arange(.10,.951,.025):
        value=score((prob[start:stop]>=threshold)&active[start:stop],
                    labels[start:stop],truth[start:stop])
        if value>best[0]:best=(value,threshold)
    return best


def main():
    p=argparse.ArgumentParser();p.add_argument("input");args=p.parse_args()
    with np.load(args.input) as data:
        features,labels,active=build_features(data)
        truth=data["true_counts"]
    for end in (10,14,24):
        X=features[:6000,:,:end][active[:6000]]
        y=labels[:6000][active[:6000]]
        model=XGBClassifier(n_estimators=350,max_depth=6,learning_rate=.06,
                            min_child_weight=10,subsample=.8,colsample_bytree=.9,
                            tree_method="hist",n_jobs=4,eval_metric="logloss")
        model.fit(X,y)
        prob=np.zeros(active.shape,dtype=np.float32)
        for start in range(6000,len(active),1000):
            stop=min(len(active),start+1000)
            prob[start:stop][active[start:stop]]=model.predict_proba(
                features[start:stop,:,:end][active[start:stop]])[:,1]
        tune,threshold=tune_threshold(prob,labels,active,truth,6000,9000)
        hold=score((prob[9000:]>=threshold)&active[9000:],labels[9000:],truth[9000:])
        print(json.dumps({"features":end,"tune_macro_f05":tune,"threshold":threshold,
                          "holdout_macro_f05":hold}),flush=True)
    base=features[:,:,0]
    tune,threshold=tune_threshold(base,labels,active,truth,6000,9000)
    hold=score((base[9000:]>=threshold)&active[9000:],labels[9000:],truth[9000:])
    print(json.dumps({"baseline_top16":True,"tune_macro_f05":tune,"threshold":threshold,
                      "holdout_macro_f05":hold}),flush=True)


if __name__=="__main__":main()
