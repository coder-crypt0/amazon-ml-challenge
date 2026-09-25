"""Apache-2.0 multilingual cross-encoder for a bounded ambiguous tail."""
import json
import os
from pathlib import Path
import numpy as np
from .corpus import Corpus
from .setmodel import macro_f05
from .reciprocal import policy_scores

MODEL_ID="Alibaba-NLP/gte-multilingual-reranker-base"
MODEL_REVISION="8215cf04918ba6f7b6a62bb44238ce2953d8831c"
CODE_REVISION="40ced75c3017eb27626c9d4ea981bde21a2662f4"
TAIL_FRACTION=.03
MAX_PAIRS_PER_QUERY=3
os.environ.setdefault("USE_TF","0")
os.environ.setdefault("USE_FLAX","0")


def _render(corpus,index):
    index=int(index)
    return f"Business: {corpus.raw_names[index]}\nAddress: {corpus.raw_addresses[index]}\nCountry: {corpus.countries[index]}"


def ambiguous_queries(probabilities,mask,threshold,fraction=TAIL_FRACTION):
    n=len(probabilities)
    if not n:return np.empty(0,dtype=np.int32)
    scores=np.where(mask,probabilities,-1.)
    top=np.sort(scores,axis=1)[:,-2:]
    nearest=np.minimum(np.abs(top[:,-1]-threshold),np.abs(top[:,0]-threshold))
    rivalry=np.maximum(0.,top[:,-1]-top[:,0])
    uncertainty=nearest+.2*rivalry
    count=max(1,min(n,int(round(n*fraction))))
    return np.sort(np.argpartition(uncertainty,count-1)[:count]).astype(np.int32)


def tail_pairs(probabilities,mask,threshold,query_indices,max_pairs=MAX_PAIRS_PER_QUERY):
    pairs=[]
    for query in query_indices:
        query=int(query)
        positions=np.flatnonzero(mask[query])
        if len(positions):
            best=positions[np.argsort(np.abs(probabilities[query,positions]-threshold),kind="stable")[:max_pairs]]
            pairs.extend((query,int(position)) for position in best)
    return pairs


def _load_model(model_path=None,device="cuda"):
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    path=str(model_path) if model_path is not None else MODEL_ID
    extra={"revision":MODEL_REVISION} if model_path is None else {}
    tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=True,**extra)
    model=AutoModelForSequenceClassification.from_pretrained(path,trust_remote_code=True,
                                                               dtype=torch.float32,
                                                               code_revision=CODE_REVISION,**extra).to(device)
    return tokenizer,model


def _training_examples(root,max_queries=4000):
    root=Path(root)
    split=json.loads((root/"split.json").read_text())
    fit_end=split["fit_end"]
    examples=[];global_q=0
    for path in sorted((root/"train_final").glob("[0-9]*.npz")):
        with np.load(path) as part:
            for local,(start,end) in enumerate(zip(part["offsets"][:-1],part["offsets"][1:])):
                if global_q+local>=fit_end:break
                labels=part["y"][start:end]
                positives=np.flatnonzero(labels>0);negatives=np.flatnonzero(labels==0)
                if not len(positives) or not len(negatives):continue
                probabilities=part["ranker_prob"][start:end]
                hard_positive=positives[np.argmin(probabilities[positives])]
                hard_negative=negatives[np.argmax(probabilities[negatives])]
                examples.append((int(part["qrows"][local]),int(part["rows"][start+hard_positive]),
                                 int(part["rows"][start+hard_negative])))
                if len(examples)>=max_queries:break
            global_q+=len(part["qrows"])
        if len(examples)>=max_queries:break
    if not examples:raise ValueError("No fit-positive/hard-negative pairs available for cross-encoder")
    return np.asarray(examples,dtype=np.int32)


def fine_tune(root,device="cuda",max_queries=4000,epochs=1,batch_queries=4):
    import torch
    import torch.nn.functional as F
    root=Path(root)
    output=root/"tail_model"
    if (output/"COMPLETE").exists():
        return output
    output.mkdir(parents=True,exist_ok=True)
    examples=_training_examples(root,max_queries)
    np.save(output/"training_pairs.npy",examples)
    queries=Corpus.load(root/"train_queries.arrow")
    targets=Corpus.load(root/"train_targets.arrow")
    tokenizer,model=_load_model(None,device)
    if hasattr(model,"gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    scaler=torch.amp.GradScaler(device,enabled=device.startswith("cuda"))
    for epoch in range(epochs):
        order=np.random.default_rng(20260925+epoch).permutation(len(examples))
        model.train();loss_sum=0.
        for begin in range(0,len(order),batch_queries):
            rows=examples[order[begin:begin+batch_queries]]
            rendered=[]
            for query,positive,negative in rows:
                source=_render(queries,query)
                rendered.extend(((source,_render(targets,positive)),(source,_render(targets,negative))))
            enc=tokenizer(rendered,padding=True,truncation=True,max_length=160,return_tensors="pt")
            enc={key:value.to(device) for key,value in enc.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0],enabled=device.startswith("cuda")):
                logits=model(**enc,return_dict=True).logits.reshape(-1).float()
                pos,neg=logits[::2],logits[1::2]
                labels=torch.tensor([1.,0.],device=device).repeat(len(rows))
                classification=F.binary_cross_entropy_with_logits(logits,labels)
                margin=F.relu(1.-pos+neg).mean()
                loss=classification+.3*margin
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update()
            loss_sum+=loss.detach().item()
        print(f"Tail reranker epoch {epoch+1}/{epochs}: loss={loss_sum/max(1,(len(examples)+batch_queries-1)//batch_queries):.4f}",flush=True)
    model.save_pretrained(output,safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output/"license.json").write_text(json.dumps({"base_model":MODEL_ID,"revision":MODEL_REVISION,
        "code_repository":"Alibaba-NLP/new-impl","code_revision":CODE_REVISION,
        "license":"apache-2.0","parameters":306000000,"fit_pairs":len(examples)*2},indent=2))
    (output/"COMPLETE").write_text("complete\n")
    return output


def score_pairs(model,tokenizer,queries,targets,pairs,device="cuda",batch_size=32):
    import torch
    output=np.zeros(len(pairs),dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(pairs),batch_size):
            block=pairs[start:start+batch_size]
            rendered=[(_render(queries,qrow),_render(targets,trow)) for qrow,trow in block]
            enc=tokenizer(rendered,padding=True,truncation=True,max_length=160,return_tensors="pt")
            enc={key:value.to(device) for key,value in enc.items()}
            with torch.autocast(device_type=device.split(":")[0],enabled=device.startswith("cuda")):
                logits=model(**enc,return_dict=True).logits.reshape(-1).float()
            output[start:start+len(block)]=torch.sigmoid(logits).cpu().numpy()
    return output


def _candidate_eval(probabilities,labels,mask,true_counts,threshold):
    pred=(probabilities>=threshold)&mask
    tp=(pred&(labels>0)).sum(1)
    predicted=pred.sum(1)
    return {"macro_f05":macro_f05(true_counts,predicted,tp),
            "micro_precision":float(tp.sum()/max(1,predicted.sum()))}


def tune_tail(root,device="cuda",fraction=TAIL_FRACTION):
    root=Path(root)
    with np.load(root/"validation_candidates.npz") as values:
        rows=values["target_rows"];qrows=values["query_rows"]
        scores=values["selected"];mask=values["mask"]
        labels=values["labels"];counts=values["true_counts"];ranker=values["ranker"]
    split=json.loads((root/"split.json").read_text())
    calibration=json.loads((root/"calibration.json").read_text())
    threshold=calibration["threshold"]
    if calibration.get("ownership_policy","none")!="none":
        with np.load(root/"ownership_train.npz") as data:
            scores=policy_scores(scores,rows,mask,ranker,split["fit_end"],
                       data["best"],data["owner"],data["second"],calibration["ownership_policy"])
    tune_end=split["tune_end"]-split["fit_end"]
    tokenizer,model=_load_model(root/"tail_model",device)
    queries=Corpus.load(root/"train_queries.arrow")
    targets=Corpus.load(root/"train_targets.arrow")
    proposal=scores.copy()
    scored_pairs=[]
    for begin,end in ((0,tune_end),(tune_end,len(scores))):
        indexes=ambiguous_queries(scores[begin:end],mask[begin:end],threshold,fraction)+begin
        scored_pairs.extend(tail_pairs(scores,mask,threshold,indexes))
    raw_pairs=[(int(qrows[q]),int(rows[q,position])) for q,position in scored_pairs]
    tail_prob=score_pairs(model,tokenizer,queries,targets,raw_pairs,device)
    base=_candidate_eval(scores[:tune_end],labels[:tune_end],mask[:tune_end],counts[:tune_end],threshold)
    trials=[{"weight":0.,"tune":base}]
    for weight in (.1,.25,.5,.75):
        proposal[:]=scores
        for (q,position),p in zip(scored_pairs,tail_prob):
            proposal[q,position]=(1-weight)*scores[q,position]+weight*p
        tune=_candidate_eval(proposal[:tune_end],labels[:tune_end],mask[:tune_end],counts[:tune_end],threshold)
        trials.append({"weight":weight,"tune":tune})
    viable=[item for item in trials if item["tune"]["micro_precision"]>=base["micro_precision"]-.002]
    winner=max(viable,key=lambda item:(item["tune"]["macro_f05"],-item["weight"]))
    active=winner["weight"]>0 and winner["tune"]["macro_f05"]>base["macro_f05"]+.0003
    weight=winner["weight"] if active else 0.
    proposal[:]=scores
    if active:
        for (q,position),p in zip(scored_pairs,tail_prob):
            proposal[q,position]=(1-weight)*scores[q,position]+weight*p
        np.savez_compressed(root/"tail_validation_overrides.npz",
                            pairs=np.asarray(scored_pairs,dtype=np.int32).reshape(-1,2),
                            probabilities=np.asarray([proposal[q,position] for q,position in scored_pairs],dtype=np.float32))
    held=_candidate_eval(proposal[tune_end:],labels[tune_end:],mask[tune_end:],counts[tune_end:],threshold)
    result={"active":active,"weight":weight,"fraction":fraction,"max_pairs_per_query":MAX_PAIRS_PER_QUERY,
            "model":MODEL_ID,"revision":MODEL_REVISION,"code_revision":CODE_REVISION,
            "license":"apache-2.0","parameters":306000000,
            "tune_trials":trials,"holdout_after_tail":held,"scored_pairs":len(scored_pairs)}
    (root/"tail_config.json").write_text(json.dumps(result,indent=2))
    report=json.loads((root/"validation.json").read_text())
    report["cross_encoder_tail"]=result
    for item in report["ablation_table"]:
        if item["stage"]=="cross_encoder_tail":item.update(status="measured",result=result)
    if active:
        report["holdout_before_tail"]=report["holdout_macro_f05"]
        report["holdout_macro_f05"]=held["macro_f05"]
        report["micro_precision"]=held["micro_precision"]
    (root/"validation.json").write_text(json.dumps(report,indent=2))
    return result


def apply_tail_test(root,device="cuda"):
    root=Path(root)
    config=json.loads((root/"tail_config.json").read_text())
    if not config["active"]:
        return {"active":False,"overrides":0}
    directory=root/"test_predictions"
    manifest=json.loads((directory/"manifest.json").read_text())
    if not (directory/"COMPLETE").is_file():raise ValueError("Test predictions must finish before tail reranking")
    uncertainties=np.empty(manifest["queries"],dtype=np.float32)
    offset=0
    threshold=json.loads((root/"calibration.json").read_text())["threshold"]
    for batch in range(manifest["batches"]):
        with np.load(directory/f"{batch:05d}.npz") as part:
            for start,end in zip(part["offsets"][:-1],part["offsets"][1:]):
                p=part["probabilities"][start:end]
                if len(p):
                    top=np.sort(p)[-2:]
                    delta=min(abs(float(value)-threshold) for value in top)
                    margin=float(top[-1]-top[0]) if len(top)>1 else 1.
                    uncertainties[offset]=delta+.2*margin
                else:
                    uncertainties[offset]=10.
                offset+=1
    count=max(1,int(round(manifest["queries"]*config["fraction"])))
    selected=set(np.argpartition(uncertainties,count-1)[:count].tolist())
    candidates=[];before=[]
    for batch in range(manifest["batches"]):
        with np.load(directory/f"{batch:05d}.npz") as part:
            for local,(start,end) in enumerate(zip(part["offsets"][:-1],part["offsets"][1:])):
                q=batch*manifest["batch_size"]+local
                if q not in selected:continue
                values=part["probabilities"][start:end]
                positions=np.argsort(np.abs(values-threshold),kind="stable")[:MAX_PAIRS_PER_QUERY]
                for position in positions:
                    candidates.append((q,int(part["rows"][start+position])))
                    before.append(float(values[position]))
    tokenizer,model=_load_model(root/"tail_model",device)
    queries=Corpus.load(root/"test_queries.arrow")
    targets=Corpus.load(root/"test_targets.arrow")
    observed=score_pairs(model,tokenizer,queries,targets,candidates,device)
    after=(1-config["weight"])*np.asarray(before,dtype=np.float32)+config["weight"]*observed
    out=root/"tail_test_overrides.npz"
    np.savez_compressed(out,pairs=np.asarray(candidates,dtype=np.int32).reshape(-1,2),probabilities=after)
    return {"active":True,"overrides":len(candidates),"queries":len(selected)}
