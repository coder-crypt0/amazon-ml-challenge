"""One-session Kaggle training and inference with resumable stage outputs."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from .corpus import Corpus,truth_for_queries
from .retrieval import build_views,retrieve_batch
from .cheap import CHEAP_NAMES,flatten_retrieval,select_top_rows,predict_ranker
from .features import FEATURE_NAMES,BASE_FEATURE_NAMES,enrich
from .expand import expand
from .setmodel import sweep,macro_f05
from .split import choose_reference_rows
from .noisy_channel import NoisyChannel
from .tabm_matcher import train_tabm,predict_tabm,load_tabm
from .reciprocal import owners_from_final_chunks,owners_from_arrays,reverse_features,tuning_result,policy_scores


def _json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")


def _config(root,options):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/"config.json"
    if path.exists() and json.loads(path.read_text())!=options:
        raise ValueError("Run configuration changed. Use a new run directory.")
    _json(path,options)


def _load_or_build(paths,path):
    path=Path(path)
    if path.exists():return Corpus.load(path)
    corpus=Corpus.from_tsv(paths)
    corpus.save(path)
    return corpus


def _paths(data,split):
    data=Path(data)/split
    return [data/f"{split}_source{i}.tsv" for i in (1,2,3)]


def _split(root):
    return json.loads((Path(root)/"split.json").read_text())


def prepare(data,root,samples,seed,cap_rate=.0025,dimension=1<<20,channel_extra=300000):
    root=Path(root)
    _config(root,{"samples":samples,"seed":seed,"cap_rate":cap_rate,"dimension":dimension,
                  "channel_extra":channel_extra,"version":3})
    environment=root/"environment.json"
    if not environment.exists():
        import importlib.metadata
        import platform
        import torch
        names=("numpy","scipy","scikit-learn","polars","xgboost","torch","rapidfuzz","anyascii","sparse-dot-topn")
        versions={name:importlib.metadata.version(name) for name in names}
        _json(environment,{"python":platform.python_version(),"packages":versions,"torch_cuda":torch.version.cuda,
                           "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    paths=_paths(data,"train")
    targets=_load_or_build(paths[1:],root/"train_targets.arrow")
    queries=_load_or_build(paths[:1],root/"train_queries.arrow")
    selected_path=root/"sample_rows.npy"
    if selected_path.exists():
        chosen=np.load(selected_path)
    else:
        chosen,split=choose_reference_rows(len(queries),samples,seed)
        np.save(selected_path,chosen)
        _json(root/"split.json",split)
    if not (root/"split.json").exists():
        raise ValueError("Sample checkpoint has no split manifest; choose a new run directory")
    truth_path=root/"sample_truth.json"
    if truth_path.exists():truth=json.loads(truth_path.read_text(encoding="utf-8"))
    else:
        truth={key:sorted(value) for key,value in truth_for_queries(Path(data)/"train/train_ground_truth.tsv",[queries.ids[int(i)] for i in chosen]).items()}
        _json(truth_path,truth)
    channel_path=root/"noisy_channel.json"
    if not channel_path.exists():
        fit_rows=chosen[:_split(root)["fit_end"]]
        excluded=set(map(int,chosen))
        available=np.fromiter((i for i in range(len(queries)) if i not in excluded),
                              dtype=np.uint32,count=len(queries)-len(excluded))
        extra_count=min(channel_extra,len(available))
        extra=np.random.default_rng(seed+2).choice(available,size=extra_count,replace=False)
        np.save(root/"channel_extra_rows.npy",extra)
        channel_truth={**truth,**{key:sorted(value) for key,value in
            truth_for_queries(Path(data)/"train/train_ground_truth.tsv",
                              [queries.ids[int(i)] for i in extra]).items()}}
        channel_rows=np.r_[fit_rows,extra]
        channel=NoisyChannel.fit(queries,targets,channel_rows,channel_truth,max_links=1600000)
        channel.save(channel_path)
        _json(root/"noisy_channel_profile.json",{**channel.profile(),
             "fit_references":len(fit_rows),"additional_fit_references":extra_count,
             "validation_entities_excluded":len(chosen)-len(fit_rows)})
    channel=NoisyChannel.load(channel_path)
    views=build_views(targets,root/"train_index",cap_rate,dimension,channel=channel,use_word=True)
    return queries,targets,chosen,truth,views


def preflight(queries,targets,chosen,truth,views,root,name_options=(32,64,128,256),address_options=(64,128,256,512),
              sample=2000,threads=4,desired_recall=.985):
    root=Path(root)
    existing=root/"retrieval_config.json"
    if existing.exists():return json.loads(existing.read_text())
    fit_end=_split(root)["fit_end"]
    held=chosen[fit_end:fit_end+min(sample,len(chosen)-fit_end)]
    truths=[truth[queries.ids[int(i)]] for i in held]
    total=sum(map(len,truths))
    if total==0:raise ValueError("Preflight contains no true links")
    trials=[]
    for name_k,address_k in zip(name_options,address_options):
        correct=0;candidates=0;started=time.monotonic()
        for start in range(0,len(held),256):
            qrows=held[start:start+256]
            found=retrieve_batch(queries,qrows,views,name_k,address_k,threads)
            for i,scored in enumerate(found):
                labels=truths[start+i]
                correct+=sum(targets.ids[int(row)] in labels for row in scored)
                candidates+=len(scored)
        recall=correct/total
        trials.append({"name_k":name_k,"address_k":address_k,"candidate_recall":recall,
                       "mean_candidates":candidates/len(held),"seconds":time.monotonic()-started})
        print(f"Preflight retrieval: name={name_k} address={address_k} recall={recall:.6f}, pairs/query={candidates/len(held):.1f}",flush=True)
        if recall>=desired_recall:break
    selected=max(trials,key=lambda x:(x["candidate_recall"]>=desired_recall,-x["mean_candidates"])) if any(t["candidate_recall"]>=desired_recall for t in trials) else trials[-1]
    char_views={key:view for key,view in views.items() if view.analyzer=="char"}
    base_hits=base_pairs=0
    for start in range(0,len(held),256):
        qrows=held[start:start+256]
        found=retrieve_batch(queries,qrows,char_views,selected["name_k"],selected["address_k"],threads)
        for i,scored in enumerate(found):
            base_hits+=sum(targets.ids[int(row)] in truths[start+i] for row in scored)
            base_pairs+=len(scored)
    word_gain=selected["candidate_recall"]-base_hits/total
    use_word=word_gain>=.001
    result={"name_k":selected["name_k"],"address_k":selected["address_k"],"trials":trials,
            "char_only_recall":base_hits/total,"char_only_mean_candidates":base_pairs/len(held),
            "word_gain":word_gain,"use_word":use_word,
            "preflight_true_links":total,"preflight_queries":len(held),"desired_recall":desired_recall}
    _json(existing,result)
    return result


def raw_training(queries,targets,chosen,truth,views,root,retrieval,batch_size=256,threads=4):
    root=Path(root);directory=root/"train_raw";directory.mkdir(exist_ok=True)
    if not retrieval["use_word"]:
        views={key:view for key,view in views.items() if view.analyzer=="char"}
    total=(len(chosen)+batch_size-1)//batch_size
    for batch in range(total):
        path=directory/f"{batch:05d}.npz"
        if path.exists():continue
        qrows=chosen[batch*batch_size:(batch+1)*batch_size]
        scored=retrieve_batch(queries,qrows,views,retrieval["name_k"],retrieval["address_k"],threads)
        x,rows,offsets=flatten_retrieval(queries,targets,qrows,scored)
        labels=np.fromiter((targets.ids[int(row)] in truth[queries.ids[int(q)]]
            for q,entry in zip(qrows,scored) for row in entry),dtype=np.uint8,count=len(rows))
        temporary=path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary,qrows=qrows,rows=rows,offsets=offsets,X=x,y=labels)
        temporary.replace(path)
        if batch%10==0 or batch+1==total:print(f"Raw training candidates {batch+1}/{total}",flush=True)
    return sorted(directory.glob("[0-9]*.npz"))


def _stack(parts,field):
    return np.concatenate([part[field] for part in parts])


def _fit_booster(params,train,tune,model_path,rounds):
    import xgboost as xgb
    model_path=Path(model_path)
    marker=model_path.with_suffix(".complete")
    if marker.exists():
        return xgb.Booster(model_file=str(model_path))
    progress=model_path.with_suffix(".progress.ubj")
    booster=xgb.Booster(model_file=str(progress)) if progress.exists() else None
    done=booster.num_boosted_rounds() if booster is not None else 0
    if done>rounds:
        raise ValueError("Existing tree checkpoint exceeds requested round count")
    while done<rounds:
        booster=xgb.train(params,train,num_boost_round=min(50,rounds-done),
                          evals=[(tune,"tune")],xgb_model=booster,verbose_eval=False)
        done=booster.num_boosted_rounds()
        temporary=progress.with_suffix(".writing.ubj")
        booster.save_model(str(temporary))
        temporary.replace(progress)
        print(f"GPU tree checkpoint: {done}/{rounds} rounds ({model_path.name})",flush=True)
    booster.save_model(str(model_path))
    marker.write_text("complete\n")
    return booster


def train_ranker(raw_paths,root,train_end,tune_end,trees=400,threads=4,device="cuda",seed=42):
    import xgboost as xgb
    root=Path(root);model_path=root/"ranker.ubj"
    if model_path.with_suffix(".complete").exists():
        model=xgb.Booster(model_file=str(model_path));model.set_param({"device":device,"nthread":threads});return model
    X=[];y=[];x_tune=[];y_tune=[]
    query_position=0
    for path in raw_paths:
        with np.load(path) as part:
            rows=len(part["qrows"])
            fit_rows=np.repeat(np.arange(query_position,query_position+rows)<train_end,np.diff(part["offsets"]))
            tune_rows=np.repeat((np.arange(query_position,query_position+rows)>=train_end)&
                       (np.arange(query_position,query_position+rows)<tune_end),np.diff(part["offsets"]))
            X.append(part["X"][fit_rows]);y.append(part["y"][fit_rows])
            if tune_rows.any():x_tune.append(part["X"][tune_rows]);y_tune.append(part["y"][tune_rows])
            query_position+=rows
    train_x=np.concatenate(X);train_y=np.concatenate(y)
    valid_x=np.concatenate(x_tune);valid_y=np.concatenate(y_tune)
    print(f"GPU ranker fitting {len(train_x):,} candidates, {int(train_y.sum()):,} positives",flush=True)
    params=dict(objective="binary:logistic",eval_metric="logloss",tree_method="hist",device=device,
                max_depth=7,min_child_weight=12,learning_rate=.06,subsample=.85,colsample_bytree=.9,
                reg_lambda=3,seed=seed,nthread=threads)
    train=xgb.QuantileDMatrix(train_x,label=train_y,feature_names=list(CHEAP_NAMES))
    valid=xgb.QuantileDMatrix(valid_x,label=valid_y,ref=train,feature_names=list(CHEAP_NAMES))
    model=_fit_booster(params,train,valid,model_path,trees)
    _json(root/"ranker_report.json",{"trees":model.num_boosted_rounds(),"fit_pairs":len(train_x),
                                        "positive_fit_pairs":int(train_y.sum()),"tune_pairs":len(valid_x)})
    return model


def final_training(queries,targets,chosen,truth,raw_paths,ranker,views,root,final_k=128,threads=4,device="cpu"):
    root=Path(root);directory=root/"train_final";directory.mkdir(parents=True,exist_ok=True)
    if not json.loads((root/"retrieval_config.json").read_text())["use_word"]:
        views={key:view for key,view in views.items() if view.analyzer=="char"}
    channel=NoisyChannel.load(root/"noisy_channel.json")
    for batch,path in enumerate(raw_paths):
        final_path=directory/path.name
        if final_path.exists():continue
        with np.load(path) as part:
            qrows=part["qrows"];X=part["X"];rows=part["rows"];offsets=part["offsets"];labels=part["y"]
            scores=predict_ranker(ranker,X,device)
            selected,indices,final_offsets=select_top_rows(scores,rows,offsets,final_k)
            selected,final_offsets,rich=expand(queries,targets,qrows,selected,final_offsets,
                                                X[indices],scores[indices],views,threads,channel=channel)
            labels=np.fromiter((targets.ids[int(row)] in truth[queries.ids[int(q)]]
                for q,(start,end) in zip(qrows,zip(final_offsets[:-1],final_offsets[1:]))
                for row in selected[start:end]),dtype=np.uint8,count=len(selected))
            temporary=final_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary,qrows=qrows,rows=selected,offsets=final_offsets,X=rich,y=labels,
                                ranker_prob=rich[:,FEATURE_NAMES.index("stage1_probability")])
            temporary.replace(final_path)
        if batch%10==0 or batch+1==len(raw_paths):print(f"Enriched training candidates {batch+1}/{len(raw_paths)}",flush=True)
    return sorted(directory.glob("[0-9]*.npz"))


def _load_final(paths,queries,chosen,truth):
    q=len(chosen);width=len(FEATURE_NAMES)
    # Width is the maximum observed selected set; pads are masked in attention.
    max_width=0
    for path in paths:
        with np.load(path) as chunk:max_width=max(max_width,int(np.diff(chunk["offsets"]).max(initial=0)))
    X=np.zeros((q,max(1,max_width),width),dtype=np.float32)
    y=np.zeros(X.shape[:2],dtype=np.float32)
    mask=np.zeros(X.shape[:2],dtype=bool)
    ranker_probs=np.zeros(X.shape[:2],dtype=np.float32)
    target_rows=np.full(X.shape[:2],-1,dtype=np.int32)
    cursor=0
    for path in paths:
        with np.load(path) as chunk:
            for i,(start,end) in enumerate(zip(chunk["offsets"][:-1],chunk["offsets"][1:])):
                size=int(end-start)
                if size:
                    X[cursor+i,:size]=chunk["X"][start:end]
                    y[cursor+i,:size]=chunk["y"][start:end]
                    ranker_probs[cursor+i,:size]=chunk["ranker_prob"][start:end]
                    target_rows[cursor+i,:size]=chunk["rows"][start:end]
                    mask[cursor+i,:size]=True
            cursor+=len(chunk["qrows"])
    if cursor!=q:raise ValueError("Final training chunks do not cover all sampled queries")
    true_count=np.asarray([len(truth[queries.ids[int(row)]]) for row in chosen],dtype=np.int16)
    return X,y,mask,true_count,ranker_probs,target_rows


def _predict_dense_booster(booster,x,mask,device):
    out=np.zeros(mask.shape,dtype=np.float32)
    for start in range(0,len(x),1024):
        stop=min(start+1024,len(x))
        selected=mask[start:stop]
        if selected.any():
            out[start:stop][selected]=predict_ranker(booster,x[start:stop][selected],device)
    return out


def train_rich_booster(x,y,mask,fit_end,tune_end,root,trees=400,threads=4,device="cuda",seed=42,
                       model_name="rich_matcher"):
    import xgboost as xgb
    root=Path(root);model_path=root/f"{model_name}.ubj"
    if model_path.with_suffix(".complete").exists():
        booster=xgb.Booster(model_file=str(model_path))
        booster.set_param({"device":device,"nthread":threads})
        return booster
    train=xgb.QuantileDMatrix(x[:fit_end][mask[:fit_end]],label=y[:fit_end][mask[:fit_end]],feature_names=list(FEATURE_NAMES))
    tune=xgb.QuantileDMatrix(x[fit_end:tune_end][mask[fit_end:tune_end]],
                             label=y[fit_end:tune_end][mask[fit_end:tune_end]],ref=train,feature_names=list(FEATURE_NAMES))
    params=dict(objective="binary:logistic",eval_metric="logloss",tree_method="hist",device=device,
                max_depth=7,min_child_weight=10,learning_rate=.05,subsample=.88,
                colsample_bytree=.9,reg_lambda=3,seed=seed,nthread=threads)
    return _fit_booster(params,train,tune,model_path,trees)


def _entity_stats(probabilities,labels,mask,true_count,threshold):
    selected=(probabilities>=threshold)&mask
    correct=(selected&(labels>0)).sum(1)
    predicted=selected.sum(1)
    truth=np.asarray(true_count)
    singletons=truth==0
    return {"macro_f05":macro_f05(truth,predicted,correct),
            "micro_precision":float(correct.sum()/max(1,predicted.sum())),
            "micro_recall":float(correct.sum()/max(1,truth.sum())),
            "candidate_recall":float(labels[mask].sum()/max(1,truth.sum())),
            "singleton_accuracy":float((predicted[singletons]==0).mean()) if singletons.any() else None,
            "queries":len(truth),"true_links":int(truth.sum()),"predicted_links":int(predicted.sum()),
            "correct_links":int(correct.sum()),"missed_by_retrieval":int(truth.sum()-labels[mask].sum()),
            "missed_by_matcher":int(labels[mask].sum()-correct.sum()),
            "false_positive_links":int(predicted.sum()-correct.sum())}


def train_matcher(queries,chosen,truth,final_paths,root,epochs=12,batch_size=8192,device="cuda",seed=42,trees=400,threads=4):
    root=Path(root)
    X,y,mask,true_count,ranker_probs,target_rows=_load_final(final_paths,queries,chosen,truth)
    split=_split(root);fit_end=split["fit_end"];tune_end=split["tune_end"]
    first=len(BASE_FEATURE_NAMES)
    Xbase=X.copy()
    Xbase[:,:,first:]=0.
    base_model=train_rich_booster(Xbase,y,mask,fit_end,tune_end,root,trees,threads,device,seed,
                                  model_name="rich_base")
    base_scores=_predict_dense_booster(base_model,Xbase[fit_end:],mask[fit_end:],device)
    del Xbase
    channel_model=train_rich_booster(X,y,mask,fit_end,tune_end,root,trees,threads,device,seed,
                                     model_name="rich_channel")
    channel_scores=_predict_dense_booster(channel_model,X[fit_end:],mask[fit_end:],device)
    random_model,random_mean,random_std,_,_,random_mining=train_tabm(
        X,y,mask,ranker_probs,true_count,fit_end,tune_end,root/"tabm_random.pt",
        epochs,batch_size,device,seed,strategy="random")
    random_scores=predict_tabm(random_model,X[fit_end:],mask[fit_end:],random_mean,random_std,device)
    del random_model
    hard_model,hard_mean,hard_std,_,_,hard_mining=train_tabm(
        X,y,mask,ranker_probs,true_count,fit_end,tune_end,root/"tabm_hard.pt",
        epochs,batch_size,device,seed,strategy="hard")
    hard_scores=predict_tabm(hard_model,X[fit_end:],mask[fit_end:],hard_mean,hard_std,device)
    model_outputs={"ranker":ranker_probs[fit_end:],"rich_base":base_scores,
                   "rich_channel":channel_scores,"tabm_random":random_scores,"tabm_hard":hard_scores}
    blends={
        "ranker":{"ranker":1.},
        "rich_base":{"rich_base":1.},
        "rich_channel":{"rich_channel":1.},
        "tabm_random":{"tabm_random":1.},
        "tabm_hard":{"tabm_hard":1.},
        "hard25_channel75":{"tabm_hard":.25,"rich_channel":.75},
        "hard50_channel50":{"tabm_hard":.5,"rich_channel":.5},
        "hard75_channel25":{"tabm_hard":.75,"rich_channel":.25},
        "random25_channel75":{"tabm_random":.25,"rich_channel":.75},
    }
    n_tune=tune_end-fit_end
    options=[]
    for label,weights in blends.items():
        scores=sum(weight*model_outputs[name][:n_tune] for name,weight in weights.items())
        threshold,tune_score=sweep(scores,y[fit_end:tune_end],mask[fit_end:tune_end],true_count[fit_end:tune_end])
        hold_scores=sum(weight*model_outputs[name][n_tune:] for name,weight in weights.items())
        hold_metrics=_entity_stats(hold_scores,y[tune_end:],mask[tune_end:],true_count[tune_end:],threshold)
        options.append({"model":label,"weights":weights,"threshold":threshold,
                        "tune_macro_f05":tune_score,"holdout":hold_metrics})
    selected=max(options,key=lambda option:(option["tune_macro_f05"],
                   float("tabm_hard" in option["weights"]),-len(option["weights"])))
    hold=selected["holdout"]
    selected_all=sum(weight*model_outputs[name] for name,weight in selected["weights"].items())
    selected_scores=selected_all[n_tune:]
    country_metrics={}
    for country in sorted(set(queries.countries[int(i)] for i in chosen[tune_end:])):
        ids=np.asarray([j for j,row in enumerate(chosen[tune_end:])
                        if queries.countries[int(row)]==country],dtype=np.int32)
        country_metrics[country]=_entity_stats(selected_scores[ids],y[tune_end:][ids],
                                               mask[tune_end:][ids],true_count[tune_end:][ids],
                                               selected["threshold"])
    channel_profile=json.loads((root/"noisy_channel_profile.json").read_text())
    ablations=[
        {"stage":"current_colab_baseline","status":"user-reported; split identity independently verified",
         "macro_f05":.9086843491,"candidate_recall":.8807383939,"holdout_true_links":30932},
        {"stage":"improved_blocking","status":"measured","result":next(x for x in options if x["model"]=="rich_base")},
        {"stage":"source_noisy_channel","status":"measured","result":next(x for x in options if x["model"]=="rich_channel")},
        {"stage":"random_negative_tabm","status":"measured","result":next(x for x in options if x["model"]=="tabm_random")},
        {"stage":"hard_negative_tabm","status":"measured","result":next(x for x in options if x["model"]=="tabm_hard")},
        {"stage":"reciprocal_ownership","status":"pending_validation"},
        {"stage":"cross_encoder_tail","status":"pending_validation"},
        {"stage":"collective_propagation","status":"pending_validation"},
    ]
    report={"split":split,"heldout_reference_ids_verified_by_true_link_count":int(true_count[tune_end:].sum())==30932
            if split.get("original_holdout")==9000 else None,
            "candidate_recall":hold["candidate_recall"],"holdout_macro_f05":hold["macro_f05"],
            "micro_precision":hold["micro_precision"],"micro_recall":hold["micro_recall"],
            "singleton_accuracy":hold["singleton_accuracy"],"holdout_queries":hold["queries"],
            "holdout_true_links":hold["true_links"],"holdout_predicted_links":hold["predicted_links"],
            "holdout_correct_links":hold["correct_links"],"holdout_missed_by_retrieval":hold["missed_by_retrieval"],
            "holdout_missed_by_matcher":hold["missed_by_matcher"],
            "holdout_false_positive_links":hold["false_positive_links"],
            "countries":country_metrics,"tuning_options":options,"selected":selected,
            "ablation_table":ablations,"channel_profile":channel_profile,"hard_negative_mining":hard_mining,
            "random_negative_control":random_mining,"final_candidate_mean":float(mask.sum()/len(mask)),
            "tabm_parameters":sum(parameter.numel() for parameter in hard_model.parameters())}
    _json(root/"validation.json",report)
    _json(root/"calibration.json",{"model":selected["model"],"weights":selected["weights"],
                                    "threshold":selected["threshold"]})
    np.savez_compressed(root/"validation_candidates.npz",target_rows=target_rows[fit_end:],
        query_rows=chosen[fit_end:],labels=y[fit_end:].astype(np.uint8),mask=mask[fit_end:],
        true_counts=true_count[fit_end:],ranker=ranker_probs[fit_end:],
        rich_channel=channel_scores,tabm_hard=hard_scores,
        selected=selected_all.astype(np.float32))
    return report


def tune_ownership(root):
    root=Path(root)
    train_paths=sorted((root/"train_final").glob("[0-9]*.npz"))
    if not train_paths:
        raise ValueError("Final training candidates are required for ownership calibration")
    import polars as pl
    target_count=pl.read_ipc(root/"train_targets.arrow",columns=["entity_id"],memory_map=False).height
    best,owner,second=owners_from_final_chunks(train_paths,target_count)
    np.savez_compressed(root/"ownership_train.npz",best=best,owner=owner,second=second)
    with np.load(root/"validation_candidates.npz") as data:
        rows=data["target_rows"];labels=data["labels"];mask=data["mask"]
        true_count=data["true_counts"];ranker=data["ranker"];scores=data["selected"]
    split=_split(root)
    chosen=np.load(root/"sample_rows.npy")
    fit_end=split["fit_end"];tune_end=split["tune_end"]
    calibration=json.loads((root/"calibration.json").read_text())
    ranker_result=tuning_result(scores,rows,mask,labels,true_count,ranker,fit_end,tune_end,
                                calibration["threshold"],best,owner,second)
    final_best,final_owner,final_second=owners_from_arrays(rows,mask,scores,fit_end,len(best))
    final_result=tuning_result(scores,rows,mask,labels,true_count,scores,fit_end,tune_end,
                               calibration["threshold"],final_best,final_owner,final_second)
    experiments=[(scope,trial) for scope,group in (("ranker",ranker_result),("final",final_result))
                 for trial in group["trials"]]
    best_trial=max(experiments,key=lambda item:(item[1]["tune_macro_f05"],item[1]["policy"]=="none"))
    baseline=ranker_result["trials"][0]["tune_macro_f05"]
    scope,trial=best_trial
    policy=(f"{scope}:{trial['policy']}" if trial["policy"]!="none" and
            trial["tune_macro_f05"]>baseline+.0002 else "none")
    result={"selected":policy,"ranker":ranker_result,"final":final_result,
            "reverse_scope":"competing S1 edges in the generated candidate graph"}
    features=reverse_features(rows,mask,ranker,fit_end,best,owner,second)
    np.savez_compressed(root/"reverse_validation_features.npz",features=features,
                        query_rows=chosen[fit_end:])
    calibration["ownership_policy"]=policy
    _json(root/"calibration.json",calibration)
    report=json.loads((root/"validation.json").read_text())
    report["reciprocal_ownership"]=result
    report["selected"]["ownership_policy"]=policy
    for stage in report["ablation_table"]:
        if stage["stage"]=="reciprocal_ownership":
            stage.update(status="measured",result=result)
    if policy!="none":
        variant,rule=policy.split(":",1)
        active=(final_best,final_owner,final_second) if variant=="final" else (best,owner,second)
        adjusted=policy_scores(scores,rows,mask,
                               scores if variant=="final" else ranker,
                               fit_end,*active,rule)
        local_start=tune_end-fit_end
        metrics=_entity_stats(adjusted[local_start:],labels[local_start:],mask[local_start:],
                              true_count[local_start:],calibration["threshold"])
        report["holdout_before_ownership"]=report["holdout_macro_f05"]
        report["holdout_macro_f05"]=metrics["macro_f05"]
        report["micro_precision"]=metrics["micro_precision"]
        report["micro_recall"]=metrics["micro_recall"]
        report["singleton_accuracy"]=metrics["singleton_accuracy"]
        report["holdout_predicted_links"]=metrics["predicted_links"]
        report["holdout_correct_links"]=metrics["correct_links"]
        report["holdout_missed_by_matcher"]=metrics["missed_by_matcher"]
        report["holdout_false_positive_links"]=metrics["false_positive_links"]
    _json(root/"validation.json",report)
    return result


def predict_test(data,root,batch_size=512,final_k=128,threads=4,device="cuda"):
    import xgboost as xgb
    root=Path(root)
    calibration=json.loads((root/"calibration.json").read_text())
    retrieval=json.loads((root/"retrieval_config.json").read_text())
    paths=_paths(data,"test")
    targets=_load_or_build(paths[1:],root/"test_targets.arrow")
    queries=_load_or_build(paths[:1],root/"test_queries.arrow")
    run_config=json.loads((root/"config.json").read_text())
    views=build_views(targets,root/"test_index",run_config["cap_rate"],run_config["dimension"],
                      channel=NoisyChannel.load(root/"noisy_channel.json"),use_word=retrieval["use_word"])
    ranker=xgb.Booster(model_file=str(root/"ranker.ubj"))
    ranker.set_param({"device":device,"nthread":threads})
    channel=NoisyChannel.load(root/"noisy_channel.json")
    weights=calibration["weights"]
    boosters={}
    for key,path_name in (("rich_base","rich_base.ubj"),("rich_channel","rich_channel.ubj")):
        if weights.get(key,0):
            boosters[key]=xgb.Booster(model_file=str(root/path_name))
            boosters[key].set_param({"device":device,"nthread":threads})
    tabm_models={}
    for key,path_name in (("tabm_random","tabm_random.pt"),("tabm_hard","tabm_hard.pt")):
        if weights.get(key,0):
            tabm_models[key]=load_tabm(root/path_name,len(FEATURE_NAMES),device)
    directory=root/"test_predictions";directory.mkdir(exist_ok=True)
    total=(len(queries)+batch_size-1)//batch_size
    manifest={"batches":total,"batch_size":batch_size,"queries":len(queries),
              "threshold":calibration["threshold"],"model":calibration["model"],"weights":weights,
              "final_k":final_k,"anchor_count":2,"anchor_extra_limit":32,
              "name_k":retrieval["name_k"],"address_k":retrieval["address_k"]}
    manifest_path=directory/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError("Prediction configuration changed; use a fresh run directory")
    _json(manifest_path,manifest)
    started=time.monotonic();completed=0
    for batch in range(total):
        path=directory/f"{batch:05d}.npz"
        if path.exists():
            completed+=min(batch_size,len(queries)-batch*batch_size)
            continue
        qrows=np.arange(batch*batch_size,min((batch+1)*batch_size,len(queries)),dtype=np.uint32)
        scored=retrieve_batch(queries,qrows,views,retrieval["name_k"],retrieval["address_k"],threads)
        cheap,rows,offsets=flatten_retrieval(queries,targets,qrows,scored)
        first=predict_ranker(ranker,cheap,device)
        selected,indices,final_offsets=select_top_rows(first,rows,offsets,final_k)
        selected,final_offsets,rich_features=expand(queries,targets,qrows,selected,final_offsets,
            cheap[indices],first[indices],views,threads,channel=channel)
        flat=np.zeros(len(selected),dtype=np.float32)
        if weights.get("ranker",0):
            flat+=weights["ranker"]*rich_features[:,FEATURE_NAMES.index("stage1_probability")]
        for name,booster in boosters.items():
            if name=="rich_base":
                model_features=rich_features.copy()
                model_features[:,len(BASE_FEATURE_NAMES):]=0.
            else:
                model_features=rich_features
            flat+=weights[name]*predict_ranker(booster,model_features,device)
        for name,(model,mean,std) in tabm_models.items():
            data=rich_features.reshape(len(rich_features),1,len(FEATURE_NAMES))
            valid=np.ones((len(rich_features),1),dtype=bool)
            flat+=weights[name]*predict_tabm(model,data,valid,mean,std,device).reshape(-1)
        temporary=path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary,rows=selected,offsets=final_offsets.astype(np.int32),
                            probabilities=np.asarray(flat,dtype=np.float32),
                            ranker_prior=rich_features[:,FEATURE_NAMES.index("stage1_probability")].astype(np.float32))
        temporary.replace(path)
        completed+=len(qrows)
        if batch%10==0 or batch+1==total:
            speed=completed/max(1,time.monotonic()-started)
            eta=(len(queries)-completed)/max(1e-9,speed)/3600
            print(f"Test predictions {batch+1}/{total}: {completed:,}/{len(queries):,}, {speed:.1f} queries/s, ~{eta:.2f} h remaining",flush=True)
    (directory/"COMPLETE").write_text("complete\n")
    return manifest


def assemble(data,root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    directory=root/"test_predictions"
    if not (directory/"COMPLETE").exists():raise ValueError("Test prediction stage is incomplete")
    manifest=json.loads((directory/"manifest.json").read_text())
    queries=Corpus.load(root/"test_queries.arrow")
    targets=Corpus.load(root/"test_targets.arrow")
    threshold=manifest["threshold"]
    tail_overrides={}
    tail_file=root/"tail_test_overrides.npz"
    tail_status=json.loads((root/"tail_config.json").read_text()) if (root/"tail_config.json").exists() else {"active":False}
    if tail_status.get("active") and not tail_file.exists():
        raise ValueError("Active cross-encoder tail has no completed test overrides")
    if tail_file.exists():
        with np.load(tail_file) as values:
            tail_overrides={(int(q),int(t)):float(p)
                            for (q,t),p in zip(values["pairs"],values["probabilities"])}
    collective_config=json.loads((root/"collective_config.json").read_text()) if (root/"collective_config.json").exists() else {"active":False}
    collective_delta=collective_config.get("selected_delta",0.) if collective_config.get("active") else 0.
    policy=json.loads((root/"calibration.json").read_text()).get("ownership_policy","none")
    owner=best=second=None
    if policy!="none":
        paths=[directory/f"{i:05d}.npz" for i in range(manifest["batches"])]
        scope,_=policy.split(":",1)
        score_key="probabilities" if scope=="final" else "ranker_prior"
        best,owner,second=owners_from_final_chunks(paths,len(targets),score_key=score_key)
    predicted=0;matched=0
    mp=output/"matching_results.tsv";cp=output/"candidate_pairs.tsv"
    with mp.open("w",encoding="utf-8",newline="") as m,cp.open("w",encoding="utf-8",newline="") as c:
        mw=csv.writer(m,delimiter="\t",lineterminator="\n");cw=csv.writer(c,delimiter="\t",lineterminator="\n")
        mw.writerow(["source1_entity_id","matched_entity_ids"])
        cw.writerow(["source1_entity_id","candidate_entity_ids"])
        for batch in range(manifest["batches"]):
            with np.load(directory/f"{batch:05d}.npz") as part:
                offsets=part["offsets"]
                for i,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
                    qid=queries.ids[batch*manifest["batch_size"]+i]
                    qid_index=batch*manifest["batch_size"]+i
                    rows=part["rows"][start:end]
                    probabilities=part["probabilities"][start:end]
                    candidates=[targets.ids[int(r)] for r in rows]
                    effective=np.asarray([tail_overrides.get((qid_index,int(target)),float(p))
                                          for target,p in zip(rows,probabilities)],dtype=np.float32)
                    if policy!="none":
                        rule=policy.split(":",1)[1]
                        for position,target in enumerate(rows):
                            target=int(target)
                            if owner[target]!=qid_index or (rule.startswith("gap:") and
                                    best[target]-second[target]<float(rule.partition(":")[2])):
                                effective[position]=0.
                    if collective_delta and len(rows):
                        from .collective import propagate_batch
                        effective=propagate_batch(effective[None,:],rows[None,:],
                            np.ones((1,len(rows)),dtype=bool),targets,threshold,collective_delta)[0]
                    matches=[]
                    for entity,target,p in zip(candidates,rows,effective):
                        if p<threshold:
                            continue
                        if policy!="none":
                            target=int(target)
                            if owner[target]!=qid_index:
                                continue
                            rule=policy.split(":",1)[1]
                            if rule.startswith("gap:") and best[target]-second[target]<float(rule.partition(":")[2]):
                                continue
                        matches.append(entity)
                    mw.writerow([qid,",".join(matches)]);cw.writerow([qid,",".join(candidates)])
                    predicted+=1;matched+=len(matches)
    if predicted!=len(queries):raise ValueError("Missing Source 1 rows in assembled output")
    report={"queries":predicted,"matched_links":matched,"ownership_policy":policy,
            "collective_delta":collective_delta,"candidate_file_bytes":cp.stat().st_size,
            "matching_file_bytes":mp.stat().st_size}
    _json(root/"prediction_summary.json",report)
    return report


def audit(data,root,output):
    from ber.audit import audit_outputs
    output=Path(output)
    report=audit_outputs(output/"matching_results.tsv",output/"candidate_pairs.tsv",Path(data)/"test")
    _json(Path(root)/"audit.json",report)
    if not report["ok"]:raise ValueError(report["errors"][:5])
    return report


def create_figures(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(root);report=json.loads((root/"validation.json").read_text())
    figures=root/"figures";figures.mkdir(exist_ok=True)
    options=report["tuning_options"]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    names=[item["model"].replace("_"," ") for item in options]
    axes[0].bar(range(len(options)),[item["tune_macro_f05"] for item in options],color="#2563eb")
    axes[0].set_xticks(range(len(options)),names,rotation=42,ha="right")
    axes[0].set(xlabel="Matcher or blend",ylabel="Tuning macro F0.5",title="CNER ablations")
    metrics=[report["candidate_recall"],report["micro_recall"],report["micro_precision"],report["holdout_macro_f05"]]
    axes[1].bar(["Candidate recall","Link recall","Precision","Macro F0.5"],metrics,color=["#64748b","#2563eb","#0d9488","#d97706"])
    axes[1].set_ylim(0,1.05);axes[1].tick_params(axis="x",rotation=28);axes[1].set_title("Untouched entity holdout")
    fig.tight_layout()
    for suffix in ("png","svg"):fig.savefig(figures/f"validation.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    countries=report["countries"]
    if countries:
        fig,ax=plt.subplots(figsize=(8,4.5))
        keys=list(countries)
        x=np.arange(len(keys))
        for position,(metric,color) in enumerate((("candidate_recall","#64748b"),("macro_f05","#2563eb"),
                                                   ("micro_precision","#0d9488"),("micro_recall","#d97706"))):
            ax.bar(x+(position-1.5)*.18,[countries[c][metric] for c in keys],width=.18,label=metric,color=color)
        ax.set_xticks(x,keys);ax.set_ylim(0,1.05);ax.set_title("Untouched holdout by observed country")
        ax.legend(loc="lower right",fontsize=8)
        fig.tight_layout()
        for suffix in ("png","svg"):fig.savefig(figures/f"country_holdout.{suffix}",dpi=180,bbox_inches="tight")
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    errors=[report["holdout_missed_by_retrieval"],report["holdout_missed_by_matcher"],
            report["holdout_false_positive_links"]]
    bars=ax.bar(["Retrieval misses","Matcher misses","False matches"],errors,
                color=["#64748b","#d97706","#dc2626"])
    ax.bar_label(bars,padding=3)
    ax.set(title="Where held-out links are lost",ylabel="Number of links")
    fig.tight_layout()
    for suffix in ("png","svg"):fig.savefig(figures/f"error_decomposition.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    _json(figures/"plot_data.json",{"source":"validation.json","metrics":report,
                                     "recreate":"python -m gpu100.runner figures --root RUN_DIR"})
    return figures


def compact_training_cache(root):
    """Free saved working space after models and validation are durable."""
    root=Path(root).resolve()
    if not (root/"validation.json").is_file() or not (root/"calibration.json").is_file():
        raise ValueError("Model fitting and validation must finish before compaction")
    import shutil
    removed=[]
    for name in ("train_index","train_raw","train_final"):
        directory=(root/name).resolve()
        if not directory.is_relative_to(root) or directory==root:
            raise ValueError("Invalid cache path")
        if directory.is_dir():
            shutil.rmtree(directory)
            removed.append(name)
    for name in ("train_targets.arrow","train_queries.arrow"):
        path=(root/name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Invalid corpus path")
        if path.exists():
            path.unlink()
            removed.append(name)
    _json(root/"compaction.json",{"removed":removed})
    return removed

def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("prepare","preflight","raw","ranker","final","matcher","ownership","tail","collective","figures","compact","test","tail_apply","assemble","audit","package","all"))
    p.add_argument("--data",default="student_resource/dataset")
    p.add_argument("--root",default="artifacts/gpu100")
    p.add_argument("--output",default="output")
    p.add_argument("--samples",type=int,default=60000)
    p.add_argument("--seed",type=int,default=20260925)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--final-k",type=int,default=128)
    p.add_argument("--trees",type=int,default=400)
    p.add_argument("--epochs",type=int,default=18)
    p.add_argument("--cap-rate",type=float,default=.0025)
    p.add_argument("--dimension",type=int,default=1<<20)
    p.add_argument("--channel-extra",type=int,default=300000)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--team",default="entity_resolution")
    p.add_argument("--members",nargs="*",default=[])
    p.add_argument("--tail",choices=("on","off"),default="on")
    a=p.parse_args();root=Path(a.root)
    root.mkdir(parents=True,exist_ok=True)
    hyperparameters={"samples":a.samples,"seed":a.seed,"trees":a.trees,"epochs":a.epochs,
        "final_k":a.final_k,"batch_size":a.batch_size,"cap_rate":a.cap_rate,
        "dimension":a.dimension,"channel_extra":a.channel_extra,
        "anchor_count":2,"anchor_extra_limit":32}
    hyper_path=root/"hyperparameters.json"
    if hyper_path.exists() and json.loads(hyper_path.read_text())!=hyperparameters:
        raise ValueError("Run hyperparameters changed; use a new run directory")
    _json(hyper_path,hyperparameters)
    stages=[a.stage] if a.stage!="all" else ["prepare","preflight","raw","ranker","final","matcher","ownership","tail","collective","figures","compact","test","tail_apply","assemble","audit","package"]
    prepared=None
    for stage in stages:
        started=time.monotonic();print(f"Stage {stage} starting",flush=True)
        if stage in ("prepare","preflight","raw","ranker","final","matcher"):
            if prepared is None:
                prepared=prepare(a.data,root,a.samples,a.seed,a.cap_rate,a.dimension,a.channel_extra)
            queries,targets,chosen,truth,views=prepared
        else:
            prepared=None
            if stage in ("compact","test") and "queries" in locals():
                import gc
                del queries,targets,chosen,truth,views
                gc.collect()
        if stage=="prepare":pass
        elif stage=="preflight":preflight(queries,targets,chosen,truth,views,root,threads=a.threads)
        elif stage=="raw":
            config=preflight(queries,targets,chosen,truth,views,root,threads=a.threads)
            raw_training(queries,targets,chosen,truth,views,root,config,a.batch_size,a.threads)
        elif stage=="ranker":
            split=_split(root)
            train_ranker(sorted((root/"train_raw").glob("[0-9]*.npz")),root,split["fit_end"],split["tune_end"],a.trees,a.threads,a.device,a.seed)
        elif stage=="final":
            import xgboost as xgb
            ranker=xgb.Booster(model_file=str(root/"ranker.ubj"))
            ranker.set_param({"device":a.device,"nthread":a.threads})
            final_training(queries,targets,chosen,truth,sorted((root/"train_raw").glob("[0-9]*.npz")),
                           ranker,views,root,a.final_k,a.threads,a.device)
        elif stage=="matcher":
            print(json.dumps(train_matcher(queries,chosen,truth,sorted((root/"train_final").glob("[0-9]*.npz")),
                          root,a.epochs,8192,a.device,a.seed,a.trees,a.threads),indent=2),flush=True)
        elif stage=="ownership":print(json.dumps(tune_ownership(root),indent=2),flush=True)
        elif stage=="tail":
            from .tail import fine_tune,tune_tail
            if a.tail=="off" or (a.device=="cpu" and a.samples<1000):
                reason="disabled_by_configuration" if a.tail=="off" else "skipped_synthetic_cpu_smoke"
                _json(root/"tail_config.json",{"active":False,"status":reason})
                print("Cross-encoder tail:",reason,flush=True)
            else:
                try:
                    fine_tune(root,a.device)
                    print(json.dumps(tune_tail(root,a.device),indent=2),flush=True)
                except Exception as exc:
                    failure={"active":False,"status":"failed","error_type":type(exc).__name__,"message":str(exc)}
                    _json(root/"tail_config.json",failure)
                    print("Cross-encoder tail failed; audited base model remains available:",json.dumps(failure),flush=True)
                    report=json.loads((root/"validation.json").read_text())
                    for row in report["ablation_table"]:
                        if row["stage"]=="cross_encoder_tail":row.update(status="failed",error=failure)
                    _json(root/"validation.json",report)
        elif stage=="collective":
            from .collective import tune_collective
            print(json.dumps(tune_collective(root),indent=2),flush=True)
        elif stage=="test":predict_test(a.data,root,a.batch_size,a.final_k,a.threads,a.device)
        elif stage=="tail_apply":
            from .tail import apply_tail_test
            try:
                print(json.dumps(apply_tail_test(root,a.device),indent=2),flush=True)
            except Exception as exc:
                failure={"active":False,"status":"failed_during_test","error_type":type(exc).__name__,"message":str(exc)}
                _json(root/"tail_config.json",failure)
                print("Cross-encoder test scoring failed; using base model:",json.dumps(failure),flush=True)
                report=json.loads((root/"validation.json").read_text())
                report["tail_inference_failure"]=failure
                if "holdout_before_tail" in report:
                    report["holdout_macro_f05"]=report["holdout_before_tail"]
                _json(root/"validation.json",report)
        elif stage=="assemble":print(json.dumps(assemble(a.data,root,a.output),indent=2),flush=True)
        elif stage=="audit":print(json.dumps(audit(a.data,root,a.output),indent=2),flush=True)
        elif stage=="figures":create_figures(root)
        elif stage=="compact":print("Removed caches:",compact_training_cache(root),flush=True)
        elif stage=="package":
            from .deliver import package
            print(package(root,a.output,Path(a.output).parent/f"{a.team}_submission.zip",a.team,a.members),flush=True)
        seconds=time.monotonic()-started
        runtime_path=root/"runtime.json"
        runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
        runtime[stage]={"seconds":seconds,"minutes":seconds/60}
        _json(runtime_path,runtime)
        print(f"Stage {stage} finished in {seconds/60:.1f} minutes",flush=True)


if __name__=="__main__":main()
