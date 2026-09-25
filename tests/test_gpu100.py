from pathlib import Path
import numpy as np
from gpu100.corpus import Corpus, normalize, name_text, address_text
from gpu100.retrieval import HashView, retrieve_batch
from gpu100.cheap import CHEAP_NAMES, flatten_retrieval, select_top_rows
from gpu100.features import FEATURE_NAMES, enrich
from gpu100.setmodel import macro_f05, make_network
from gpu100.expand import expand
from gpu100.runner import compact_training_cache,_sample_ranker_rows,train_ranker
from gpu100.noisy_channel import NoisyChannel, CHANNEL_FEATURE_NAMES
from gpu100.reciprocal import owners_from_arrays,reverse_features,tuning_result
from gpu100.tail import ambiguous_queries,tail_pairs
from gpu100.collective import propagate_batch
from gpu100.split import choose_reference_rows
import csv
from tests.fixtures import make_dataset


def test_unicode_and_open_country_retrieval(tmp_path):
    data=make_dataset(tmp_path/"dataset",train_queries=25,test_queries=9)
    q=Corpus.from_tsv([data/"test/test_source1.tsv"])
    t=Corpus.from_tsv([data/"test/test_source2.tsv",data/"test/test_source3.tsv"])
    assert "France" in t.by_country()
    assert name_text("Café SARL") == "cafe"
    assert address_text("23 Rue du Centre") == "23 rue du centre"
    views={}
    for country,rows in t.by_country().items():
        for field in ("name","address"):
            views[country,field]=HashView.build(t,country,field,rows,grams=(3,4),dimension=1<<14,cap=1000)
    scored=retrieve_batch(q,np.arange(len(q)),views,name_k=12,address_k=12,threads=1)
    x,rows,offsets=flatten_retrieval(q,t,np.arange(len(q)),scored)
    assert x.shape[1]==len(CHEAP_NAMES)
    assert len(offsets)==len(q)+1
    chosen,indices,new_offsets=select_top_rows(np.ones(len(x)),rows,offsets,12)
    rich=enrich(q,t,np.arange(len(q)),chosen,new_offsets,x[indices],np.ones(len(indices)),threads=1)
    assert rich.shape==(len(chosen),len(FEATURE_NAMES))
    assert np.isfinite(rich).all()


def test_set_model_handles_empty_candidates():
    import torch
    model=make_network(len(FEATURE_NAMES))
    x=torch.zeros((2,8,len(FEATURE_NAMES)))
    mask=torch.zeros((2,8),dtype=torch.bool)
    mask[1,0]=True
    logits=model(x,mask)
    assert logits.shape==(2,8)
    assert torch.isfinite(logits).all()
    assert macro_f05([0,1],[0,0],[0,0])==.5


def test_anchor_recovers_another_source(tmp_path):
    data=make_dataset(tmp_path/"dataset",train_queries=20,test_queries=3)
    q=Corpus.from_tsv([data/"train/train_source1.tsv"])
    t=Corpus.from_tsv([data/"train/train_source2.tsv",data/"train/train_source3.tsv"])
    views={}
    for country,rows in t.by_country().items():
        for field in ("name","address"):
            for variant in ("original","sorted"):
                views[country,field,variant]=HashView.build(t,country,field,rows,grams=(3,4),
                    dimension=1<<14,cap=1000,sort_tokens=variant=="sorted")
    anchor=t.ids.index("S2-1")
    cheap=np.zeros((1,len(CHEAP_NAMES)),dtype=np.float32)
    rows,offsets,features=expand(q,t,np.asarray([1],dtype=np.uint32),
        np.asarray([anchor],dtype=np.uint32),np.asarray([0,1]),cheap,
        np.asarray([.99],dtype=np.float32),views,threads=1)
    ids=[t.ids[int(i)] for i in rows]
    assert ids[0]=="S2-1"
    assert "S3-1" in ids
    assert len(ids)==len(set(ids))
    assert features[ids.index("S3-1"),FEATURE_NAMES.index("anchor_expanded")]==1


def test_compaction_keeps_models_and_reports(tmp_path):
    root=tmp_path/"run"
    (root/"train_index").mkdir(parents=True)
    (root/"train_index"/"part.npy").write_bytes(b"index")
    (root/"validation.json").write_text("{}")
    (root/"calibration.json").write_text("{}")
    (root/"ranker.ubj").write_bytes(b"model")
    removed=compact_training_cache(root)
    assert "train_index" in removed
    assert not (root/"train_index").exists()
    assert (root/"ranker.ubj").read_bytes()==b"model"
    assert (root/"validation.json").is_file()


def test_ranker_sampling_keeps_positives_and_confusing_negatives():
    x=np.zeros((241,len(CHEAP_NAMES)),dtype=np.float32)
    y=np.zeros(241,dtype=np.uint8)
    y[[3,89,160]]=1
    x[:,2]=np.arange(241,dtype=np.float32)/241
    offsets=np.asarray([0,120,120,241])
    chosen=_sample_ranker_rows(x,y,offsets,np.random.default_rng(7),hard_per_query=4,random_per_query=2)
    assert {3,89,160,116,117,118,119,237,238,239,240}.issubset(set(chosen))
    assert len(chosen)==15
    assert len(set(chosen))==len(chosen)
    assert np.array_equal(chosen,_sample_ranker_rows(x,y,offsets,np.random.default_rng(7),4,2))


def test_ranker_fits_bounded_sample_from_saved_candidates(tmp_path):
    import json
    root=tmp_path/"run"
    root.mkdir()
    rng=np.random.default_rng(21)
    x=rng.random((6*120,len(CHEAP_NAMES)),dtype=np.float32)
    y=np.zeros(len(x),dtype=np.uint8)
    y[np.arange(6)*120+3]=1
    path=root/"raw.npz"
    np.savez_compressed(path,qrows=np.arange(6),rows=np.arange(len(x)),
                        offsets=np.arange(7)*120,X=x,y=y)
    model=train_ranker([path],root,train_end=4,tune_end=6,trees=2,threads=2,device="cpu")
    report=json.loads((root/"ranker_report.json").read_text())
    assert model.num_boosted_rounds()==2
    assert report["fit_pairs"]<=4*81
    assert report["positive_fit_pairs"]==4
    assert report["tune_pairs"]<=2*81


def test_compaction_unlinks_resume_cache_without_touching_checkpoint(tmp_path):
    import pytest
    source=tmp_path/"prior"/"train_raw"
    source.mkdir(parents=True)
    (source/"00000.npz").write_bytes(b"checkpoint")
    root=tmp_path/"new"
    root.mkdir()
    try:
        (root/"train_raw").symlink_to(source,target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks unavailable")
    (root/"validation.json").write_text("{}")
    (root/"calibration.json").write_text("{}")
    assert "train_raw" in compact_training_cache(root)
    assert not (root/"train_raw").exists()
    assert (source/"00000.npz").read_bytes()==b"checkpoint"


def test_channel_uses_only_fit_positives_and_keeps_sources_separate(tmp_path):
    data=make_dataset(tmp_path/"dataset",train_queries=20,test_queries=3)
    source3=data/"train/train_source3.tsv"
    with source3.open(encoding="utf-8",newline="") as handle:
        rows=list(csv.reader(handle,delimiter="\t"))
    for row in rows[1:]:
        if row[0]=="S3-15":
            row[1]=row[1].replace("Workshop","W0rkshop")
        elif row[0].startswith("S3-"):
            row[1]=row[1].replace("Workshop","Wurkshop")
    with source3.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle,delimiter="\t",lineterminator="\n")
        writer.writerows(rows)
    q=Corpus.from_tsv([data/"train/train_source1.tsv"])
    t=Corpus.from_tsv([data/"train/train_source2.tsv",source3])
    truth={q.ids[i]:([f"S2-{i}",f"S3-{i}"] if i%10 else []) for i in range(len(q))}
    channel=NoisyChannel.fit(q,t,np.arange(10,dtype=np.uint32),truth,min_alias_count=2)
    assert channel.fit_links==18
    assert channel.ops["3:name"].get("R:o>u",0)>0
    assert channel.ops.get("2:name",{}).get("R:o>u",0)==0
    assert channel.ops["3:name"].get("R:o>0",0)==0  # Held-out S3-15 is excluded.
    features=channel.score_pair(q.names[1],q.addresses[1],t.names[t.ids.index("S3-1")],
                                t.addresses[t.ids.index("S3-1")],3)
    assert len(features)==len(CHANNEL_FEATURE_NAMES)


def test_reverse_owner_and_tail_gate():
    rows=np.array([[2,4],[2,5]],dtype=np.int32)
    mask=np.ones_like(rows,dtype=bool)
    prior=np.array([[.9,.7],[.8,.75]],dtype=np.float32)
    best,owner,second=owners_from_arrays(rows,mask,prior,0,6)
    assert owner[2]==0 and np.isclose(second[2],.8)
    features=reverse_features(rows,mask,prior,0,best,owner,second)
    assert features[0,0,0]==1 and features[1,0,0]==2
    chosen=ambiguous_queries(prior,mask,.78,fraction=.5)
    assert len(chosen)==1
    assert all(mask[q,p] for q,p in tail_pairs(prior,mask,.78,chosen,max_pairs=1))


def test_single_conservative_propagation_pass(tmp_path):
    data=make_dataset(tmp_path/"dataset",train_queries=20,test_queries=3)
    targets=Corpus.from_tsv([data/"train/train_source2.tsv",data/"train/train_source3.tsv"])
    same=[targets.ids.index("S2-1"),targets.ids.index("S3-1")]
    different=targets.ids.index("S2-2")
    rows=np.asarray([[same[0],same[1],different]],dtype=np.int32)
    values=np.asarray([[.99,.61,.60]],dtype=np.float32)
    result=propagate_batch(values,rows,np.ones_like(rows,dtype=bool),targets,.65,.1)
    assert result[0,1]>=.65
    assert result[0,2]==values[0,2]


def test_added_fit_entities_preserve_original_tune_and_holdout():
    original,base=choose_reference_rows(150000,60000,20260925)
    larger,new=choose_reference_rows(150000,100000,20260925)
    np.testing.assert_array_equal(original[:42000],larger[:42000])
    np.testing.assert_array_equal(original[42000:],larger[new["fit_end"]:])
    assert len(set(map(int,larger)))==len(larger)
    assert base["tune_end"]==51000 and new["tune_end"]==91000
