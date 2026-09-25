from argparse import Namespace
import json
from pathlib import Path
from ber.audit import audit_outputs
from ber.build import build
from ber.checkpoint import restore_file
from ber.pipeline import create_training_candidates,train_and_evaluate
from ber.predict import predict,assemble
from tests.fixtures import make_dataset


def test_train_predict_and_resume(tmp_path,monkeypatch):
    data=make_dataset(tmp_path/"dataset")
    artifacts=tmp_path/"artifacts"
    checkpoint=tmp_path/"durable"
    monkeypatch.setenv("BER_ARTIFACT_ROOT",str(artifacts))
    monkeypatch.setenv("BER_CHECKPOINT_ROOT",str(checkpoint))
    monkeypatch.chdir(tmp_path)
    build(data,artifacts,"train",workers=1)
    args=Namespace(data=str(data),artifacts=str(artifacts),samples=100,seed=17,top_k=12,
                   max_postings=150,workers=1,trees=20,xgboost_device="cpu")
    create_training_candidates(args)
    train_and_evaluate(args)
    build(data,artifacts,"test",workers=1)
    first=predict(data,artifacts,workers=1,batch_size=5)
    chunk=artifacts/"predictions/batch_000000.npz"
    saved=chunk.read_bytes()
    chunk.unlink()
    assert restore_file(chunk) and chunk.read_bytes()==saved
    before=chunk.stat().st_mtime_ns
    second=predict(data,artifacts,workers=1,batch_size=5)
    assert first["queries"]==second["queries"]==15
    assert chunk.stat().st_mtime_ns==before
    assemble(artifacts,tmp_path/"output",exclusive=True)
    audit=audit_outputs(tmp_path/"output/matching_results.tsv",tmp_path/"output/candidate_pairs.tsv",data/"test")
    assert audit["ok"],audit
    assert audit["counts"]["s1"]==15
    metrics=json.loads((artifacts/"experiment/validation.json").read_text())
    assert metrics["holdout"]["candidate_recall"]==1.
    # Tuned policy: the tuner runs on the saved experiment and assembly passes the audit.
    from ber.policy import main as tune_policy
    tune_policy(artifacts/"experiment",tmp_path/"policy.json")
    assemble(artifacts,tmp_path/"output_policy",policy=tmp_path/"policy.json",data=data)
    audit=audit_outputs(tmp_path/"output_policy/matching_results.tsv",tmp_path/"output_policy/candidate_pairs.tsv",data/"test")
    assert audit["ok"],audit
    # A policy equal to the plain threshold reproduces the plain output byte for byte,
    # including through the per-country lookup path.
    threshold=json.loads((artifacts/"experiment/calibration.json").read_text())["threshold"]
    flat={"t":threshold,"t1":threshold,"r":0.0}
    (tmp_path/"flat.json").write_text(json.dumps({"global":flat,"by_country":{"Nowhere":flat}}))
    assemble(artifacts,tmp_path/"output_plain",exclusive=False)
    assemble(artifacts,tmp_path/"output_flat",policy=tmp_path/"flat.json",data=data)
    for name in ("matching_results.tsv","candidate_pairs.tsv"):
        assert (tmp_path/"output_flat"/name).read_bytes()==(tmp_path/"output_plain"/name).read_bytes()
    # Per-country shift profile: covers every test reference, and with the plain threshold its
    # holdout F0.5 (query-weighted over countries) equals the pipeline's own holdout metric.
    from ber.shift import profile
    result=profile(artifacts,data,tmp_path/"policy.json")
    assert sum(r["queries"] for r in result["test"].values())==15
    plain=profile(artifacts,data)["holdout"]
    weighted=sum(r["macro_f05"]*r["queries"] for r in plain.values())/sum(r["queries"] for r in plain.values())
    assert abs(weighted-metrics["holdout"]["macro_f05"])<1e-9
