from argparse import Namespace
import csv
import json
import numpy as np
from ber import tokens
from ber.audit import audit_outputs
from ber.checkpoint import restore_file
from ber.features3 import FEATURE_NAMES, TokenOdds, prepare, query_features, word_keys
from ber.retrieval import TokenIndex
from ber.store import RecordStore
from tests.fixtures import make_dataset


def _table(tmp_path, monkeypatch):
    path = tmp_path / "translit.json"
    path.write_text(json.dumps({"name": {"प्राइवेट": "private", "राम": "ram", "सिस्टम्स": "systems"},
                                "addr": {"गुजरात": "gujarat"}}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv(tokens.ENV, str(path))
    tokens._tables.cache_clear()
    return path


def test_tokens_transliterate_split_aliases_and_canonicalize_numbers(tmp_path, monkeypatch):
    _table(tmp_path, monkeypatch)
    assert tokens.name_parts("राम सिस्टम्स प्राइवेट")[2] == ["ram", "systems"]
    assert tokens.name_parts("Nexo formerly: Clean Horizon")[1] == ["cleanhorizon"]
    assert tokens.is_domain("cozyyoga.com") and tokens.is_domain("#perfectsystems") and not tokens.is_domain("Cozy Yoga")
    words, numbers = tokens.address_parts("0821 PARROTT ST, 70nd Street, गुजरात")
    assert numbers == ["821", "70"] and "street" in words and "gujarat" in words and "nd" not in words
    assert tokens.name_parts("Institut E.U.R.L.")[2] == ["institut"] == tokens.name_parts("Institut [S.A.S]")[2]
    assert tokens.name_parts("Pazos and Delk, L.L.C.")[2] == ["pazos", "delk"]
    words, numbers = tokens.address_parts("N° 86 R. ACHILLE, K.V.Rangareddy")
    assert "north" not in words and numbers == ["86"] and "rangareddy" in words
    assert all(t.startswith("france|") for t in tokens.record_tokens({"business_name": "Acme", "country": "France"}))
    tokens._tables.cache_clear()


def _store(tmp_path, rows):
    path = tmp_path / "targets.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["entity_id", "business_name", "business_address", "country"])
        writer.writerows(rows)
    return RecordStore.build([path], tmp_path / "store")


def test_token_index_recovers_script_domain_and_empty_address_copies(tmp_path, monkeypatch):
    table = _table(tmp_path, monkeypatch)
    rows = [["S2-1", "Ram Systems Private Limited", "703 Diamond Tower, Surat, Gujarat", "India"],
            ["S2-2", "राम सिस्टम्स प्राइवेट", "DOOR NO 703, DIAMOND TOWER, SURAT, गुजरात", "India"],
            ["S3-3", "#ramsystems", "703 Diamond Tower, Surat, GJ", "India"],
            ["S3-4", "Ram Systems Pvt Ltd", "", "India"],
            ["S2-5", "Other Widgets Ltd", "12 Other Road, Pune", "India"]]
    rows += [[f"S2-{i}", f"Filler {i} Traders", f"{i} Filler Lane, Delhi", "India"] for i in range(6, 40)]
    store = _store(tmp_path, rows)
    index = TokenIndex.build(store, tmp_path / "tokens", table, workers=1, k_all=3, k_name=2)
    query = {"entity_id": "S1-1", "business_name": "Ram Systems Private Limited",
             "business_address": "Surat, Gujarat, 703 Diamond Tower", "country": "India"}
    found, score_all, score_name, channel = index.query(query)
    ids = [store[int(r)]["entity_id"] for r in found]
    assert set(ids) == {"S2-1", "S2-2", "S3-3", "S3-4"} and channel.tolist().count(1) >= 1
    again = TokenIndex(tmp_path / "tokens", k_all=3, k_name=2).query(query)
    assert again[0].tolist() == found.tolist()
    left, targets = prepare(query, index), [prepare(store[int(r)], index) for r in found]
    X, extra, missing = query_features(left, targets, score_all, score_name, channel)
    assert X.shape == (len(found), len(FEATURE_NAMES)) and np.isfinite(X).all() and len(extra) == len(missing) == len(found)
    store.close()
    tokens._tables.cache_clear()


def test_token_odds_are_signed_and_empty_lists_are_neutral():
    e_keys, e_counts = word_keys([["center"], ["midtown"], ["center"], ["midtown"], []], "e")
    m_keys, m_counts = word_keys([[], [], [], [], []], "m")
    labels = np.array([1, 0, 1, 0, 1])
    odds = TokenOdds.fit(e_keys, e_counts, m_keys, m_counts, labels, strength=1.)
    f = odds.features(e_keys, e_counts, m_keys, m_counts)
    assert f.shape == (5, 5) and f[0, 0] > 0 > f[1, 0] and not f[4].any()


def test_v3_train_predict_resume_assemble_and_package(tmp_path, monkeypatch):
    from ber.finalize3 import finalize
    from ber.pipeline3 import create_training_candidates, train_and_evaluate
    from ber.predict import assemble
    from ber.predict3 import predict
    data = make_dataset(tmp_path / "dataset")
    artifacts, checkpoint = tmp_path / "artifacts", tmp_path / "durable"
    monkeypatch.setenv("BER_ARTIFACT_ROOT", str(artifacts))
    monkeypatch.setenv("BER_CHECKPOINT_ROOT", str(checkpoint))
    monkeypatch.setenv(tokens.ENV, "")  # training points this at its learned table; restored afterwards
    monkeypatch.chdir(tmp_path)
    args = Namespace(data=str(data), artifacts=str(artifacts), samples=100, seed=17, df_max=50000, k_all=12, k_name=4,
                     workers=1, trees=20, xgboost_device="cpu")
    create_training_candidates(args)
    report = train_and_evaluate(args)
    assert report["holdout"]["candidate_recall"] == 1. and len(report["feature_importance"]) == 92
    first = predict(data, artifacts, workers=1, batch_size=5)
    chunk = artifacts / "predictions/batch_000000.npz"
    saved = chunk.read_bytes()
    chunk.unlink()
    assert restore_file(chunk) and chunk.read_bytes() == saved
    assert predict(data, artifacts, workers=1, batch_size=5)["queries"] == first["queries"] == 15
    one = artifacts / "predictions/batch_000001.npz"
    with np.load(one) as z:
        before = {k: z[k] for k in z.files}
    one.unlink()
    (checkpoint / "predictions/batch_000001.npz").unlink()
    assert predict(data, artifacts, workers=1, batch_size=5, reverse=True)["scored_here"] == 5
    with np.load(one) as z:
        assert all(np.array_equal(z[k], v) for k, v in before.items())
    for name, exclusive in (("output", False), ("output_exclusive", True)):
        assemble(artifacts, tmp_path / name, exclusive=exclusive)
        audit = audit_outputs(tmp_path / name / "matching_results.tsv", tmp_path / name / "candidate_pairs.tsv", data / "test")
        assert audit["ok"] and audit["counts"]["s1"] == 15, audit
    project = tmp_path / "project"
    project.mkdir()
    for name in ("README.md", "requirements.txt"):
        (project / name).write_text("x\n")
    target, audit = finalize(data, artifacts, tmp_path / "output", project, "team")
    assert target.is_file() and audit["ok"] and "Candidate Generation" in (project / "Documentation_template.md").read_text()
