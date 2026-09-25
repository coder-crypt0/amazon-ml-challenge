"""Audited source package and methodology for the Kaggle run."""
import json
from pathlib import Path
import time
import zipfile


def package(root,output,destination,team="entity_resolution",members=()):
    root=Path(root);output=Path(output);destination=Path(destination)
    audit=json.loads((root/"audit.json").read_text())
    if not audit["ok"]:raise ValueError("Audit must pass before packaging")
    metrics=json.loads((root/"validation.json").read_text())
    selected=json.loads((root/"calibration.json").read_text())
    ablation_lines=[]
    for stage in metrics.get("ablation_table",[]):
        name=stage["stage"]
        status=stage["status"]
        result=stage.get("result",{})
        hold=result.get("holdout",{}) if isinstance(result,dict) else {}
        value=hold.get("macro_f05")
        if name=="current_colab_baseline":value=stage.get("macro_f05")
        if name=="reciprocal_ownership" and result.get("selected"):
            value=next((x["holdout_macro_f05"] for group in (result.get("ranker",{}),result.get("final",{}))
                        for x in group.get("trials",[]) if x["policy"] in result["selected"]),value)
        if name=="cross_encoder_tail":value=result.get("holdout_after_tail",{}).get("macro_f05",value)
        if name=="collective_propagation" and result.get("trials"):
            value=next((x["holdout"]["macro_f05"] for x in result["trials"]
                        if x["delta"]==result.get("selected_delta",0)),value)
        ablation_lines.append(f"| {name} | {status} | {value:.6f} |" if value is not None
                              else f"| {name} | {status} | — |")
    ablation_table="\n".join(["| Stage | Status | Held-out macro F0.5 |",
                              "|---|---|---:|",*ablation_lines])
    text=f"""# Business Entity Resolution Methodology

Team: {team}. Members: {', '.join(members) if members else 'Not specified in run configuration'}. Date: {time.strftime('%Y-%m-%d')}.

## Methodology

Country-partitioned, hashed character TF-IDF views retrieve original and token-sorted name/address representations; rare word views incorporate token equivalences learned exclusively from fit-positive S1–S2 and S1–S3 links. Frequent n-grams are pruned using a cap scaled to each observed country pool. A GPU-trained XGBoost ranker uses cheap pair features to retain the final candidate set per reference. Exactly those candidates are passed to the matching models and written to candidate_pairs.tsv. Batched RapidFuzz and source-conditioned edit/alias likelihoods add rich pair evidence. A compact TabM neural pair model is trained with hard negative candidates, then compared with a rich-feature GPU tree model. The winner or blend is selected by tuning macro F0.5.

All model weights are trained from the supplied data. There is no external business lookup or pretrained business model. The attention network is far below eight billion parameters. The pipeline code is MIT; XGBoost is Apache-2.0. Country labels are open strings and France has no labels during fitting or tuning.

## Candidate generation

Country-specific name and address character 3-4 gram retrieval is searched independently in original and token-sorted views, using fused top-k sparse products. Additional rare word views include learned source-specific token aliases and address-number tokens if they improve tuning retrieval recall. Adaptive frequency filtering prevents common sequences from creating huge candidate sets. A GPU ranker sorts the union using cosine similarity, length, exactness, address number and source features. High-confidence target anchors provide one bounded second hop. The final candidates are the exact pair set used by the matcher. Refer to retrieval_config.json and candidate_pairs.tsv for actual breadth and counts.

## Model architecture

The ranker is gradient boosted trees on cheap features. The final Apache-2.0 TabM network reads lexical, numeric, rank, source-conditioned noisy-channel and alias features and outputs a binary probability per candidate. One TabM control uses random negatives; the other mines the highest-scoring false candidates. A separate GPU boosted-tree model sees the same rich features. A tuning sweep compares these models and blends; scores support multiple true matches and singletons. The original 60,000-reference split remains fixed, with additional training references inserted only into the fit fold. Tuning and untouched holdout entities are unchanged.

## Validation and limitations

Held-out macro F0.5: {metrics['holdout_macro_f05']:.6f}. Candidate recall: {metrics['candidate_recall']:.6f}. Micro precision: {metrics['micro_precision']:.6f}. Micro recall: {metrics['micro_recall']:.6f}. Selected model/blend: {selected['model']} with weights {selected['weights']}. Threshold: {selected['threshold']:.4f}. Retrieval and matching are evaluated against the entire training target corpus. France has no supplied labels; its test accuracy is not measurable locally. This result is a local holdout, not a leaderboard score. Component selection used the tuning fold; the locked holdout was not used to choose thresholds or blends.

{ablation_table}

Reciprocal ownership is evaluated within the generated candidate graph, including reverse rank, mutual top-k and competing-reference score/margin. If it reduces tuning F0.5, it remains disabled. The optional ambiguous-tail model is `Alibaba-NLP/gte-multilingual-reranker-base` (306M parameters, Apache-2.0), with weights pinned to `8215cf04918ba6f7b6a62bb44238ce2953d8831c` and separate Apache-2.0 remote code pinned to `40ced75c3017eb27626c9d4ea981bde21a2662f4`. It sees only a small uncertainty-selected subset. A single high-confidence sibling propagation pass is measured separately and enabled only when it raises tuning F0.5 without sacrificing precision.

## Reproduction

See code/business_entity_resolution/README.md and run_config.json. The notebook/source implement input loading, blocking, model fitting, inference, exact candidate export, strict audit and plotting. validation.json and figures/plot_data.json preserve the graph measurements.
"""
    method=root/"Documentation_template.md";method.write_text(text,encoding="utf-8")
    destination.parent.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv","candidate_pairs.tsv"):
            z.write(output/name,f"output/{name}")
        for module in ("gpu100","ber"):
            for p in (project/"src"/module).rglob("*.py"):
                if "__pycache__" not in p.parts:
                    z.write(p,(Path("code/business_entity_resolution/src")/module/p.relative_to(project/"src"/module)).as_posix())
        for local,archive in (("GPU100_README.md","README.md"),("requirements-kaggle.txt","requirements.txt"),("LICENSE","LICENSE")):
            p=project/local
            if not p.exists():raise FileNotFoundError(p)
            z.write(p,f"code/business_entity_resolution/{archive}")
        z.write(root/"hyperparameters.json","code/business_entity_resolution/run_config.json")
        for name in ("validation.json","audit.json","prediction_summary.json","retrieval_config.json","environment.json","source_manifest.json","dataset_manifest.json","runtime.json","compaction.json","noisy_channel_profile.json","split.json","tail_config.json","collective_config.json"):
            if (root/name).exists():z.write(root/name,f"code/business_entity_resolution/reports/{name}")
        for p in (root/"figures").glob("*"):
            if p.is_file():z.write(p,f"code/business_entity_resolution/reports/figures/{p.name}")
        z.write(method,"Documentation_template.md")
    return destination
