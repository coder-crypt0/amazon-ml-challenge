"""Write measured methodology and a verified submission package."""
from datetime import date
import json
from pathlib import Path
from .audit import audit_outputs
from .package import package_submission


def finalize(data,artifacts,output,project,team="entity_resolution",members=()):
    artifacts,output,project=Path(artifacts),Path(output),Path(project)
    audit=audit_outputs(output/"matching_results.tsv",output/"candidate_pairs.tsv",Path(data)/"test")
    (output/"audit.json").write_text(json.dumps(audit,indent=2))
    if not audit["ok"]:
        raise ValueError(f"Output audit failed: {audit['errors'][:5]}")
    report=json.loads((artifacts/"experiment/validation.json").read_text())
    calibration=json.loads((artifacts/"experiment/calibration.json").read_text())
    retrieval=json.loads((artifacts/"experiment/retrieval.json").read_text())
    holdout=report["holdout"]
    selected=calibration.get("exclusive",False)
    if selected:
        holdout=report["graph_ablation"]["holdout"]["exclusive"]
    text=f'''# ML Challenge 2026: Business Entity Resolution

**Team:** {team}  
**Members:** {", ".join(members) if members else "Not specified in run configuration"}  
**Date:** {date.today().isoformat()}

## 1. Executive Summary

This hybrid pipeline uses country-aware multi-pass retrieval, a supervised string-similarity ensemble, and a validation-selected target-exclusivity constraint. It optimizes the challenge's macro F0.5 including singleton entities. All business identity data comes from the supplied files; there are no external business lookups or pretrained language models.

## 2. Methodology

### 2.1 Problem Analysis

Training contains US and India; France occurs only in test. Country values are open strings, not a fixed model category. Names undergo accents, suffix variations, reordering, typos and cross-script transformations; some target addresses are missing. Multiple target records may match a reference; predicting an empty set is explicitly supported.

### 2.2 Solution Strategy

The design combines complementary text retrieval channels with a learned pair scorer and set-level evaluation. The structural extension assigns a target to at most one reference when that constraint improves tuning performance. This is a tested engineering combination, not a claim of a previously unseen algorithm.

## 3. Candidate Generation (Blocking)

Unicode-preserving normalization and AnyAscii transliteration feed normalized name, sorted name, address, sorted address, salient name token/pair/prefix, and house-number-plus-street/locality keys. Keys are country-prefixed and hashed deterministically. Sorted on-disk postings support bounded memory. Keys with more than {retrieval["max_postings"]} postings are skipped in their entirety; no biased prefix of a frequent block is used. A field-similarity ranker retains at most {retrieval["top_k"]} targets per reference. These are exactly the pairs scored by the ML model and exported in candidate_pairs.tsv.

Final test candidates: {audit["counts"]["candidate_ids"]:,}. Holdout candidate recall: {holdout["candidate_recall"]:.6f}. Blocking losses are measured, not assumed absent.

## 4. Matching Model

There are 47 pair features: raw/transliterated and legal-suffix-stripped name similarities, token order/containment, Unicode similarity, address similarities, number conflicts, house/postal agreement, missingness and field interactions. Numeric record identifiers are never features. Country identity is not one-hot encoded.

LightGBM (MIT) and optional XGBoost (Apache-2.0) are trained from scratch. XGBoost ensemble weight: {calibration.get("xgboost_weight",0):g}. The model has no pretrained transformer parameters and is far below the 8B limit. The reference-entity split is 70% fitting, 15% tuning, 15% untouched holdout. Candidate search uses the complete target corpus. Ensemble weight and threshold ({calibration["threshold"]:.6f}) are selected using tuning macro F0.5 only. Target exclusivity selected: {selected}.

## 5. Results & Error Analysis

- Held-out reference queries: {holdout["queries"]:,}
- Macro F0.5: {holdout["macro_f05"]:.6f}
- Micro precision: {holdout["micro_precision"]:.6f}
- Micro recall: {holdout["micro_recall"]:.6f}
- Singleton accuracy: {holdout["singleton_accuracy"]}

These are local validation measurements, not leaderboard results. France has no supplied training labels, so its test accuracy remains unmeasured. reports/validation.json records country slices, retrieval coverage, threshold/ensemble sweeps, feature gains and structural ablations. Country transfer, when enabled, is a diagnostic rather than a direct estimate of France performance. False negatives include candidate-generation losses; pairwise validation alone would hide them.

## 6. Conclusion

The pipeline prioritizes measured precision, complete reference coverage, and reproducibility. Persistent chunk checkpoints support interrupted Colab sessions. Remaining uncertainty concerns unseen-country transfer and leaderboard distribution differences.

## Appendix A. Code Artifacts

All code is under code/business_entity_resolution/src/. See README.md and run_config.json for exact configuration and the reproduce command. requirements.txt pins dependencies. Hash manifests identify the source version and supplied archive used by the notebook.

## Appendix B. Additional Results

The package includes aggregate metrics and PNG/SVG figures with plot_data.json. Regenerate plots with `python -m ber.plots --report reports/validation.json --output reports/figures` after setting PYTHONPATH=src.
'''
    documentation=project/"Documentation_template.md"
    documentation.write_text(text,encoding="utf-8")
    reports=project/"reports";reports.mkdir(exist_ok=True)
    (reports/"validation.json").write_text(json.dumps(report,indent=2))
    target=output.parent/f"{team}_submission.zip"
    package_submission(target,output,project,documentation)
    return target,audit
