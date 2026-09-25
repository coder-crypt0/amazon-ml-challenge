"""Write the v3 methodology from measured results and a verified submission package."""
from datetime import date
import json
from pathlib import Path
from .audit import audit_outputs
from .features3 import MODEL_NAMES
from .package import package_submission


def finalize(data, artifacts, output, project, team="entity_resolution", members=(), label=""):
    artifacts, output, project = Path(artifacts), Path(output), Path(project)
    audit = audit_outputs(output / "matching_results.tsv", output / "candidate_pairs.tsv", Path(data) / "test")
    (output / "audit.json").write_text(json.dumps(audit, indent=2))
    if not audit["ok"]:
        raise ValueError(f"Output audit failed: {audit['errors'][:5]}")
    report = json.loads((artifacts / "experiment/validation.json").read_text())
    retrieval = json.loads((artifacts / "experiment/retrieval.json").read_text())
    summary = json.loads((output / "prediction_summary.json").read_text())
    translit = json.loads((artifacts / "translit.json").read_text(encoding="utf-8"))
    h, countries = report["holdout"], report["holdout_by_country"]
    by_country = "; ".join(f"{c} {m['macro_f05']:.4f}" for c, m in countries.items())
    text = f'''# ML Challenge 2026: Business Entity Resolution

**Team:** {team}
**Members:** {", ".join(members) if members else "Not specified in run configuration"}
**Date:** {date.today().isoformat()}

## 1. Executive Summary

Learned-transliteration IDF token retrieval feeds a LightGBM/XGBoost pair scorer with retrieval-aware, house-number and candidate-context features. It is tuned for the challenge's macro F0.5, including singleton references. All business identity evidence comes from the supplied files: no external lookups and no pretrained language models.

## 2. Methodology

### 2.1 Problem Analysis

- Indian Source 2/3 names often appear in native scripts (Devanagari, Gujarati, Telugu, Bengali and others) as token-for-token transliterations of the reference name; state names are also written natively.
- Other name noise: legal-form changes and reordering, duplicated or dropped tokens, accents, typos, bracketed words, junk prefixes, domain or hashtag forms (`cozyyoga.com`, `#perfectsystems`), aliases (`formerly:`, `f/k/a`, `doing business as`) and fully rebranded names at the same address.
- Unmatched Source 2/3 records are mostly near-duplicate sibling businesses: one name word changed and the house number shifted by a small amount, each with its own noisy copies. True copies instead show digit-level noise (substitution, truncation, zero padding, suffix letters, ranges).
- Every matched target belongs to exactly one reference in the training labels (match counts sum to the number of distinct matched targets).
- Test-only France follows the same patterns with French legal forms, street abbreviations and region/département substitutions. Country is treated as an open string.

### 2.2 Solution Strategy

**Approach type:** blocking (retrieval) + gradient-boosted pair classifier + calibrated threshold.
**Core contribution:** a transliteration table learned from training pairs, IDF-scored token retrieval that does not rely on any single exact key, and features that distinguish sibling businesses from noisy copies.

## 3. Candidate Generation (Blocking)

- **Normalization:** native-script tokens are mapped with a table learned from aligned training pairs ({len(translit["name"]):,} name tokens, {len(translit["addr"]):,} address tokens; sampled validation entities excluded), then AnyAscii. Alias variants are split, the concatenated name is kept for domain forms, address abbreviations are expanded, and house numbers lose leading zeros and ordinal suffixes.
- **Tokens:** country-scoped name tokens, concatenated-name stems, address words and numbers.
- **Scoring:** a candidate's score is the summed IDF of the tokens it shares with the reference, over a sorted, memory-mapped posting index. Tokens with more than {retrieval["df_max"]:,} postings do not generate candidates.
- **Channels:** the top {retrieval["k_all"]} targets by all tokens, plus the top {retrieval["k_name"]} further targets by name tokens alone, which recovers missing or rewritten addresses. These are exactly the pairs scored by the model and exported in `candidate_pairs.tsv`.
- **Candidate pairs generated:** {audit["counts"]["candidate_ids"]:,} for {audit["counts"]["s1"]:,} test references.
- **How losses were measured:** holdout candidate recall is {h["candidate_recall"]:.4f}. Validation references search the complete training target pool, so blocking misses count as false negatives.

## 4. Matching Model

**Features ({len(MODEL_NAMES)}):**
- **Base string similarities:** raw, transliterated and legal-suffix-stripped name ratios; token-set, sort and partial ratios; Jaro-Winkler; token containment; address similarity; digit and postcode agreement; missingness.
- **Retrieval-aware:** IDF-weighted shared, extra and missing name and address evidence; IDF-weighted fuzzy token alignment; concatenated-name ratio and partial ratio for domain forms; native-script, domain and empty-address flags.
- **House numbers:** exact agreement, membership, log absolute difference, edit distance, and a small-forward-shift indicator for sibling businesses.
- **Context within the reference's candidate list:** retrieval rank and relative score, channel, number of near-duplicate names, number of candidates sharing this candidate's house number, and best competing name similarity.
- **Name-word odds:** smoothed log-odds for words present on only one side, learned from training pairs. Fit pairs receive out-of-fold values.

Record identifiers are never features, and country is never one-hot encoded.

**Model type:** LightGBM (MIT){" + XGBoost (Apache-2.0), ensemble weight " + format(report["xgboost_weight"], "g") if report["xgboost_weight"] else ""}, trained from scratch; far below the 8B-parameter limit.
**Threshold selection method:** reference entities are split 70/15/15 into fit/tune/holdout. The probability threshold ({report["threshold"]:.3f}) and ensemble weight maximize tuning macro F0.5; the holdout is reported untouched. Target exclusivity in this output: {summary.get("exclusive", False)}.

## 5. Results & Error Analysis

- **F0.5 score (macro, untouched holdout of {h["queries"]:,} references):** {h["macro_f05"]:.4f} ({by_country})
- **Micro precision / recall:** {h["micro_precision"]:.4f} / {h["micro_recall"]:.4f}; singleton accuracy {h["singleton_accuracy"]}
- **Common false positives:** sibling businesses whose house numbers differ only slightly, and same-name records without an address.
- **Common false negatives:** fully rebranded names identified only by address, generic names without an address, and heavily truncated native-script records.

These are local validation measurements, not leaderboard results. France has no labels, so its accuracy is unmeasured; the features are country-agnostic.

## 6. Conclusion

The largest gain came from retrieval: IDF token search with learned transliteration raises the recall ceiling, and sibling-aware features let the matcher use the extra candidates without losing precision. Remaining uncertainty is unseen-country transfer.

## Appendix

### A. Code Artefacts

Source is under `code/business_entity_resolution/src/ber/`. Entry point: `python -m ber.reproduce3 --data <dataset> --device cpu` (see README.md and run_config.json). Stages: `translit` (learned table), `retrieval` (token index), `features3`, `pipeline3` (candidates, training), `predict3` (scoring), `finalize3` (this document and packaging).

### B. Additional Results

`reports/validation.json` holds threshold sweeps, country slices and feature gains. Regenerate figures with `python -m ber.plots --report reports/validation.json --output reports/figures`.
'''
    documentation = project / "Documentation_template.md"
    documentation.write_text(text, encoding="utf-8")
    reports = project / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "validation.json").write_text(json.dumps(report, indent=2))
    target = output.parent / f"{team}{label}_submission.zip"
    package_submission(target, output, project, documentation)
    return target, audit
