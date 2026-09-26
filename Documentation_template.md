# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** {{TEAM_NAME}}  
**Team Members:** {{TEAM_MEMBERS}}  
**Submission Date:** 26 September 2026

---

## 1. Executive Summary
We treat the task as record-to-entity assignment: every Source 2/3 record belongs to at most one Source 1
entity.
- **Candidates** come from a two-stage sparse TF-IDF retrieval over country-scoped tokens that runs in both
  directions (target → S1 and S1 → target).
- **Scoring** uses a cross-fitted gradient-boosted model. Its strongest signals are *competition* features (how
  a pair ranks among all S1 entities claiming the same target) and *sibling-support* features (whether the
  target's differences from the S1 are shared by other copies from the other source).
- **Decision**: a precision-oriented rule keeps each target only for its best S1 and applies a threshold tuned
  for macro F0.5.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Size:** train has 2.21M S1, 5.03M S2, 5.29M S3 records and 7.64M true pairs; 5.6% of S1 are singletons,
  with 3.46 matches per S1 on average.
- **Many-to-one structure:** every matched S2/S3 record belongs to exactly one S1, and 100% of true pairs share
  the country label. About 26% of S2/S3 records are distractors: businesses without an S1 record.
- **Higher distractor density in test:** test has more S2/S3 records per S1 than train (5.8 vs 4.7). We
  therefore train on a graph where 19% of train S1 are removed, so their copies become distractors
  (5.77 targets per S1, matching test).
- **Hard negatives are sibling businesses:** the same street, a house number a few units away, often one
  different name word, and their own S2/S3 copies.
- **Name noise:** case, typos, leetspeak (`Téchn0logies`), injected accents, reordering, legal-form changes,
  truncation, aliases (dba/aka/fka), domains and handles, 2-letter acronyms, and native-script names (Indian
  names written in Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, etc.).
- **Address noise:** abbreviations, state code ↔ name, missing components, landmarks, number prefixes and typos,
  null tokens.
- **France** appears only in test, with accents, French legal forms and street abbreviations.

### 2.2 Solution Strategy
**Approach Type:** Blocking (two-directional sparse retrieval) + gradient-boosted pair classifier + graph-constrained
decision (optionally stacked with a collective second stage).  
**Core Innovation:**
1. Test-density-matched training.
2. Many-to-one competition features over the full candidate graph.
3. Cross-source sibling-support features in both directions: the target's extra tokens, and the S1 tokens the
   target lacks.
4. Two-stage retrieval: rare-token pools re-ranked by exact cosine.

---

## 3. Candidate Generation (Blocking)
- **Normalization:**
  - Native-script tokens are mapped to Latin with a table learned from train ground-truth pairs, since native
    names are token-for-token transliterations. AnyAscii handles the rest.
  - Names: alias split, dotted acronyms, leetspeak repair, legal-form removal for the core name.
  - Addresses: abbreviations, state names → codes, null tokens, ordinals.
- **Blocking keys (tokens):**
  - Tokens are country-scoped (`country|kind|value`), so the country label is only a partition key and unseen
    labels (France) work unchanged.
  - Kinds: name word, consonant skeleton, concatenated core name and its 6-character prefix (catches domains and
    handles), address word, address number.
  - TF-IDF weights are computed per country, with separate unit vectors for name and address.
- **Retrieval:** a numba sparse top-k.
  - Each query spends a posting budget on its rarest tokens. Wide candidate pools are then re-ranked by exact
    cosine over all tokens.
  - **Reverse view:** each S2/S3 record keeps its top 8 S1.
  - **Forward views:** each S1 keeps its top 12 targets by name+address, top 6 by name, and top 6 by address.
- **Candidate pairs generated (test):** 90,811,167 (52.4 per S1). `candidate_pairs.tsv` is exactly the set
  scored by the model.
- **How we ensured true matches were not lost:** the reverse view follows the many-to-one structure: a target's
  own S1 is almost always among its top few (R@1 0.965, R@8 0.982). The union of views reaches train pair
  recall **0.988** (oracle macro F0.5 0.996).

---

## 4. Matching Model

**Features used (~110):**
- **Name:**
  - exact TF-IDF cosine; token Jaccard; IDF coverage on each side; rarest shared / unshared token IDF
  - skeleton and concatenated-name overlap; acronym match
  - RapidFuzz ratio, token_sort, token_set, partial and Jaro-Winkler on full, core and concatenated names
- **Address:**
  - exact TF-IDF cosine; word and number overlap statistics
  - first house number: equal, absolute difference, sibling band 3..21; minimum difference over all numbers;
    Levenshtein on the house number
  - RapidFuzz ratio, token_set and token_sort
- **Competition (full candidate graph, label-free):** rank, gap and margin of the pair
  - among all S1 claiming the same target
  - among the S1's own candidates
  - for combined, name and address cosines
- **Sibling support:** among the S1's other candidates **from the other source**, count how many share the
  target's extra tokens, and how many carry the S1 tokens the target lacks.
- **Ambiguity / structure:** number of S1 entities and targets sharing the core name; script, alias, domain and
  empty-address flags; lengths; source.
- **Stacked stage (later submissions):**
  - the stage-1 probability, and its ranks and margins within the S1 and the target
  - agreement with the S1's confident co-candidates
  - a copy-count prior for ambiguous targets

**Model type:**
- LightGBM (255 leaves, lr 0.06, early stopping). Two models are cross-fitted on disjoint S1 folds, and test
  uses their average.
- Easy negatives are down-sampled, with weights restored.
- Later submissions also ensemble XGBoost models.

**Threshold selection method:**
1. Each target is kept only for its highest-probability S1.
2. τ maximizes exact macro F0.5 on out-of-fold predictions at test-like density, including singletons and
   blocking misses. τ = 0.75.
3. Scores are reported on an untouched validation fold.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9835** on the validation fold (US 0.9849, India 0.9813); out-of-fold 0.9832.
  The oracle given the candidates is 0.9959.
- **Common false positives (wrong merges):** sibling businesses: the same street, a house number a few units
  away and one different name word. Also chains or generic names at nearby addresses.
- **Common false negatives (missed matches):**
  - records with an empty address and a generic name shared by several S1 entities (not decidable from the
    name alone; F0.5 favours abstaining)
  - aliases with only a house number
  - targets outside the candidate set

---

## 6. Conclusion
Framing the problem as many-to-one assignment gave three things: an efficient reverse retrieval, the most
important features (competition among S1 for a target), and an exact decision constraint. Matching the test's
distractor density in training and adding sibling-support signals targets the dominant error: false merges
with lookalike businesses.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/ber5.py` is a single file. `README.md` gives the exact command:
`python src/ber5.py --data <dataset> --work <dir> --stage all --cfg '<json>'`.

Main functions:
- `load_split`, `load_gt`: TSV loading.
- `learn_translit`: learned transliteration table.
- `norm_name`, `norm_addr`, `_norm_chunk`: normalization and tokens.
- `build_graph`: TF-IDF, retrieval (`_retrieve`), competition features.
- `featurize`, `_pair_tok`, `_support`: pair features. `collective`, `target_p_stats`: stacked stage.
- `run_train`: cross-fit, threshold, validation report in `work/model/meta.json`.
- `run_test`: scoring, decision, `work/output/matching_results.tsv` and `candidate_pairs.tsv`.

No external data, APIs or pretrained weights are used; all models are trained from scratch on the supplied data.

### B. Additional Results
| retrieval | pair recall | candidates / S1 |
|---|---|---|
| reverse view only (top 8) | 0.982 | 46 |
| union, budget 10k/30k postings | 0.987 | 52.9 |
| union, budget 25k/80k postings | 0.990 | 52.7 |
