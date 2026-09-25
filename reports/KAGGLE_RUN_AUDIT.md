# Kaggle CNER run audit (26 September 2026)

The resumed Kaggle run reused all 235 raw candidate batches from the failed first run. A bounded ranker fit 8,506,584 sampled pairs, including all 346,584 positives, and completed 400 trees. Final candidate enrichment and validation also completed. The first run's failure was a host-RAM kill while loading its unbounded ranker training matrix; the resume fixed that failure.

## Measured validation

The frozen 9,000-reference holdout contained 30,932 true links. The selected model (`random25_channel75`, threshold 0.675) achieved macro-F0.5 **0.907215**, micro precision **0.958653**, micro recall **0.847019**, and singleton accuracy **0.823204**. Candidate recall was **0.982090**. It missed 554 links in retrieval and 4,178 links in the matcher, and produced 1,130 false-positive links. India macro-F0.5 was 0.871288; US was 0.931568. This is below the prior reported Colab holdout 0.908684 and far below the user-reported qualifying score 0.984307.

Full-test inference processed roughly 33.5 Source-1 queries per second. Its own ETA exceeded 13 hours with 3.5 hours already elapsed, so the run could not finish within a 12-hour Kaggle GPU session. It was stopped. Kaggle preserved its completed model, validation, and partial test artifacts in the canceled version's output.

## Controlled local ablations from the saved validation candidates

`scripts/analyze_validation_decoding.py` used the saved Kaggle `validation_candidates.npz` output (18,000 tune-plus-holdout references). An oracle selecting only labeled true candidate links scored 0.993492 macro-F0.5; thus retrieval is close enough in principle. Giving the existing ranking the true cardinality for each entity scored only 0.925778. A small floor-count decoder tuned on the 9,000 tuning entities failed to improve the untouched holdout reliably. The remaining gap is predominantly pair ranking and discrimination.

`scripts/pilot_sibling_matcher.py` used the saved top-16 candidates per entity, original supplied names/addresses, and a new source-2/source-3 sibling-consistency feature set. It trained on 6,000 tuning entities, selected its threshold on the remaining 3,000 tuning entities, and evaluated on the same untouched 9,000-reference holdout. Its 24-feature XGBoost variant scored **0.916383** versus **0.907215** for the saved top-16 baseline, a measured gain of 0.009168. The 14-feature version scored 0.911705. This is a useful ablation, but it remains 0.067924 below the reported qualification cutoff. The full test runtime would also require redesign; this pilot was not promoted to a final inference run.

No leaderboard score was obtained for the canceled Kaggle run. These are held-out scores and controlled pilot results, not claims of qualification.
