# CNER research and ablation evidence

The earlier 60,000-reference Colab run scored **0.902 on the public leaderboard** (user-reported). Its 9,000-reference local holdout had macro F0.5 **0.9086843491** and positive candidate recall **0.8807383939**. Replaying the original reservoir sampler and seed recovered **30,932** true links in that exact holdout. The CNER notebook preserves those 9,000 tuning and 9,000 holdout entities; additional references enter fitting only. Scores on a different split must not be compared as if controlled.

## Why candidate retrieval changed

Among 745 known links missed by the old top-32 blocker on a separate 1,800-reference local holdout, 375 shared only over-frequency keys, 293 shared a usable key but were dropped by ranking, and 77 shared no generated key. Widening the old final candidate set to 256 raised candidate recall from 0.8814 to only 0.9277. This established blocking as the first ceiling.

The retrieval design follows [Sparkly's TF-IDF blocking study](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-tr22.pdf) and incorporates source-learned token variants. The [SC-Block paper](https://arxiv.org/abs/2303.03132) motivated measuring pair completeness before matcher quality. We built country-partitioned name/address character views and rare-word/number views. Country labels are open strings, including France.

On 1,000 pilot tuning references against **1.8 million** targets, four character views recovered 0.9808 of true links with 269 candidates/query; source-learned rare-word/number views raised recall to **0.9864** with 285 candidates/query. The word views were kept because that gain exceeded the predeclared tuning floor. These numbers are from the smaller pilot pool, not the complete challenge target pool. The notebook repeats the preflight against all **10.3 million** training targets.

The [sparse-dot-topn C++ implementation](https://github.com/ing-bank/sparse_dot_topn) fuses multiplication with top-k selection. Batched [RapidFuzz `cpdist`](https://rapidfuzz.github.io/RapidFuzz/Usage/process.html) avoids per-pair Python string scoring. In the 1.8-million-target pilot, adding name ratio and address token-set features to the GPU prefilter recovered 12 additional true links at top-64 across the held-out references.

## Source-conditioned noisy channel

The corruption tables are fitted from **fit-fold positive links only**. S1→S2 and S1→S3 have separate name/address character operation frequencies, recurring token substitutions and aliases. Weighted edit cost, operation support, token equivalence and repeated alias evidence are pair features. Target-side learned token variants can also help the rare-word blocker. The source prefix is an input source indicator; the numeric ID suffix is never a feature.

The Kaggle run supplements its frozen 42,000 original fit entities with additional fit-only references. It samples a further 300,000 training entities outside the sampled tuning/holdout set to fit the corruption channel; their labels never enter validation. The 6,000-reference local pilot could not test this additional-channel-data setting, so its benefit remains an explicit Kaggle ablation hypothesis.

The real-data 6,000-reference pilot used a 1.8-million-record distractor pool augmented with every positive target for those references. On its fixed 900-reference holdout, the rich-feature GPU tree rose from **0.9301** without noisy-channel features to **0.9412** with them; final candidate recall stayed **0.9852**. Random-negative TabM reached **0.9121**; hard-negative TabM reached **0.9164**, so mining helped TabM but neither TabM model beat the rich tree in this pilot. These pilot scores overestimate performance on the complete target pool and cannot be presented as leaderboard scores.

The [official TabM project](https://github.com/yandex-research/tabm) is Apache-2.0. Our implementation uses a compact k=8 model and mines the highest-scoring false candidates. Both random-negative and hard-negative variants share the same fit/tuning/holdout entities. The final matcher or blend is selected by tuning macro F0.5; the original 9,000-entity holdout is not used to pick weights or thresholds.

## Reciprocal and collective evidence

Candidate edges are also grouped from each S2/S3 target back to competing S1 references, exposing reverse rank, mutual-top-k and ownership margin. In the pilot, strict ranker-based ownership reduced tuning macro F0.5 from **0.9501** to **0.9460**; final-score ownership had no measured gain. The structural filter was disabled. It is tested again on the larger fixed split and used only if tuning improvement clears the configured floor.

The source2/source3 siblings of an S1 motivate one conservative propagation pass. It only considers candidates already scored by the matcher, requires a high-confidence sibling and compatible name/address evidence, and refuses conflicting house/postal numbers. It is enabled only if entity-level tuning F0.5 rises without a material precision drop.

## Ambiguous multilingual tail and runtime

The optional [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) has **305,959,681 parameters** and an Apache-2.0 license. A local CPU forward pass was verified. Both its model weights (`8215cf04918ba6f7b6a62bb44238ce2953d8831c`) and separately hosted Apache-2.0 implementation (`40ced75c3017eb27626c9d4ea981bde21a2662f4`) are pinned. It is fine-tuned on fit positives and mined negatives with classification plus margin loss, then applied to at most three candidates for roughly 3% of uncertain references. It is retained only after a tuning-fold gain and precision check. A Kaggle GPU fine-tuning run has **not** yet been verified.

[Kaggle's notebook specifications](https://www.kaggle.com/docs/notebooks) list four CPU cores, about 29 GB RAM on P100/T4x2, 20 GB saved working space and a 12-hour GPU session. The 5½-hour target is a design budget, not a verified full-run duration. The notebook saves stage timings and prediction ETA; it frees training-only cache after models and validation are durable. No external business identities, geocoders, registries or matching APIs are used.
