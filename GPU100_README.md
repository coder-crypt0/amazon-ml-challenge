# CNER: collective noisy-channel entity resolution

`CNER_Kaggle.ipynb` is self-contained. Attach the original student-resource ZIP as a **private Kaggle Dataset**, enable a CUDA GPU and Internet for package/model downloads, then run cells in order. Business identities come only from the supplied ZIP. The pretrained multilingual reranker is a generic Apache-2.0 model, not a business lookup service.

The training split reproduces the original Colab baseline: 42,000 fit references, 9,000 tuning references and 9,000 untouched holdout references. The fixed holdout contains 30,932 true links. Extra references sampled by this notebook enter the fit fold only. Every validation reference searches the full training Source 2/3 target corpus. Candidate recall and macro F0.5 are reported separately, with no-match references included.

Four character TF-IDF views retrieve original and sorted names and addresses by open country label. Two small word views add rare names, address numbers and token equivalents learned from **fit-positive** S1–S2 and S1–S3 links. A GPU ranker narrows the union; up to two high-confidence targets seed one bounded retrieval hop. `candidate_pairs.tsv` records the exact final set scored by the matcher.

The source-conditioned channel learns recurring name/address edits, character substitutions, token variants and aliases independently for S2 and S3. Its costs and evidence are added to lexical, numeric and retrieval features. A rich GPU tree without these features is an ablation control. Apache-2.0 TabM models compare random-negative and mined hard-negative training; tuning selects the best matcher or blend. All decisions, including thresholds, use entity-level macro F0.5 on the tuning fold. The 9,000-entity holdout is reported separately.

Reciprocal ownership ranks S1 candidates for each target in the generated candidate graph. A conflict filter is used only if it improves tuning. For an ambiguous 3% of references, a 306M-parameter Apache-2.0 multilingual cross-encoder is fine-tuned on fit-positive and mined-negative pairs using classification plus ranking loss. Its base weights and separate remote code are pinned by commit. It is disabled if tuning shows no gain or if model setup fails; failure is recorded. A final bounded sibling-evidence pass for borderline matches is also gated by tuning and precision. France remains an open country with no labeled accuracy estimate.

The notebook records an ablation table, country metrics, runtime, plots and plot-source JSON. Checkpoints live in `/kaggle/working/cner_v1`; it frees training-only caches after the fitted models and reports are durable. Re-running the same notebook skips completed model stages and test batches. Kaggle documents a 12-hour GPU session; 5½ hours is the requested target, not an observed guarantee.

## Reproduce from the final code package

Install pinned packages from `requirements.txt` with a CUDA-enabled PyTorch runtime. Set `PYTHONPATH=src` and run:

```sh
python -m gpu100.runner all --data /path/to/dataset --root /path/to/run --output /path/to/output --device cuda --samples 120000
```

The final ZIP contains both required TSVs, code, run configuration, methodology, strict audit and aggregate figures/JSON. The neural model uses far fewer than eight billion parameters. XGBoost, TabM and the optional Alibaba reranker meet the challenge's Apache-2.0 license rule; project code is MIT. No geocoding, government registry, commercial matching service or external business identity lookup is used.
