# Business Entity Resolution

Resumable multi-pass retrieval, 47-feature LightGBM/XGBoost matching, macro-F0.5 calibration, and validation-selected target exclusivity. Uses supplied data only. Country labels are open strings, including unseen France.

## v3: IDF token retrieval (recommended)

v3 learns a native-script → Latin token table from training pairs (`ber.translit`), retrieves candidates by summed IDF of shared name/address/house-number tokens (`ber.retrieval`, top-50 overall plus top-10 by name), and scores 92 features: the 47 base features plus IDF, house-number, domain-form, candidate-context and out-of-fold name-word odds features (`ber.features3`). Local 5-fold validation on 10k training references: candidate recall 0.899 → 0.967 and macro F0.5 0.912 → 0.969 against a replica of the earlier pipeline. France remains unmeasured.

Colab: open `Business_Entity_Resolution_Colab_v3.ipynb` (build it with `python scripts/build_notebook_v3.py`), edit `DATA_ZIP`, run all cells. It writes `output/` (threshold) and `output_exclusive/` (each target assigned to at most one reference), each with an audited package.

Local, one command (set `PYTHONPATH=src`; `run_config.json` supplies samples, trees, workers and seed):

```sh
python -m ber.reproduce3 --config run_config.json --data /path/to/dataset --device cpu [--exclusive]
```

Stages: `python -m ber.pipeline3 candidates|train`, `python -m ber.predict3 score|assemble`.

## Colab

Open `Business_Entity_Resolution_Colab.ipynb`. Upload the original student-resource ZIP to Drive, select a GPU runtime, and edit `DATA_ZIP` in the configuration cell. Run cells in order. The notebook embeds the complete source; no GitHub login or separate code upload is needed.

Keep approximately 12 GB free in Drive for the archive, checkpoints, and outputs, and at least 25 GB free in the runtime. Retrieval uses CPU/RAM; the optional XGBoost model uses CUDA. A GPU does not accelerate every pipeline stage. Runtime depends heavily on CPU allocation and Drive throughput.

After a disconnect, open the same notebook, keep the same `RUN_NAME` and settings, and run from the top. Completed stores/indexes are restored. Retrieval shards, candidate feature batches, 50-tree training segments, and prediction batches are checkpointed. The currently running unit may need to repeat. Changing source/configuration requires a new run name. Do not run two notebooks against the same run directory concurrently.

## Local setup

Python 3.12 is tested. Create a virtual environment and install `requirements.txt`. Extract the student resource archive so `dataset/train` and `dataset/test` are available. Set `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH='src'`).

```sh
python -m pip install -r requirements.txt
python -m ber.reproduce --config run_config.json --data /path/to/dataset --device cpu
```

The notebook writes `run_config.json` into the final package with the exact experiment parameters. CUDA training uses `--device cuda`; `none` trains LightGBM only. The reproducibility configuration includes sample size, seed, tree count, top-k, posting cap, workers and batch size.

Individual stages:

```sh
python -m ber.build --data student_resource/dataset --split train
python -m ber.pipeline candidates --samples 60000 --workers 4
python -m ber.pipeline train --trees 500 --xgboost-device cuda
python -m ber.experiments --country-transfer
python -m ber.build --data student_resource/dataset --split test
python -m ber.predict score
python -m ber.predict assemble
python -m ber.audit --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
python -m ber.plots --report artifacts/v1/experiment/validation.json --output reports/figures
```

For durable checkpoints outside Colab, set `BER_ARTIFACT_ROOT` to the absolute local artifact directory and `BER_CHECKPOINT_ROOT` to the absolute backup directory.

## Validation and limitations

Sampled Source 1 entities split 70/15/15 into fit/tune/holdout. Every sampled query retrieves against all training Source 2/3 records. Threshold and ensemble selection use tuning only. Metrics include complete reference coverage, candidate misses and singletons. IDs are row keys, never features. France has no labeled validation score; optional leave-country-out evaluation is diagnostic only.

The graph constraint allows multiple targets per reference but at most one reference per target. It is enabled only if tuning macro-F0.5 improves; ties abstain. `decision.py` contains a tested exact expected-F0.5 experiment, excluded from the default prediction path.

Check `reports/validation.json` for measured results. A local holdout is not a leaderboard score. There is no claim of winning or algorithmic novelty.

## Tests

```sh
python -m pip install pytest==8.4.2
python -m pytest tests -q
```

Tests cover Unicode, identifier invariance, candidate caps, macro scoring, exact expected utility, unknown-country outputs, model fitting, checkpoint recovery, inference and output audit. The audit checks the existence of all target IDs and match/candidate containment.

## Artifacts and licenses

The final ZIP contains the two required TSVs, runnable source, pinned requirements, run configuration, completed methodology and aggregate reports/figures. PNG/SVG figures can be recreated from JSON. Competition data and intermediate artifacts stay out of Git. LightGBM is MIT-licensed; XGBoost is Apache-2.0. Both models are trained from scratch and are well below the 8-billion-parameter limit.

GPU API reference: https://xgboost.readthedocs.io/en/stable/gpu/
