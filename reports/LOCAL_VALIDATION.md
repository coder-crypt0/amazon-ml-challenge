# Local validation evidence

- Supplied training pool: 10,320,219 target records; 173,921,530 blocking postings.
- Sampled references: 12,000; fitting 8,400, calibration 1,800, untouched holdout 1,800.
- Macro F0.5: 0.907635.
- Micro precision: 0.986662; micro recall: 0.812868.
- Candidate recall: 0.881351.
- Selected ensemble: equal LightGBM and XGBoost weights, threshold 0.70.
- Exclusivity did not improve this tuning sample and remains disabled.
- 12 automated tests passed. All 26 notebook cells executed successfully on a synthetic fixture, including checkpoint recovery, plots, audit and ZIP packaging.
- Real-data fitting used CPU. Actual Colab authentication, Drive service behavior and CUDA execution were not available locally and remain to be verified in Colab.
- No test-set labels or leaderboard score were available. France accuracy is unmeasured. India retrieval is the main measured weakness.

The downloadable notebook uses 60,000 sampled references and 500 trees by default. These are configured cloud settings, not the settings behind the local score above.
