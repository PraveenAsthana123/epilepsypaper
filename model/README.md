# Model
**RandomForest(300, balanced)** on engineered EEG features (see accuracy/README.md).
The model is defined + trained reproducibly in `code/reproducible/chbmit_loso_pipeline.py`.
No pickled binary is committed (subject-dependent,
retrained per LOSO fold); run the pipeline to regenerate. Real per-fold metrics in
`accuracy/chbmit_loso_results.json`.
