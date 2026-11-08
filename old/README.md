# Archived files

Code and results from earlier stages of the project that the final pipeline no longer uses. They are kept for reference and are not maintained.

- `lib.py`: the original shared library. The parts the final pipeline uses now live in `src/dataset.py` and `src/model.py`.
- `train-feature.py`, `test_model.py`, `pruning-feature.py`, `test.py`: training and testing of the earlier feature MLP and CNN variants.
- `balance.py`, `find_balance.py`: searches for a balanced train/validation split.
- `data_analysis.py`, `analysis.md`, `images/`: earlier data analysis, replaced by `notebooks/EDA.ipynb`.
- `setup.txt`: the earlier conda setup, replaced by `requirements.txt`.
- `results/cnn_metrics.csv`, `results/mlp_metrics.csv`: early results, replaced by `results/experiments.csv`.
- `README_task1.md`: the README from Task 1.

The scripts expect `lib.py` in the same folder and the data one level up, so they no longer run from here without changes.
