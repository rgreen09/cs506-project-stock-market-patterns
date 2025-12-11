Stock Market Pattern Pipeline
=============================

Small-test workflow (keeps default on full dataset)
---------------------------------------------------

1) Create a tiny subset (about 2 days by timestamp) from the combined CSV:
```
python -m src.utils.create_subset \
  --input data/combined_dataset.csv \
  --output data/test/combined_subset.csv \
  --days 2 \
  --timestamp-col timestamp
```
Optional: add `--start-date YYYY-MM-DD` to anchor the window elsewhere.

2) Run the pipeline on the subset (default still uses the full dataset unless you pass `--use-subset` or `--input-path`):
```
# Build+combine+train for all enabled patterns using the subset
python -m src.main run --all --use-subset

# Or target a single pattern
python -m src.main run --pattern triangle --use-subset
```

3) To switch back to the full dataset, omit the flag:
```
python -m src.main run --all
```
