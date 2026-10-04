# SCSE3040 — Practical 06: Remembering Every Experiment

**Name:** Shubh Om Yadav
**Roll number:** s24cseu0529
**Course:** SCSE3040 — Machine Learning Operations (Bennett University, Session 2026-27)

Experiment tracking and the model registry with **MLflow**. Every training run is recorded
(parameters, metrics, artifacts) in a SQLite tracking store, the best run is found by querying
the store, and the winning model is registered under a name and version.

## Repository structure

```
.
├── P06-mlflow/
│   ├── P06.ipynb            # walkthrough + tasks T1, T2, T3 (executed, outputs visible)
│   ├── README.md            # practical handout
│   └── work/
│       └── mlflow.db        # SQLite tracking store (parameters, metrics, tags, registry)
└── data/
    └── delivery_times.csv   # the shared 600-row delivery dataset
```

The trained-model artifacts (`mlruns/`) are not committed (they are large); the notebook rebuilds
them when it is run.

## What was done

1. Pointed MLflow at a SQLite database instead of the legacy `./mlruns` folder store.
2. Logged a first run, then a sweep of `RandomForestRegressor` settings.
3. Found the best run by querying: `mlflow.search_runs(order_by=["metrics.mae_minutes ASC"])`.
4. Saved the winning model as an artifact and registered it as `delivery-time-model`.
5. Loaded it back by name and version.
6. Ran the MLflow UI (`python -m mlflow ui --backend-store-uri sqlite:///.../work/mlflow.db`).

## Tasks completed

- **T1** — logged a `linear-baseline` run (LinearRegression, `model_type` parameter,
  `mae_minutes` metric) and stored its run id.
- **T2** — used `mlflow.search_runs` to find the lowest-`mae_minutes` run by query.
- **T3** — logged a tagged (`family=tree`) `DecisionTreeRegressor` sweep at depths 2, 4, 6 and
  counted the tagged runs by filtering.

## Result

The best run was **linear-baseline** with **MAE = 1.925 minutes**
(run id `e0526ab34e4f4a8b9e756afaf0003eb0`), beating the best RandomForest
(`rf-n300-d12`, MAE 2.379) and the worst run (`tree-d2`, MAE 5.834).

## How to run

```
cd P06-mlflow
jupyter lab P06.ipynb          # run top to bottom
python -m mlflow ui --backend-store-uri "sqlite:///work/mlflow.db" --port 5000
```

## AI-assistance disclosure

Parts of this practical were prepared with the help of an AI coding assistant (opencode). The
notebook was executed from a clean kernel, the self-check reports 8 of 8 checks passed, and the
work can be explained and modified on request.
