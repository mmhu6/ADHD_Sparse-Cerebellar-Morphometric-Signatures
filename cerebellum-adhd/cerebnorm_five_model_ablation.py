"""Five-classifier ablation of CerebNorm feature representations.

The experiment keeps the five classifiers used in the paper and changes only
the feature representation.  Every preprocessing operation, including the
control-defined normative reference, is fitted inside the current training
partition.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from cereb_adhd_screen import FoldResidualizer
from cerebnorm_adhd_screen import (
    ControlNormativeTransformer,
    DIRECT_COVARIATES,
    FORBIDDEN_MODEL_INPUTS,
    NORM_FEATURES,
)


SEED = 42
FEATURE_VIEWS = ("raw_only", "norm_only", "raw_plus_norm")
MODEL_NAMES = ("SVM", "RandomForest", "ElasticNet", "XGBoost", "CatBoost")


@dataclass(frozen=True)
class ExperimentMode:
    name: str
    outer_splits: int
    outer_repeats: int
    inner_splits: int
    bootstrap_samples: int
    compact_grid: bool


SMOKE_MODE = ExperimentMode("smoke", 3, 1, 2, 200, True)
FULL_MODE = ExperimentMode("full", 5, 3, 3, 5000, False)


def _numeric(series: pd.Series, fill: float | None = None) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    if fill is not None:
        values = np.where(np.isfinite(values), values, fill)
    return values


class FiveModelFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create one of three fold-fitted CerebNorm feature representations."""

    def __init__(self, view: str = "raw_only"):
        self.view = view

    def fit(self, X: pd.DataFrame, y: Sequence[int]):
        X = pd.DataFrame(X)
        if self.view not in FEATURE_VIEWS:
            raise ValueError(f"Unknown feature view: {self.view}")
        missing = (set(NORM_FEATURES) | set(DIRECT_COVARIATES)).difference(X.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")

        self.residualizer_ = None
        if self.view in {"raw_only", "raw_plus_norm"}:
            self.residualizer_ = FoldResidualizer(
                NORM_FEATURES, include_icv=True
            ).fit(X)

        self.normative_ = None
        if self.view in {"norm_only", "raw_plus_norm"}:
            self.normative_ = ControlNormativeTransformer().fit(X, y)

        self.covariate_medians_ = {}
        for name in DIRECT_COVARIATES:
            values = _numeric(X[name])
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"Covariate {name} has no finite values")
            self.covariate_medians_[name] = float(np.median(finite))

        output_names: list[str] = []
        if self.residualizer_ is not None:
            output_names.extend(f"raw_resid__{name}" for name in NORM_FEATURES)
        if self.normative_ is not None:
            output_names.extend(self.normative_.z_feature_names)
            output_names.extend(("norm_mean_abs_z", "norm_extreme_count"))
        output_names.extend(f"direct__{name}" for name in DIRECT_COVARIATES)
        self.output_names_ = tuple(output_names)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        parts: list[np.ndarray] = []
        if self.residualizer_ is not None:
            parts.append(self.residualizer_.transform(X).to_numpy(float))
        if self.normative_ is not None:
            parts.append(self.normative_.transform(X).to_numpy(float))
        parts.append(
            np.column_stack(
                [
                    _numeric(X[name], self.covariate_medians_[name])
                    for name in DIRECT_COVARIATES
                ]
            )
        )
        matrix = np.column_stack(parts)
        if not np.isfinite(matrix).all():
            raise ValueError("Transformed matrix contains non-finite values")
        return pd.DataFrame(matrix, columns=self.output_names_, index=X.index)


class CatBoostCompat(ClassifierMixin, BaseEstimator):
    """Small sklearn-compatible wrapper matching the paper's CatBoost model."""

    def __init__(
        self,
        iterations: int = 200,
        depth: int = 5,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 5.0,
        balanced: bool = True,
        random_state: int = SEED,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.balanced = balanced
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            verbose=0,
            random_seed=self.random_state,
            auto_class_weights="Balanced" if self.balanced else None,
            allow_writing_files=False,
            thread_count=1,
        )
        self.model_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).astype(int).ravel()

    def predict_proba(self, X):
        return np.asarray(self.model_.predict_proba(X))


def make_model(name: str, seed: int):
    if name == "SVM":
        return SVC(
            kernel="rbf", class_weight="balanced", probability=True,
            random_state=seed,
        )
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=seed,
            n_jobs=1,
        )
    if name == "ElasticNet":
        return LogisticRegression(
            penalty="elasticnet", solver="saga", class_weight="balanced",
            max_iter=5000, random_state=seed,
        )
    if name == "XGBoost":
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", n_jobs=1,
            random_state=seed, verbosity=0,
        )
    if name == "CatBoost":
        return CatBoostCompat(balanced=True, random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def parameter_grid(name: str, compact: bool = False) -> dict[str, list[Any]]:
    if compact:
        grids = {
            "SVM": {"model__C": [1.0], "model__gamma": ["scale"]},
            "RandomForest": {
                "model__max_depth": [5], "model__min_samples_leaf": [5],
                "model__max_features": ["sqrt"],
            },
            "ElasticNet": {"model__C": [0.1], "model__l1_ratio": [0.5]},
            "XGBoost": {
                "model__n_estimators": [50], "model__max_depth": [2],
                "model__learning_rate": [0.05], "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
            },
            "CatBoost": {
                "model__iterations": [50], "model__depth": [3],
                "model__learning_rate": [0.05], "model__l2_leaf_reg": [5],
            },
        }
        return grids[name]

    grids = {
        "SVM": {"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01]},
        "RandomForest": {
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [2, 5],
            "model__max_features": ["sqrt"],
        },
        "ElasticNet": {
            "model__C": [0.03, 0.1, 1, 10],
            "model__l1_ratio": [0.25, 0.5, 0.75],
        },
        "XGBoost": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [2, 3],
            "model__learning_rate": [0.03, 0.1],
            "model__subsample": [0.8],
            "model__colsample_bytree": [0.8],
        },
        "CatBoost": {
            "model__iterations": [100, 200],
            "model__depth": [3, 5],
            "model__learning_rate": [0.03, 0.1],
            "model__l2_leaf_reg": [5],
        },
    }
    return grids[name]


def model_pipeline(view: str, name: str, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("features", FiveModelFeatureTransformer(view)),
            ("scale", StandardScaler()),
            ("model", make_model(name, seed)),
        ]
    )


def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy().reset_index(drop=True)
    required = set(NORM_FEATURES) | set(DIRECT_COVARIATES) | {"DX"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data["DX"] = pd.to_numeric(data["DX"], errors="raise").astype(int)
    if not set(data["DX"].unique()).issubset({0, 1}):
        raise ValueError("DX must be binary 0/1")
    return data


def _metrics(y: Sequence[int], probability: np.ndarray) -> dict[str, float]:
    y_array = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_array, predicted, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(y_array, probability)),
        "AP": float(average_precision_score(y_array, probability)),
        "BACC": float(balanced_accuracy_score(y_array, predicted)),
        "Sensitivity": float(recall_score(y_array, predicted, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "F1": float(f1_score(y_array, predicted, zero_division=0)),
    }


def _bootstrap_delta(
    y: Sequence[int],
    candidate: np.ndarray,
    reference: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    y_array = np.asarray(y, dtype=int)
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    observed = float(
        roc_auc_score(y_array, candidate) - roc_auc_score(y_array, reference)
    )
    rng = np.random.RandomState(seed)
    class_indices = [np.flatnonzero(y_array == value) for value in (0, 1)]
    deltas = np.empty(n_boot, dtype=float)
    for boot_id in range(n_boot):
        sampled = np.concatenate(
            [rng.choice(indices, len(indices), replace=True) for indices in class_indices]
        )
        deltas[boot_id] = (
            roc_auc_score(y_array[sampled], candidate[sampled])
            - roc_auc_score(y_array[sampled], reference[sampled])
        )
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    p_lower = (np.sum(deltas <= 0) + 1) / (n_boot + 1)
    p_upper = (np.sum(deltas >= 0) + 1) / (n_boot + 1)
    return {
        "delta_auc": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(min(1.0, 2.0 * min(p_lower, p_upper))),
    }


def _wide_auc_table(comparison: pd.DataFrame) -> pd.DataFrame:
    wide = comparison.pivot(index="model", columns="feature_view", values="AUC")
    wide = wide.reindex(index=MODEL_NAMES, columns=FEATURE_VIEWS).reset_index()
    wide["delta_norm_minus_raw"] = wide["norm_only"] - wide["raw_only"]
    wide["delta_combined_minus_raw"] = wide["raw_plus_norm"] - wide["raw_only"]
    return wide


def run_experiment(
    data: pd.DataFrame,
    mode: ExperimentMode,
    output_dir: Path,
    n_jobs: int = -1,
) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _prepare_data(data)
    y = data["DX"].astype(int).reset_index(drop=True)

    outer = RepeatedStratifiedKFold(
        n_splits=mode.outer_splits,
        n_repeats=mode.outer_repeats,
        random_state=SEED,
    )
    outer_splits = list(outer.split(data, y))
    keys = [(view, model) for view in FEATURE_VIEWS for model in MODEL_NAMES]
    score_sum = {key: np.zeros(len(y), dtype=float) for key in keys}
    score_count = np.zeros(len(y), dtype=int)
    fold_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []

    for fold_id, (train_idx, valid_idx) in enumerate(outer_splits, start=1):
        X_train, X_valid = data.iloc[train_idx], data.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        inner = StratifiedKFold(
            n_splits=mode.inner_splits,
            shuffle=True,
            random_state=SEED + fold_id,
        )
        for view_id, view in enumerate(FEATURE_VIEWS):
            for model_id, model_name in enumerate(MODEL_NAMES):
                model_seed = SEED + fold_id * 100 + view_id * 10 + model_id
                search = GridSearchCV(
                    model_pipeline(view, model_name, model_seed),
                    parameter_grid(model_name, mode.compact_grid),
                    scoring="roc_auc",
                    cv=inner,
                    n_jobs=n_jobs,
                    refit=True,
                    error_score="raise",
                )
                search.fit(X_train, y_train)
                probability = np.asarray(
                    search.best_estimator_.predict_proba(X_valid)
                )[:, 1]
                key = (view, model_name)
                score_sum[key][valid_idx] += probability
                fold_rows.append(
                    {
                        "outer_fold": fold_id,
                        "feature_view": view,
                        "model": model_name,
                        **_metrics(y_valid, probability),
                    }
                )
                tuning_rows.append(
                    {
                        "outer_fold": fold_id,
                        "feature_view": view,
                        "model": model_name,
                        "inner_best_auc": float(search.best_score_),
                        "n_input_features": len(
                            search.best_estimator_.named_steps["features"].output_names_
                        ),
                        "best_parameters": json.dumps(
                            search.best_params_, sort_keys=True
                        ),
                    }
                )
        score_count[valid_idx] += 1
        print(
            f"Outer fold {fold_id:02d}/{len(outer_splits)} completed",
            flush=True,
        )

    if not np.all(score_count == mode.outer_repeats):
        raise RuntimeError("Unexpected repeated-OOF prediction counts")
    aggregated = {key: score_sum[key] / score_count for key in keys}
    fold_metrics = pd.DataFrame(fold_rows)
    fold_summary = (
        fold_metrics.groupby(["feature_view", "model"])["AUC"]
        .agg(mean_fold_AUC="mean", sd_fold_AUC="std")
        .reset_index()
    )
    comparison = pd.DataFrame(
        [
            {
                "feature_view": view,
                "model": model,
                **_metrics(y, aggregated[(view, model)]),
            }
            for view, model in keys
        ]
    ).merge(fold_summary, on=["feature_view", "model"], how="left")
    view_order = {view: index for index, view in enumerate(FEATURE_VIEWS)}
    model_order = {name: index for index, name in enumerate(MODEL_NAMES)}
    comparison["_view_order"] = comparison["feature_view"].map(view_order)
    comparison["_model_order"] = comparison["model"].map(model_order)
    comparison = (
        comparison.sort_values(["_model_order", "_view_order"])
        .drop(columns=["_view_order", "_model_order"])
        .reset_index(drop=True)
    )

    bootstrap_rows = []
    for model_id, model_name in enumerate(MODEL_NAMES):
        raw = aggregated[("raw_only", model_name)]
        for comparison_id, candidate_view in enumerate(
            ("norm_only", "raw_plus_norm")
        ):
            bootstrap_rows.append(
                {
                    "model": model_name,
                    "comparison": f"{candidate_view} - raw_only",
                    **_bootstrap_delta(
                        y,
                        aggregated[(candidate_view, model_name)],
                        raw,
                        mode.bootstrap_samples,
                        SEED + 1000 + model_id * 10 + comparison_id,
                    ),
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    wide_auc = _wide_auc_table(comparison)

    predictions = pd.DataFrame(
        {
            "row_id": np.arange(len(data)),
            "Sub": data["Sub"] if "Sub" in data else np.arange(len(data)),
            "DX": y,
            "Age": data["Age"],
            "Gender": data["Gender"],
            "prediction_count": score_count,
        }
    )
    for view, model_name in keys:
        predictions[f"{model_name}__{view}"] = aggregated[(view, model_name)]

    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(
        output_dir / "selected_hyperparameters.csv", index=False
    )
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    wide_auc.to_csv(output_dir / "feature_view_auc.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_comparisons.csv", index=False)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)

    input_counts = {"raw_only": 8, "norm_only": 10, "raw_plus_norm": 15}
    config = {
        "mode": asdict(mode),
        "seed": SEED,
        "n_jobs": n_jobs,
        "n": len(data),
        "adhd": int((y == 1).sum()),
        "control": int((y == 0).sum()),
        "models": list(MODEL_NAMES),
        "feature_views": list(FEATURE_VIEWS),
        "input_feature_counts": input_counts,
        "locked_brain_features": list(NORM_FEATURES),
        "direct_covariates_in_every_view": list(DIRECT_COVARIATES),
        "normative_reference": "current_training_partition_controls_only",
        "derived_norm_features": ["norm_mean_abs_z", "norm_extreme_count"],
        "forbidden_model_inputs": sorted(FORBIDDEN_MODEL_INPUTS),
        "feature_selection_note": (
            "The five locked brain features were selected previously on the full "
            "cohort; this is an exploratory post-selection ablation."
        ),
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "CerebNorm five-classifier feature-ablation summary",
        "",
        wide_auc.to_string(index=False),
        "",
        bootstrap.to_string(index=False),
        "",
        "All participants received predictions; no grey zone or rejection was used.",
        "Normative references were fitted only from controls in each training partition.",
        "This is exploratory post-selection internal validation, not external validation.",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return {
        "comparison": comparison,
        "wide_auc": wide_auc,
        "bootstrap": bootstrap,
        "predictions": predictions,
        "fold_metrics": fold_metrics,
        "config": config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cerebellum.csv"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = SMOKE_MODE if args.mode == "smoke" else FULL_MODE
    output = args.output or Path(f"cerebnorm_five_model_ablation_outputs_{mode.name}")
    data = pd.read_csv(args.data)
    print(
        f"CerebNorm five-model ablation | mode={mode.name} | N={len(data)} | "
        f"output={output}",
        flush=True,
    )
    result = run_experiment(data, mode, output, n_jobs=args.n_jobs)
    print(result["wide_auc"].to_string(index=False), flush=True)
    print(f"Finished. Outputs: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
