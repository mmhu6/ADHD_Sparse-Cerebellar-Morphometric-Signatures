"""Bounded optimization experiment for Cereb-ADHDScreen.

This module is deliberately separate from ``ADHD_c.py`` and
``cereb_adhd_screen.py``.  It develops one compact XGBoost branch from a
predeclared 12-feature pool, freezes its settings, and then evaluates the
frozen XGBoost/SVM fusion with repeated 5x3 out-of-fold predictions.

Feature-set, parameter, and fusion-weight development use all 748 subjects on
three different predefined five-fold splits.  The final results are therefore
exploratory internal validation, not independent or external validation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from xgboost import XGBClassifier

from cereb_adhd_screen import (
    FoldResidualizer,
    LEGACY_FEATURES,
    LOCKED_XGB_PARAMS,
    _bootstrap_delta,
    _fit_legacy_svm,
    _fit_locked_xgb,
    _make_xgb,
    _metrics,
)


FEATURE_SELECTION_SEED = 142
PARAMETER_SELECTION_SEED = 242
FUSION_SELECTION_SEED = 342
VALIDATION_SEED = 42
NON_INFERIORITY_TOLERANCE = 0.002

PAPER_LOCKED_FEATURES = tuple(LEGACY_FEATURES)
INITIAL_EIGHT_FEATURES = (
    "VIleftcorticalthickness",
    "VIIIBrightcorticalthickness",
    "Xleft",
    "IXrightcorticalthickness",
    "Xright",
    "VIIIArightcorticalthickness",
    "VIIIBtotalcm3",
    "VIIIBleftcorticalthickness",
)
STABILITY_FEATURES = (
    "VImeancorticalthickness",
    "CrusImeancorticalthickness",
    "Vleftcorticalthickness",
    "Xtotal",
)
CANDIDATE_FEATURES = INITIAL_EIGHT_FEATURES + STABILITY_FEATURES

DIRECT_COVARIATES = ("Age", "Gender", "ICVcm3")
FORBIDDEN_MODEL_INPUTS = frozenset(
    {"DX", "ADHDIndex", "Inattentive", "HyperImpulsive", "MedStatus", "Sub"}
)
SVM_WEIGHT_GRID = (0.2, 0.3, 0.4, 0.5, 0.6)

# The first entry is the fixed anchor used during backward selection.  The
# remaining entries test a small range of stronger tree/regularization shapes.
XGB_PARAMETER_LIBRARY: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "locked_reference",
        dict(LOCKED_XGB_PARAMS),
    ),
    (
        "shallow_high_regularization",
        {
            "n_estimators": 260,
            "max_depth": 1,
            "learning_rate": 0.04,
            "min_child_weight": 8,
            "gamma": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.75,
            "reg_lambda": 25.0,
            "scale_pos_weight": 1.0,
        },
    ),
    (
        "depth2_sparse",
        {
            "n_estimators": 340,
            "max_depth": 2,
            "learning_rate": 0.02,
            "min_child_weight": 10,
            "gamma": 0.15,
            "subsample": 0.80,
            "colsample_bytree": 0.65,
            "reg_alpha": 1.0,
            "reg_lambda": 35.0,
            "scale_pos_weight": 1.0,
        },
    ),
    (
        "depth2_balanced",
        {
            "n_estimators": 220,
            "max_depth": 2,
            "learning_rate": 0.03,
            "min_child_weight": 6,
            "gamma": 0.10,
            "subsample": 0.90,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.30,
            "reg_lambda": 15.0,
            "scale_pos_weight": 1.0,
        },
    ),
)


@dataclass(frozen=True)
class ExperimentMode:
    name: str
    development_splits: int
    minimum_features: int
    parameter_trials: int
    fusion_weights: tuple[float, ...]
    outer_splits: int
    outer_repeats: int
    bootstrap_samples: int


SMOKE_MODE = ExperimentMode(
    "smoke", 3, 10, 2, (0.3, 0.4, 0.5), 3, 1, 300
)
FULL_MODE = ExperimentMode(
    "full", 5, 4, len(XGB_PARAMETER_LIBRARY), SVM_WEIGHT_GRID, 5, 3, 5000
)


def model_input_columns(features: Sequence[str]) -> tuple[str, ...]:
    """Declare all columns read by a candidate model, excluding the target."""

    columns = tuple(features) + DIRECT_COVARIATES
    if set(columns).intersection(FORBIDDEN_MODEL_INPUTS):
        raise ValueError("Candidate model input contains a forbidden column")
    return columns


def _numeric_with_fill(series: pd.Series, fill: float) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    return np.where(np.isfinite(values), values, fill)


@dataclass
class CandidateMatrixTransformer:
    features: tuple[str, ...]
    residualizer: FoldResidualizer
    covariate_medians: dict[str, float]

    @classmethod
    def fit(cls, X: pd.DataFrame, features: Sequence[str]):
        features = tuple(features)
        model_input_columns(features)
        residualizer = FoldResidualizer(features, include_icv=True).fit(X)
        medians: dict[str, float] = {}
        for name in DIRECT_COVARIATES:
            values = pd.to_numeric(X[name], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"Covariate {name} has no finite values")
            medians[name] = float(np.median(finite))
        return cls(features, residualizer, medians)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        brain = self.residualizer.transform(X).to_numpy(float)
        direct = np.column_stack(
            [
                _numeric_with_fill(X[name], self.covariate_medians[name])
                for name in DIRECT_COVARIATES
            ]
        )
        return np.column_stack([brain, direct])


@dataclass
class FittedCandidateXGB:
    transformer: CandidateMatrixTransformer
    model: XGBClassifier

    @property
    def input_columns(self) -> tuple[str, ...]:
        return model_input_columns(self.transformer.features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probability = self.model.predict_proba(self.transformer.transform(X))[:, 1]
        return np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)


def _fit_candidate_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    params: dict[str, Any],
    seed: int,
) -> FittedCandidateXGB:
    transformer = CandidateMatrixTransformer.fit(X, features)
    model = _make_xgb(dict(params), seed).fit(transformer.transform(X), y)
    return FittedCandidateXGB(transformer, model)


def _fit_candidate_svm(X: pd.DataFrame, y: pd.Series, seed: int):
    """Fit the existing locked-five optimized SVM comparison branch."""

    return _fit_legacy_svm(X, y, seed=seed, direct_icv=True)


def fixed_probability_fusion(
    svm_probability: np.ndarray,
    xgb_probability: np.ndarray,
    svm_weight: float,
) -> np.ndarray:
    if not 0.0 <= svm_weight <= 1.0:
        raise ValueError("svm_weight must be within [0, 1]")
    svm_probability = np.asarray(svm_probability, dtype=float)
    xgb_probability = np.asarray(xgb_probability, dtype=float)
    if svm_probability.shape != xgb_probability.shape:
        raise ValueError("SVM and XGBoost probabilities must have equal shape")
    if not np.all(np.isfinite(svm_probability)) or not np.all(
        np.isfinite(xgb_probability)
    ):
        raise ValueError("Fusion probabilities must be finite")
    probability = svm_weight * svm_probability + (1.0 - svm_weight) * xgb_probability
    return np.clip(probability, 0.0, 1.0)


def _fixed_splits(y: pd.Series, n_splits: int, seed: int):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def _xgb_oof(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    params: dict[str, Any],
    splits,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    probability = np.full(len(y), np.nan, dtype=float)
    fold_auc: list[float] = []
    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = _fit_candidate_xgb(
            X.iloc[train_idx],
            y.iloc[train_idx],
            features,
            params,
            seed + fold_id,
        )
        fold_probability = model.predict_proba(X.iloc[valid_idx])
        probability[valid_idx] = fold_probability
        fold_auc.append(float(roc_auc_score(y.iloc[valid_idx], fold_probability)))
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("XGBoost OOF prediction contains non-finite values")
    return probability, fold_auc


def _svm_oof(
    X: pd.DataFrame, y: pd.Series, splits, seed: int
) -> tuple[np.ndarray, list[float]]:
    probability = np.full(len(y), np.nan, dtype=float)
    fold_auc: list[float] = []
    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = _fit_candidate_svm(
            X.iloc[train_idx], y.iloc[train_idx], seed + fold_id
        )
        fold_probability = model.predict_proba(X.iloc[valid_idx])
        probability[valid_idx] = fold_probability
        fold_auc.append(float(roc_auc_score(y.iloc[valid_idx], fold_probability)))
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("SVM OOF prediction contains non-finite values")
    return probability, fold_auc


def _score_oof(y: pd.Series, probability: np.ndarray, fold_auc: Sequence[float]):
    return {
        "pooled_auc": float(roc_auc_score(y, probability)),
        "mean_fold_auc": float(np.mean(fold_auc)),
        "sd_fold_auc": float(np.std(fold_auc, ddof=1)) if len(fold_auc) > 1 else 0.0,
    }


def greedy_backward_selection(
    X: pd.DataFrame,
    y: pd.Series,
    splits,
    minimum_features: int,
    tolerance: float = NON_INFERIORITY_TOLERANCE,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Greedily remove features while staying within tolerance of best AUC."""

    current = tuple(CANDIDATE_FEATURES)
    anchor_params = XGB_PARAMETER_LIBRARY[0][1]
    initial_probability, initial_folds = _xgb_oof(
        X, y, current, anchor_params, splits, FEATURE_SELECTION_SEED
    )
    initial_score = _score_oof(y, initial_probability, initial_folds)
    best_seen_auc = initial_score["pooled_auc"]
    rows: list[dict[str, Any]] = [
        {
            "step": 0,
            "candidate_size": len(current),
            "removed_feature": "",
            "features": ";".join(current),
            **initial_score,
            "chosen_for_step": True,
            "accepted": True,
            "best_seen_auc": best_seen_auc,
        }
    ]

    step = 0
    while len(current) > minimum_features:
        step += 1
        candidates: list[tuple[str, tuple[str, ...], dict[str, float]]] = []
        for feature in current:
            reduced = tuple(name for name in current if name != feature)
            probability, fold_auc = _xgb_oof(
                X,
                y,
                reduced,
                anchor_params,
                splits,
                FEATURE_SELECTION_SEED,
            )
            candidates.append((feature, reduced, _score_oof(y, probability, fold_auc)))
        chosen_index = max(
            range(len(candidates)),
            key=lambda index: (candidates[index][2]["pooled_auc"], -index),
        )
        chosen_auc = candidates[chosen_index][2]["pooled_auc"]
        accepted = chosen_auc >= best_seen_auc - tolerance
        updated_best = max(best_seen_auc, chosen_auc) if accepted else best_seen_auc
        for index, (removed, reduced, score) in enumerate(candidates):
            rows.append(
                {
                    "step": step,
                    "candidate_size": len(reduced),
                    "removed_feature": removed,
                    "features": ";".join(reduced),
                    **score,
                    "chosen_for_step": index == chosen_index,
                    "accepted": bool(accepted and index == chosen_index),
                    "best_seen_auc": updated_best,
                }
            )
        if not accepted:
            break
        current = candidates[chosen_index][1]
        best_seen_auc = updated_best
        print(
            f"Feature step {step}: removed {candidates[chosen_index][0]}; "
            f"n={len(current)}; pooled AUC={chosen_auc:.4f}",
            flush=True,
        )
    return current, pd.DataFrame(rows)


def compare_feature_sets(
    X: pd.DataFrame,
    y: pd.Series,
    selected_features: Sequence[str],
    splits,
) -> pd.DataFrame:
    sets = (
        ("paper_locked_5", PAPER_LOCKED_FEATURES),
        ("paper_initial_8", INITIAL_EIGHT_FEATURES),
        ("stability_selected", tuple(selected_features)),
    )
    rows = []
    for name, features in sets:
        probability, fold_auc = _xgb_oof(
            X,
            y,
            features,
            XGB_PARAMETER_LIBRARY[0][1],
            splits,
            FEATURE_SELECTION_SEED,
        )
        rows.append(
            {
                "feature_set": name,
                "n_features": len(features),
                "features": ";".join(features),
                **_score_oof(y, probability, fold_auc),
            }
        )
    return pd.DataFrame(rows).sort_values("pooled_auc", ascending=False)


def select_xgb_parameters(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    splits,
    n_trials: int,
) -> tuple[str, dict[str, Any], pd.DataFrame]:
    rows = []
    candidates = XGB_PARAMETER_LIBRARY[:n_trials]
    for name, params in candidates:
        probability, fold_auc = _xgb_oof(
            X, y, features, params, splits, PARAMETER_SELECTION_SEED
        )
        rows.append({"parameter_set": name, **_score_oof(y, probability, fold_auc)})
    trials = pd.DataFrame(rows)
    best_index = int(trials["pooled_auc"].to_numpy().argmax())
    best_name = str(trials.iloc[best_index]["parameter_set"])
    params_by_name = {name: params for name, params in candidates}
    return best_name, dict(params_by_name[best_name]), trials.sort_values(
        "pooled_auc", ascending=False
    )


def select_fusion_weight(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    params: dict[str, Any],
    splits,
    weights: Sequence[float],
) -> tuple[float, pd.DataFrame]:
    xgb_probability, xgb_folds = _xgb_oof(
        X, y, features, params, splits, FUSION_SELECTION_SEED
    )
    svm_probability, svm_folds = _svm_oof(X, y, splits, FUSION_SELECTION_SEED)
    rows = [
        {"candidate": "xgb_only", "svm_weight": 0.0, **_score_oof(y, xgb_probability, xgb_folds)},
        {"candidate": "svm_only", "svm_weight": 1.0, **_score_oof(y, svm_probability, svm_folds)},
    ]
    fold_indices = [valid_idx for _, valid_idx in splits]
    for weight in weights:
        probability = fixed_probability_fusion(
            svm_probability, xgb_probability, weight
        )
        fold_auc = [
            float(roc_auc_score(y.iloc[index], probability[index]))
            for index in fold_indices
        ]
        rows.append(
            {
                "candidate": "fixed_fusion",
                "svm_weight": float(weight),
                **_score_oof(y, probability, fold_auc),
            }
        )
    trials = pd.DataFrame(rows)
    fusion_trials = trials[trials["candidate"] == "fixed_fusion"].reset_index(drop=True)
    best_index = int(fusion_trials["pooled_auc"].to_numpy().argmax())
    best_weight = float(fusion_trials.iloc[best_index]["svm_weight"])
    return best_weight, trials.sort_values("pooled_auc", ascending=False)


def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    required = (
        set(CANDIDATE_FEATURES)
        | set(DIRECT_COVARIATES)
        | {"DX"}
    )
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    clean = data.copy()
    clean["DX"] = pd.to_numeric(clean["DX"], errors="raise").astype(int)
    clean = clean.dropna(subset=["DX", *DIRECT_COVARIATES]).reset_index(drop=True)
    if set(clean["DX"].unique()) != {0, 1}:
        raise ValueError("DX must contain binary values 0 and 1")
    return clean


def frozen_repeated_validation(
    data: pd.DataFrame,
    selected_features: Sequence[str],
    selected_params: dict[str, Any],
    svm_weight: float,
    mode: ExperimentMode,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = data["DX"].astype(int).reset_index(drop=True)
    outer = RepeatedStratifiedKFold(
        n_splits=mode.outer_splits,
        n_repeats=mode.outer_repeats,
        random_state=VALIDATION_SEED,
    )
    model_names = (
        "StableSelectedXGB",
        "LockedFiveSVM",
        "StableFixedFusion",
        "ExistingLockedXGB",
        "ExistingCerebADHDScreen",
    )
    score_sum = {name: np.zeros(len(y), dtype=float) for name in model_names}
    score_count = np.zeros(len(y), dtype=int)
    fold_rows: list[dict[str, Any]] = []

    for fold_id, (train_idx, valid_idx) in enumerate(outer.split(data, y), start=1):
        X_train, X_valid = data.iloc[train_idx], data.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        optimized_xgb = _fit_candidate_xgb(
            X_train,
            y_train,
            selected_features,
            selected_params,
            VALIDATION_SEED + fold_id * 975,
        )
        # Match the already reported Cereb-ADHDScreen seeds exactly so the two
        # existing-model rows are strict same-fold reproducibility controls.
        svm = _fit_candidate_svm(
            X_train, y_train, VALIDATION_SEED + fold_id * 875
        )
        existing_xgb = _fit_locked_xgb(
            X_train, y_train, VALIDATION_SEED + fold_id * 900
        )
        xgb_probability = optimized_xgb.predict_proba(X_valid)
        svm_probability = svm.predict_proba(X_valid)
        fusion_probability = fixed_probability_fusion(
            svm_probability, xgb_probability, svm_weight
        )
        existing_xgb_probability = existing_xgb.predict_proba(X_valid)
        existing_screen_probability = fixed_probability_fusion(
            svm_probability, existing_xgb_probability, 0.4
        )
        probabilities = {
            "StableSelectedXGB": xgb_probability,
            "LockedFiveSVM": svm_probability,
            "StableFixedFusion": fusion_probability,
            "ExistingLockedXGB": existing_xgb_probability,
            "ExistingCerebADHDScreen": existing_screen_probability,
        }
        for name, probability in probabilities.items():
            score_sum[name][valid_idx] += probability
            fold_rows.append(
                {"outer_fold": fold_id, "model": name, **_metrics(y_valid, probability)}
            )
        score_count[valid_idx] += 1
        print(
            f"Validation fold {fold_id:02d}: stable fusion AUC="
            f"{roc_auc_score(y_valid, fusion_probability):.3f}",
            flush=True,
        )

    if not np.all(score_count == mode.outer_repeats):
        raise RuntimeError("Each participant must have one prediction per repeat")
    aggregated = {name: score_sum[name] / score_count for name in model_names}
    fold_metrics = pd.DataFrame(fold_rows)
    fold_summary = fold_metrics.groupby("model")["AUC"].agg(["mean", "std"]).reset_index()
    fold_summary = fold_summary.rename(
        columns={"mean": "mean_fold_AUC", "std": "sd_fold_AUC"}
    )
    comparison = pd.DataFrame(
        [{"model": name, **_metrics(y, aggregated[name])} for name in model_names]
    ).merge(fold_summary, on="model", how="left")
    comparison = comparison.sort_values("AUC", ascending=False).reset_index(drop=True)

    predictions = pd.DataFrame(
        {
            "row_id": np.arange(len(y)),
            "Sub": data["Sub"] if "Sub" in data else "",
            "DX": y,
            "Age": pd.to_numeric(data["Age"], errors="coerce"),
            "Gender": pd.to_numeric(data["Gender"], errors="coerce"),
            "prediction_count": score_count,
            **aggregated,
        }
    )
    bootstrap = pd.DataFrame(
        [
            {
                "comparison": "StableFixedFusion - ExistingCerebADHDScreen",
                **_bootstrap_delta(
                    y.to_numpy(),
                    aggregated["StableFixedFusion"],
                    aggregated["ExistingCerebADHDScreen"],
                    mode.bootstrap_samples,
                    VALIDATION_SEED,
                ),
            },
            {
                "comparison": "StableFixedFusion - StableSelectedXGB",
                **_bootstrap_delta(
                    y.to_numpy(),
                    aggregated["StableFixedFusion"],
                    aggregated["StableSelectedXGB"],
                    mode.bootstrap_samples,
                    VALIDATION_SEED + 1,
                ),
            },
        ]
    )
    return fold_metrics, comparison, predictions, bootstrap


def run_experiment(
    data: pd.DataFrame, mode: ExperimentMode, output_dir: Path
) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _prepare_data(data)
    y = data["DX"].astype(int)

    feature_splits = _fixed_splits(
        y, mode.development_splits, FEATURE_SELECTION_SEED
    )
    selected_features, feature_path = greedy_backward_selection(
        data,
        y,
        feature_splits,
        minimum_features=mode.minimum_features,
    )
    feature_benchmarks = compare_feature_sets(
        data, y, selected_features, feature_splits
    )

    parameter_splits = _fixed_splits(
        y, mode.development_splits, PARAMETER_SELECTION_SEED
    )
    parameter_name, selected_params, parameter_trials = select_xgb_parameters(
        data,
        y,
        selected_features,
        parameter_splits,
        mode.parameter_trials,
    )

    fusion_splits = _fixed_splits(y, mode.development_splits, FUSION_SELECTION_SEED)
    svm_weight, fusion_trials = select_fusion_weight(
        data,
        y,
        selected_features,
        selected_params,
        fusion_splits,
        mode.fusion_weights,
    )
    print(
        f"Frozen candidate: n_features={len(selected_features)}, "
        f"params={parameter_name}, svm_weight={svm_weight:.1f}",
        flush=True,
    )

    fold_metrics, comparison, predictions, bootstrap = frozen_repeated_validation(
        data,
        selected_features,
        selected_params,
        svm_weight,
        mode,
    )

    feature_path.to_csv(output_dir / "development_feature_path.csv", index=False)
    feature_benchmarks.to_csv(
        output_dir / "development_feature_benchmarks.csv", index=False
    )
    parameter_trials.to_csv(
        output_dir / "development_parameter_trials.csv", index=False
    )
    fusion_trials.to_csv(output_dir / "development_fusion_trials.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_comparisons.csv", index=False)

    config = {
        "mode": mode.__dict__,
        "n": len(data),
        "adhd": int((y == 1).sum()),
        "control": int((y == 0).sum()),
        "development_scope": "global_full_sample_exploratory_internal_development",
        "feature_selection_seed": FEATURE_SELECTION_SEED,
        "parameter_selection_seed": PARAMETER_SELECTION_SEED,
        "fusion_selection_seed": FUSION_SELECTION_SEED,
        "validation_seed": VALIDATION_SEED,
        "candidate_features": list(CANDIDATE_FEATURES),
        "selected_features": list(selected_features),
        "selected_parameter_set": parameter_name,
        "selected_xgb_params": selected_params,
        "selected_svm_weight": svm_weight,
        "selected_xgb_weight": 1.0 - svm_weight,
        "forbidden_model_inputs": sorted(FORBIDDEN_MODEL_INPUTS),
        "elapsed_seconds": time.time() - started,
        "stretch_target_auc": 0.75,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    stable_auc = float(
        comparison.loc[comparison.model == "StableFixedFusion", "AUC"].iloc[0]
    )
    lines = [
        "Cereb-ADHDScreen bounded optimization summary",
        "",
        "Frozen development configuration",
        f"Selected features ({len(selected_features)}): {'; '.join(selected_features)}",
        f"Selected XGBoost parameter set: {parameter_name}",
        f"Fusion: SVM={svm_weight:.1f}, XGBoost={1.0 - svm_weight:.1f}",
        "",
        comparison.to_string(index=False),
        "",
        bootstrap.to_string(index=False),
        "",
        f"Stretch target AUC >= 0.750: {'MET' if stable_auc >= 0.750 else 'NOT MET'}",
        "Feature/parameter/weight development used the full cohort.",
        "These results are exploratory internal validation, not independent validation.",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return {
        "feature_path": feature_path,
        "feature_benchmarks": feature_benchmarks,
        "parameter_trials": parameter_trials,
        "fusion_trials": fusion_trials,
        "comparison": comparison,
        "predictions": predictions,
        "bootstrap": bootstrap,
        "config": config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cerebellum.csv"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = SMOKE_MODE if args.mode == "smoke" else FULL_MODE
    output = args.output or Path(f"cereb_adhd_optimization_outputs_{mode.name}")
    data = pd.read_csv(args.data)
    print(
        f"ADHD bounded optimization | mode={mode.name} | N={len(data)} | output={output}",
        flush=True,
    )
    result = run_experiment(data, mode, output)
    print(result["comparison"].to_string(index=False), flush=True)
    print(f"Finished. Outputs: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
