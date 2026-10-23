import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tests.synthetic_data import make_synthetic_frame

from cereb_adhd_screen import (
    BranchConfig,
    BrainViewTransformer,
    FoldResidualizer,
    LEGACY_FEATURES,
    SCREEN_SVM_WEIGHT,
    _fit_adaptive_meta,
    _fit_branch,
    _fit_locked_xgb,
    _fit_legacy_svm,
    _fit_meta,
    _screen_probability,
    feature_groups,
    raw_for_norm,
)


class FeatureSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_synthetic_frame()

    def test_feature_group_counts_and_disjointness(self):
        groups = feature_groups(self.data.columns)
        self.assertEqual(len(groups["thickness_raw"]), 39)
        self.assertEqual(len(groups["thickness_norm"]), 39)
        self.assertEqual(len(groups["other_morphometry"]), 195)
        self.assertEqual(len(groups["brain_raw"]), 234)
        self.assertTrue(set(groups["thickness_raw"]).isdisjoint(groups["thickness_norm"]))
        self.assertTrue(set(groups["thickness_raw"]).isdisjoint(groups["other_morphometry"]))

    def test_every_norm_feature_maps_to_existing_raw_feature(self):
        groups = feature_groups(self.data.columns)
        mapped = [raw_for_norm(name) for name in groups["thickness_norm"]]
        self.assertEqual(len(set(mapped)), 39)
        self.assertTrue(set(mapped).issubset(self.data.columns))

    def test_norm_formula_matches_reference_icv_scaling(self):
        groups = feature_groups(self.data.columns)
        icv = pd.to_numeric(self.data["ICVcm3"], errors="coerce")
        for norm_name in groups["thickness_norm"]:
            raw_name = raw_for_norm(norm_name)
            raw = pd.to_numeric(self.data[raw_name], errors="coerce")
            observed = pd.to_numeric(self.data[norm_name], errors="coerce")
            expected = raw * np.power(1000.0 / icv, 1.0 / 3.0)
            ok = observed.notna() & expected.notna()
            self.assertLess(float(np.max(np.abs(observed[ok] - expected[ok]))), 5e-4)


class LeakageSafetyTests(unittest.TestCase):
    def make_frame(self):
        rng = np.random.RandomState(7)
        n = 80
        age = rng.uniform(7, 15, n)
        gender = rng.binomial(1, 0.6, n)
        icv = rng.normal(1450, 120, n)
        return pd.DataFrame(
            {
                "Age": age,
                "Gender": gender,
                "ICVcm3": icv,
                "VIleftcorticalthickness": 5 + 0.04 * age + rng.normal(0, 0.2, n),
                "VIIIBrightcorticalthickness": 4.8 - 0.1 * gender + rng.normal(0, 0.2, n),
                "Xleft": 0.5 + 0.0001 * icv + rng.normal(0, 0.04, n),
                "DX": rng.binomial(1, 0.4, n),
                "ADHDIndex": rng.normal(50, 10, n),
                "Sub": [f"S{i}" for i in range(n)],
            }
        )

    def test_residualizer_uses_training_statistics_only(self):
        frame = self.make_frame()
        train = frame.iloc[:60].copy()
        test = frame.iloc[60:].copy()
        r = FoldResidualizer(
            ["VIleftcorticalthickness", "VIIIBrightcorticalthickness"],
            include_icv=True,
        ).fit(train)
        first = r.transform(test)
        changed = test.copy()
        changed["DX"] = 1 - changed["DX"]
        changed["ADHDIndex"] = 1000
        second = r.transform(changed)
        np.testing.assert_allclose(first, second)

    def test_brain_transformer_never_emits_metadata_or_target(self):
        frame = self.make_frame()
        features = [
            "VIleftcorticalthickness",
            "VIIIBrightcorticalthickness",
            "Xleft",
        ]
        t = BrainViewTransformer(features, include_icv=True, max_features=3).fit(
            frame.iloc[:60], frame.DX.iloc[:60]
        )
        out = t.transform(frame.iloc[60:])
        self.assertEqual(out.shape[0], 20)
        self.assertTrue(set(t.selected_features_).issubset(features))
        self.assertTrue({"DX", "Age", "Gender", "ICVcm3", "ADHDIndex", "Sub"}.isdisjoint(t.selected_features_))


class ModelComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = make_synthetic_frame()
        cls.data = pd.concat([
            data[data.DX == 0].head(70),
            data[data.DX == 1].head(70),
        ]).sample(frac=1, random_state=11).reset_index(drop=True)

    def test_xgboost_branch_outputs_valid_probabilities(self):
        groups = feature_groups(self.data.columns)
        config = BranchConfig(
            view="thickness_raw",
            max_features=8,
            corr_threshold=0.95,
            params={
                "n_estimators": 20,
                "max_depth": 2,
                "learning_rate": 0.05,
                "min_child_weight": 2,
                "gamma": 0.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
                "scale_pos_weight": 1.0,
            },
        )
        train, test = self.data.iloc[:100], self.data.iloc[100:]
        model = _fit_branch(train, train.DX, groups, config, seed=3)
        probability = model.predict_proba(test)
        self.assertEqual(probability.shape, (40,))
        self.assertTrue(np.all(np.isfinite(probability)))
        self.assertTrue(np.all((probability >= 0) & (probability <= 1)))

    def test_optimized_svm_adds_direct_icv_without_changing_legacy_baseline(self):
        train = self.data.iloc[:100]
        legacy = _fit_legacy_svm(train, train.DX, seed=3)
        optimized = _fit_legacy_svm(train, train.DX, seed=3, direct_icv=True)
        self.assertEqual(legacy.scaler.n_features_in_, len(LEGACY_FEATURES) + 2)
        self.assertEqual(optimized.scaler.n_features_in_, len(LEGACY_FEATURES) + 3)

    def test_locked_xgb_and_fixed_screen_blend(self):
        train, test = self.data.iloc[:100], self.data.iloc[100:]
        model = _fit_locked_xgb(train, train.DX, seed=3)
        xgb_probability = model.predict_proba(test)
        svm_probability = np.linspace(0.1, 0.9, len(test))
        screen_probability = _screen_probability(svm_probability, xgb_probability)
        self.assertEqual(model.model.n_features_in_, len(LEGACY_FEATURES) + 3)
        self.assertTrue(np.all((xgb_probability >= 0) & (xgb_probability <= 1)))
        np.testing.assert_allclose(
            screen_probability,
            SCREEN_SVM_WEIGHT * svm_probability + (1 - SCREEN_SVM_WEIGHT) * xgb_probability,
        )

    def test_meta_model_uses_only_branch_probabilities(self):
        rng = np.random.RandomState(5)
        y = pd.Series(np.r_[np.zeros(60, dtype=int), np.ones(60, dtype=int)])
        branch_probability = np.column_stack([
            np.clip(0.25 + 0.45 * y + rng.normal(0, 0.12, len(y)), 0.01, 0.99),
            np.clip(0.35 + 0.25 * y + rng.normal(0, 0.16, len(y)), 0.01, 0.99),
            np.clip(0.40 + 0.15 * y + rng.normal(0, 0.18, len(y)), 0.01, 0.99),
        ])
        model = _fit_meta(branch_probability, y, seed=9)
        probability = model.predict_proba(branch_probability)[:, 1]
        self.assertEqual(model.n_features_in_, 3)
        self.assertGreater(float(np.corrcoef(probability, y)[0, 1]), 0.5)

    def test_adaptive_fusion_is_convex_and_can_fall_back_to_strong_branch(self):
        rng = np.random.RandomState(17)
        y = pd.Series(np.r_[np.zeros(80, dtype=int), np.ones(80, dtype=int)])
        strong = np.clip(0.30 + 0.40 * y + rng.normal(0, 0.16, len(y)), 0.01, 0.99)
        features = np.column_stack([
            strong,
            np.clip(0.35 + 0.30 * y + rng.normal(0, 0.20, len(y)), 0.01, 0.99),
            np.clip(0.40 + 0.15 * y + rng.normal(0, 0.22, len(y)), 0.01, 0.99),
            rng.uniform(0.2, 0.8, len(y)),
            rng.uniform(0.2, 0.8, len(y)),
        ])
        model = _fit_adaptive_meta(features, y, seed=9)
        probability = model.predict_proba(features)
        self.assertEqual(probability.shape, (len(y),))
        self.assertTrue(np.all(model.weights >= 0))
        self.assertAlmostEqual(float(model.weights.sum()), 1.0)
        self.assertGreaterEqual(
            roc_auc_score(y, probability), roc_auc_score(y, strong) - 1e-12
        )


if __name__ == "__main__":
    unittest.main()
