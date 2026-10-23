import unittest

import numpy as np
import pandas as pd

from tests.synthetic_data import make_synthetic_frame

from cereb_adhd_v2 import (
    ENGINEERED_FEATURE_COUNT,
    FORBIDDEN_MODEL_INPUTS,
    AnatomicalFeatureEngineer,
    _ecdf_transform,
    _fit_v2_model,
    candidate_feature_names,
    model_source_columns,
    simplex_weight_grid,
)


class AnatomyFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_synthetic_frame()
        cls.engineered = AnatomicalFeatureEngineer().transform(cls.data.head(8))

    def test_engineered_feature_count_and_exact_formulas(self):
        d = self.data.head(8)
        e = self.engineered
        engineered_names = AnatomicalFeatureEngineer.engineered_feature_names()
        self.assertEqual(len(engineered_names), ENGINEERED_FEATURE_COUNT)
        self.assertEqual(len(set(engineered_names)), ENGINEERED_FEATURE_COUNT)
        np.testing.assert_allclose(
            e["relative_volume__VI"], d["VItotalcm3"] / d["Cerebellumtotalcm3"]
        )
        np.testing.assert_allclose(
            e["relative_greymatter__VIIIB"],
            d["VIIIBgreymattercm3"] / d["Cerebellumgreymattercm3"],
        )
        np.testing.assert_allclose(
            e["local_greymatter_fraction__IX"],
            d["IXgreymattercm3"] / d["IXtotalcm3"],
        )
        np.testing.assert_allclose(
            e["relative_thickness_left__III_A"],
            d["IIIleftcorticalthickness_A"]
            / d["Cerebellumleftcorticalthickness"],
        )
        self.assertTrue(np.isfinite(e[list(engineered_names)].to_numpy()).all())

    def test_candidate_sources_exclude_target_symptoms_medication_and_id(self):
        sources = set(model_source_columns(self.data.columns))
        candidates = candidate_feature_names(self.data.columns)
        self.assertTrue(sources.isdisjoint(FORBIDDEN_MODEL_INPUTS))
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertTrue(set(AnatomicalFeatureEngineer.engineered_feature_names()).issubset(candidates))


class V2ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = make_synthetic_frame()
        cls.data = pd.concat(
            [data[data.DX == 0].head(70), data[data.DX == 1].head(70)]
        ).sample(frac=1, random_state=29).reset_index(drop=True)
        cls.features = (
            "VIleftcorticalthickness",
            "VIIIBrightcorticalthickness",
            "relative_volume__VI",
            "relative_greymatter__VIIIB",
            "local_greymatter_fraction__IX",
            "relative_thickness_left__III_A",
        )

    def test_three_model_families_emit_finite_probabilities(self):
        train, test = self.data.iloc[:100], self.data.iloc[100:]
        configs = (
            ("elastic_net", {"C": 0.1, "l1_ratio": 0.5}),
            ("rbf_svm", {"C": 1.0, "gamma": "scale"}),
            (
                "xgboost",
                {
                    "n_estimators": 20,
                    "max_depth": 1,
                    "learning_rate": 0.05,
                    "min_child_weight": 5,
                    "gamma": 0.0,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.5,
                    "reg_lambda": 10.0,
                    "scale_pos_weight": 1.0,
                },
            ),
        )
        for family, params in configs:
            model = _fit_v2_model(
                train, train["DX"].astype(int), self.features, family, params, seed=11
            )
            probability = model.predict_proba(test)
            self.assertEqual(probability.shape, (len(test),))
            self.assertTrue(np.isfinite(probability).all())
            self.assertTrue(np.all((probability >= 0) & (probability <= 1)))
            self.assertTrue(set(model.input_columns).isdisjoint(FORBIDDEN_MODEL_INPUTS))

    def test_ecdf_and_simplex_grid_are_bounded_and_deterministic(self):
        reference = np.array([0.1, 0.2, 0.5, 0.8])
        values = np.array([0.05, 0.2, 0.9])
        observed = _ecdf_transform(reference, values)
        np.testing.assert_allclose(observed, [0.0, 0.5, 1.0])
        weights = simplex_weight_grid(step=0.25, dimensions=3)
        self.assertEqual(len(weights), 15)
        self.assertEqual(len({tuple(w) for w in weights}), 15)
        for weight in weights:
            self.assertTrue(np.all(weight >= 0))
            self.assertAlmostEqual(float(weight.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
