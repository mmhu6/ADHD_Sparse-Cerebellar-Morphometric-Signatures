import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests.synthetic_data import make_synthetic_frame

from cerebnorm_five_model_ablation import (
    ExperimentMode,
    FEATURE_VIEWS,
    MODEL_NAMES,
    FiveModelFeatureTransformer,
    run_experiment,
)


class FiveModelFeatureTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = make_synthetic_frame()
        cls.data = pd.concat(
            [data[data.DX == 0].head(80), data[data.DX == 1].head(80)]
        ).sample(frac=1, random_state=73).reset_index(drop=True)

    def test_three_views_have_expected_dimensions_and_ignore_validation_labels(self):
        train, test = self.data.iloc[:120], self.data.iloc[120:].copy()
        expected_dimensions = {
            "raw_only": 8,
            "norm_only": 10,
            "raw_plus_norm": 15,
        }
        for view in FEATURE_VIEWS:
            transformer = FiveModelFeatureTransformer(view).fit(
                train, train["DX"].astype(int)
            )
            observed = transformer.transform(test)
            self.assertEqual(observed.shape, (len(test), expected_dimensions[view]))
            self.assertTrue(np.isfinite(observed.to_numpy()).all())

            changed = test.copy()
            changed["DX"] = 1 - changed["DX"]
            changed["ADHDIndex"] = 9999
            changed["Sub"] = "changed"
            pd.testing.assert_frame_equal(observed, transformer.transform(changed))


class FiveModelAblationExperimentTests(unittest.TestCase):
    def test_compact_experiment_emits_complete_paired_oof_outputs(self):
        data = make_synthetic_frame()
        data = pd.concat(
            [data[data.DX == 0].head(90), data[data.DX == 1].head(90)]
        ).sample(frac=1, random_state=91).reset_index(drop=True)
        mode = ExperimentMode(
            name="test",
            outer_splits=2,
            outer_repeats=1,
            inner_splits=2,
            bootstrap_samples=20,
            compact_grid=True,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_experiment(data, mode, Path(temp_dir), n_jobs=1)
            self.assertEqual(len(result["comparison"]), 15)
            self.assertEqual(
                set(result["comparison"]["model"]), set(MODEL_NAMES)
            )
            self.assertEqual(
                set(result["comparison"]["feature_view"]), set(FEATURE_VIEWS)
            )
            self.assertEqual(len(result["predictions"]), len(data))
            self.assertTrue((result["predictions"]["prediction_count"] == 1).all())
            probability_columns = [
                column for column in result["predictions"] if "__" in column
            ]
            self.assertEqual(len(probability_columns), 15)
            probability = result["predictions"][probability_columns].to_numpy()
            self.assertTrue(np.isfinite(probability).all())
            self.assertTrue(np.all((probability >= 0) & (probability <= 1)))
            self.assertEqual(len(result["bootstrap"]), 10)
            for filename in (
                "model_comparison.csv",
                "feature_view_auc.csv",
                "bootstrap_comparisons.csv",
                "oof_predictions.csv",
                "run_config.json",
                "summary.txt",
            ):
                self.assertTrue((Path(temp_dir) / filename).exists())


if __name__ == "__main__":
    unittest.main()
