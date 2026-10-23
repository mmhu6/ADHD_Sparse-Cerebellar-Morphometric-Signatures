from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from examples.generate_synthetic_data import write_synthetic_csv


class SyntheticExampleTests(unittest.TestCase):
    def test_csv_is_complete_reproducible_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = write_synthetic_csv(root / "first.csv", rows=160, seed=9)
            second = write_synthetic_csv(root / "second.csv", rows=160, seed=9)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            data = pd.read_csv(first)
            self.assertEqual(data.shape, (160, 282))
            self.assertTrue(data.Sub.str.startswith("SYNTHETIC_").all())
            self.assertEqual(data.DX.value_counts().to_dict(), {0: 80, 1: 80})
            self.assertTrue(np.isfinite(data.drop(columns="Sub").to_numpy()).all())
            original = first.read_bytes()
            with self.assertRaises(FileExistsError):
                write_synthetic_csv(first, rows=240, seed=10)
            self.assertEqual(first.read_bytes(), original)

    def test_too_few_rows_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not_created" / "data.csv"
            with self.assertRaisesRegex(ValueError, "at least 160"):
                write_synthetic_csv(output, rows=80)
            self.assertFalse(output.parent.exists())


if __name__ == "__main__":
    unittest.main()
