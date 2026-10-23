import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class LegacyEntryPointTests(unittest.TestCase):
    def test_import_and_help_work_without_data_or_output_side_effects(self):
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(root))
        commands = (
            [sys.executable, "-c", "import ADHD_c; assert ADHD_c.DATA_PATH == 'cerebellum.csv'"],
            [sys.executable, str(root / "ADHD_c.py"), "--help"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            for command in commands:
                result = subprocess.run(
                    command, cwd=temporary, env=environment, capture_output=True,
                    text=True, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("--data", result.stdout)
            self.assertIn("--output", result.stdout)
            self.assertFalse((Path(temporary) / "adhd_cerebellum_paper_outputs_v5").exists())


if __name__ == "__main__":
    unittest.main()
