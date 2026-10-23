import hashlib
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from scripts.prepare_release import REQUIRED_FILES, build_release


class ReleaseTests(unittest.TestCase):
    def make_source(self, source):
        source.mkdir()
        for name in REQUIRED_FILES:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"public placeholder: {name}\n", encoding="utf-8")

    def test_allowlist_excludes_private_records_and_manifest_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            self.make_source(source)
            for name in ("cerebellum.csv", "draft.docx", "outputs/oof_predictions.csv", "docs/plans/private.md", "unlisted.py"):
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("private marker", encoding="utf-8")
            folder, archive = build_release(source, root / "dist" / "cerebellum-adhd")
            expected = set(REQUIRED_FILES) | {"MANIFEST.sha256"}
            actual = {file.relative_to(folder).as_posix() for file in folder.rglob("*") if file.is_file()}
            self.assertEqual(actual, expected)
            with ZipFile(archive) as zipped:
                self.assertEqual(set(zipped.namelist()), {f"cerebellum-adhd/{name}" for name in expected})
                self.assertIsNone(zipped.testzip())
            for line in (folder / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(digest, hashlib.sha256((folder / name).read_bytes()).hexdigest())
            self.assertEqual((source / "cerebellum.csv").read_text(encoding="utf-8"), "private marker")

    def test_missing_source_fails_before_export_and_existing_release_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            destination = root / "dist" / "cerebellum-adhd"
            with self.assertRaises(FileNotFoundError):
                build_release(source, destination)
            self.assertFalse(destination.exists())
            source.rmdir()
            self.make_source(source)
            folder, archive = build_release(source, destination)
            original = archive.read_bytes()
            with self.assertRaises(FileExistsError):
                build_release(source, destination)
            self.assertEqual(archive.read_bytes(), original)
            self.assertTrue((folder / "README.md").is_file())

    def test_selected_license_is_included_without_inventing_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            self.make_source(source)
            (source / "LICENSE").write_text("Chosen by the owner", encoding="utf-8")
            folder, _ = build_release(source, root / "release")
            self.assertEqual((folder / "LICENSE").read_text(encoding="utf-8"), "Chosen by the owner")


if __name__ == "__main__":
    unittest.main()
