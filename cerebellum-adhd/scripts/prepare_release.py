"""Export an explicit allowlist of public source files; never scan data folders."""

import hashlib
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile


REQUIRED_FILES = (
    ".gitignore",
    ".gitattributes",
    ".github/workflows/tests.yml",
    "README.md",
    "requirements.txt",
    "requirements-paper.txt",
    "ADHD_c.py",
    "cereb_adhd_screen.py",
    "cereb_adhd_optimization.py",
    "cereb_adhd_v2.py",
    "cerebnorm_adhd_screen.py",
    "cerebnorm_five_model_ablation.py",
    "docs/DATA.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RELEASING.md",
    "examples/generate_synthetic_data.py",
    "scripts/prepare_release.py",
    "tests/__init__.py",
    "tests/synthetic_data.py",
    "tests/test_cereb_adhd_screen.py",
    "tests/test_cereb_adhd_optimization.py",
    "tests/test_cereb_adhd_v2.py",
    "tests/test_cerebnorm_adhd_screen.py",
    "tests/test_cerebnorm_five_model_ablation.py",
    "tests/test_legacy_cli.py",
    "tests/test_release.py",
    "tests/test_synthetic_example.py",
)
OPTIONAL_FILES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE", "CITATION.cff")


def public_files(source_root: Path) -> tuple[str, ...]:
    root = Path(source_root).resolve()
    selected = REQUIRED_FILES + tuple(name for name in OPTIONAL_FILES if (root / name).exists())
    for name in selected:
        source = root / name
        if not source.is_file():
            raise FileNotFoundError(f"Required public source file is missing: {name}")
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError(f"Public source must be a regular file inside the project: {name}")
    return selected


def build_release(source_root: Path, destination: Path) -> tuple[Path, Path]:
    source_root = Path(source_root).resolve()
    destination = Path(destination).absolute()
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError(f"Release already exists: {destination} or {archive}")
    selected = public_files(source_root)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = []
    for name in selected:
        output = destination / name
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, output)
        manifest.append(f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {name}\n")
    (destination / "MANIFEST.sha256").write_text("".join(manifest), encoding="utf-8", newline="\n")
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as zipped:
        for name in (*selected, "MANIFEST.sha256"):
            zipped.write(destination / name, f"{destination.name}/{name}")
    return destination, archive


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    try:
        destination, archive = build_release(root, root / "dist" / "cerebellum-adhd")
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Public source folder: {destination}")
    print(f"Source ZIP: {archive}")
    print("Original data and experiment outputs were left in place.")
    if not any((destination / name).is_file() for name in ("LICENSE", "LICENSE.txt", "LICENSE.md")):
        print("License selection is pending; add a license before an open-source release.")


if __name__ == "__main__":
    main()
