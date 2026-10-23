"""Write independently generated inputs for a software demonstration."""

import argparse
from pathlib import Path
import sys

# Support running this file directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.synthetic_data import make_synthetic_frame


def write_synthetic_csv(output: Path, rows: int = 240, seed: int = 42) -> Path:
    if rows < 160:
        raise ValueError("Use at least 160 rows for the normative training partitions")
    frame = make_synthetic_frame(n=rows, seed=seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic/cerebellum.csv"))
    parser.add_argument("--rows", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        output = write_synthetic_csv(args.output, args.rows, args.seed)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Synthetic software-demo input: {output.resolve()} ({args.rows} rows)")
    print("All values are generated. Scores on this input are not research results.")


if __name__ == "__main__":
    main()
