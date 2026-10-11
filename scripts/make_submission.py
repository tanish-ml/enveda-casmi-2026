"""Kaggle submission builder and format validator for Enveda CASMI 2026.

Ensures:
1. Header matches exact specification: `molecule_id,smiles`.
2. Semicolon delimiter between candidate SMILES strings.
3. Up to 25 candidates per molecule (excess truncated).
4. Candidates are deduplicated on InChIKey14 to avoid wasting slots on stereoisomers.
5. All test molecules are present with zero missing rows.

Usage:
    python scripts/make_submission.py --test-file data/test.parquet --candidates-file raw_cands.json --output submission.csv
"""

import argparse
import sys
import polars as pl
from typing import Dict, List
import json
from src.utils.chem import deduplicate_candidates


def parse_args():
    parser = argparse.ArgumentParser(description="Create and validate Kaggle submission file.")
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test.parquet or test CSV containing test molecule_id column.",
    )
    parser.add_argument(
        "--candidates-file",
        type=str,
        required=True,
        help="Path to JSON file mapping molecule_id -> list of candidate SMILES strings.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Path to save output submission.csv (default: submission.csv).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=25,
        help="Maximum candidates per molecule (competition limit is 25).",
    )
    return parser.parse_args()


def build_submission(
    test_file: str,
    candidates_path: str,
    output_path: str,
    max_candidates: int = 25,
):
    print(f"Reading test references from: {test_file}")
    if test_file.endswith(".parquet"):
        test_df = pl.read_parquet(test_file)
    else:
        test_df = pl.read_csv(test_file)

    test_ids = test_df["molecule_id"].to_list()
    print(f"Total test molecule_ids: {len(test_ids)}")

    print(f"Reading candidate predictions from: {candidates_path}")
    with open(candidates_path, "r", encoding="utf-8") as f:
        cand_dict: Dict[str, List[str]] = json.load(f)

    rows = []
    missing_count = 0

    for mol_id in test_ids:
        raw_cands = cand_dict.get(mol_id, [])
        if not raw_cands:
            missing_count += 1
            formatted_smiles = ""
        else:
            # Deduplicate by InChIKey14 and keep top max_candidates
            clean_cands, _ = deduplicate_candidates(raw_cands, max_k=max_candidates)
            formatted_smiles = ";".join(clean_cands)

        rows.append({
            "molecule_id": mol_id,
            "smiles": formatted_smiles,
        })

    sub_df = pl.DataFrame(rows)
    sub_df.write_csv(output_path)
    print(f"\nSaved submission to: {output_path}")
    print(f"  Rows written: {len(sub_df)}")
    print(f"  Missing predictions: {missing_count} / {len(test_ids)}")

    # Verification checks
    assert list(sub_df.columns) == ["molecule_id", "smiles"], "Column headers must be 'molecule_id,smiles'"
    print("Verification passed successfully! File is ready for submission.")


def main():
    args = parse_args()
    build_submission(
        test_file=args.test_file,
        candidates_path=args.candidates_file,
        output_path=args.output,
        max_candidates=args.max_candidates,
    )


if __name__ == "__main__":
    main()
