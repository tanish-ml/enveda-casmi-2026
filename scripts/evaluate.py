"""Evaluation CLI for Enveda CASMI 2026.

Evaluates predicted ranked candidates against ground truth structures using the official
competition metric: MRR@25 on InChIKey14 after RDKit tautomer canonicalization.

Usage:
    python scripts/evaluate.py --val-file data/val.parquet --pred-file submissions/val_preds.csv
"""

import argparse
import sys
import polars as pl
from src.metric import compute_mrr_at_k


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions using Enveda CASMI 2026 official MRR@25 metric."
    )
    parser.add_argument(
        "--val-file",
        type=str,
        required=True,
        help="Path to validation Parquet or CSV file containing molecule_id and normalized_smiles.",
    )
    parser.add_argument(
        "--pred-file",
        type=str,
        required=True,
        help="Path to predictions CSV containing molecule_id and semicolon-separated smiles.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=25,
        help="Maximum rank cutoff (default: 25).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading validation data from: {args.val_file}")
    if args.val_file.endswith(".parquet"):
        val_df = pl.read_parquet(args.val_file)
    else:
        val_df = pl.read_csv(args.val_file)

    print(f"Loading predictions from: {args.pred_file}")
    pred_df = pl.read_csv(args.pred_file)

    if "molecule_id" not in val_df.columns or "normalized_smiles" not in val_df.columns:
        print("Error: val-file must contain 'molecule_id' and 'normalized_smiles' columns.", file=sys.stderr)
        sys.exit(1)

    if "molecule_id" not in pred_df.columns or "smiles" not in pred_df.columns:
        print("Error: pred-file must contain 'molecule_id' and 'smiles' columns.", file=sys.stderr)
        sys.exit(1)

    # Join on molecule_id to align samples
    joined = val_df.select(["molecule_id", "normalized_smiles"]).join(
        pred_df.select(["molecule_id", "smiles"]),
        on="molecule_id",
        how="inner"
    )

    print(f"Evaluating {len(joined)} aligned samples...")

    gt_smiles = joined["normalized_smiles"].to_list()
    raw_preds = joined["smiles"].to_list()

    # Split semicolon-delimited candidate strings
    candidate_lists = [
        [s.strip() for s in str(p).split(";") if s.strip()] if p else []
        for p in raw_preds
    ]

    metrics = compute_mrr_at_k(gt_smiles, candidate_lists, max_k=args.max_k)

    print("\n" + "=" * 50)
    print("           OFFICIAL EVALUATION RESULTS            ")
    print("=" * 50)
    print(f"  MRR@{args.max_k} (Leaderboard Metric): {metrics[f'mrr@{args.max_k}']:.5f}")
    print(f"  Top-1 Accuracy:               {metrics['top_1_accuracy'] * 100:.2f}%")
    print(f"  Top-5 Accuracy:               {metrics['top_5_accuracy'] * 100:.2f}%")
    print(f"  Top-10 Accuracy:              {metrics['top_10_accuracy'] * 100:.2f}%")
    print(f"  Top-{args.max_k} Accuracy:             {metrics[f'top_{args.max_k}_accuracy'] * 100:.2f}%")
    print(f"  Total Samples Evaluated:      {metrics['num_samples']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
