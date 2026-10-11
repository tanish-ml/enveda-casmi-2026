"""Official Competition Metric: Mean Reciprocal Rank @ 25 (MRR@25) on InChIKey14.

Scoring details:
- Molecular structures are converted to RDKit molecules.
- Tautomer canonicalization is performed via rdMolStandardize.TautomerEnumerator().Canonicalize(mol).
- The first 14 characters of the InChIKey (InChIKey14 / skeleton) are compared.
- For each test molecule, up to 25 ranked candidate SMILES strings are evaluated.
- If the true InChIKey14 matches candidate at 1-based rank r (1 <= r <= 25), RR = 1 / r.
- If the true molecule is not in the top 25 candidates, RR = 0.
- Overall score is the mean RR across all evaluated molecules.
"""

from typing import List, Dict, Union, Optional
import numpy as np
from src.utils.chem import smiles_to_inchikey14


def reciprocal_rank_single(
    true_smiles_or_key: str,
    pred_candidates: List[str],
    max_k: int = 25,
    is_key: bool = False,
) -> float:
    """Compute Reciprocal Rank (RR@K) for a single molecule.
    
    Args:
        true_smiles_or_key: Ground truth SMILES string or already extracted InChIKey14.
        pred_candidates: Ordered list of predicted SMILES strings (best guess first).
        max_k: Maximum rank cutoff (default 25).
        is_key: True if true_smiles_or_key is already an InChIKey14 string.
        
    Returns:
        Reciprocal rank float in range [0.0, 1.0].
    """
    if is_key:
        true_key = true_smiles_or_key
    else:
        true_key = smiles_to_inchikey14(true_smiles_or_key)

    if not true_key:
        return 0.0

    for rank, candidate_smi in enumerate(pred_candidates[:max_k], start=1):
        cand_key = smiles_to_inchikey14(candidate_smi)
        if cand_key is not None and cand_key == true_key:
            return 1.0 / rank

    return 0.0


def compute_mrr_at_k(
    ground_truth: List[str],
    predictions: List[List[str]],
    max_k: int = 25,
    gt_is_key: bool = False,
) -> Dict[str, float]:
    """Compute MRR@K along with Top-1, Top-5, Top-10, and Top-K recall metrics.
    
    Args:
        ground_truth: List of true SMILES (or InChIKey14 strings if gt_is_key=True).
        predictions: List of lists containing predicted SMILES strings per sample.
        max_k: Rank cutoff (default 25).
        gt_is_key: Whether ground_truth already contains precomputed InChIKey14 strings.
        
    Returns:
        Dictionary containing:
            - 'mrr@k': Mean Reciprocal Rank @ K (Competition Leaderboard Metric)
            - 'top_1_accuracy': Top-1 exact skeleton match rate
            - 'top_5_accuracy': Top-5 recall
            - 'top_10_accuracy': Top-10 recall
            - f'top_{max_k}_accuracy': Top-K recall
            - 'num_samples': Number of evaluated samples
    """
    if len(ground_truth) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(ground_truth)} ground truth vs {len(predictions)} predictions."
        )

    rrs = []
    top_1_hits = 0
    top_5_hits = 0
    top_10_hits = 0
    top_k_hits = 0

    for true_item, preds in zip(ground_truth, predictions):
        rr = reciprocal_rank_single(true_item, preds, max_k=max_k, is_key=gt_is_key)
        rrs.append(rr)

        if rr >= 1.0:
            top_1_hits += 1
        if rr >= 1.0 / 5.0:
            top_5_hits += 1
        if rr >= 1.0 / 10.0:
            top_10_hits += 1
        if rr > 0.0:
            top_k_hits += 1

    n = len(ground_truth)
    if n == 0:
        return {
            f"mrr@{max_k}": 0.0,
            "top_1_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "top_10_accuracy": 0.0,
            f"top_{max_k}_accuracy": 0.0,
            "num_samples": 0,
        }

    return {
        f"mrr@{max_k}": float(np.mean(rrs)),
        "top_1_accuracy": top_1_hits / n,
        "top_5_accuracy": top_5_hits / n,
        "top_10_accuracy": top_10_hits / n,
        f"top_{max_k}_accuracy": top_k_hits / n,
        "num_samples": n,
    }
