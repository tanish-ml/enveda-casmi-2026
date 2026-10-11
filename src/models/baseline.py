"""Baseline neural models for MS/MS spectra encoding and candidate re-ranking."""

from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.chem import deduplicate_candidates, smiles_to_inchikey14

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class SpectrumMLPEncoder(nn.Module):
    """Deep MLP encoder mapping binned MS/MS spectra and metadata to latent embeddings."""

    def __init__(
        self,
        input_dim: int = 1980,
        meta_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (1024, 512, 512),
        output_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        curr_in = input_dim + 64
        layers = []
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(curr_in, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            curr_in = h_dim

        layers.append(nn.Linear(curr_in, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, spectrum: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            spectrum: Binned spectrum tensor of shape (batch, input_dim).
            metadata: Metadata tensor of shape (batch, meta_dim).
            
        Returns:
            Latent representation tensor of shape (batch, output_dim).
        """
        meta_emb = self.meta_proj(metadata)
        x = torch.cat([spectrum, meta_emb], dim=-1)
        return self.network(x)


class FingerprintPredictor(nn.Module):
    """Predicts probabilistic molecular fingerprints (e.g. 2048-bit Morgan/ECFP4)
    directly from MS/MS spectra and metadata."""

    def __init__(
        self,
        input_dim: int = 1980,
        meta_dim: int = 3,
        fp_dim: int = 2048,
        hidden_dims: Tuple[int, ...] = (1024, 1024, 1024),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.encoder = SpectrumMLPEncoder(
            input_dim=input_dim,
            meta_dim=meta_dim,
            hidden_dims=hidden_dims,
            output_dim=fp_dim,
            dropout=dropout,
        )

    def forward(self, spectrum: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """Predicts fingerprint bit logits."""
        return self.encoder(spectrum, metadata)

    def predict_probs(self, spectrum: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """Returns bit probabilities in [0, 1]."""
        logits = self.forward(spectrum, metadata)
        return torch.sigmoid(logits)


class CandidateReRanker:
    """Scores candidate molecules against query spectra using fingerprint similarity
    and outputs deduplicated top-25 candidate lists compliant with competition rules.
    """

    def __init__(self, fp_radius: int = 2, fp_bits: int = 2048):
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits

    def compute_morgan_fp(self, smiles: str) -> Optional[np.ndarray]:
        """Compute bit vector for candidate SMILES."""
        if not RDKIT_AVAILABLE or not smiles:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
            arr = np.zeros((self.fp_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception:
            return None

    def rank_candidates(
        self,
        predicted_fp_probs: np.ndarray,
        candidate_smiles: List[str],
        max_k: int = 25,
    ) -> List[str]:
        """Rank a pool of candidate molecules by continuous Tanimoto similarity
        to the predicted fingerprint, deduplicating on InChIKey14.
        
        Continuous Tanimoto (generalized Jaccard):
            T(p, c) = sum(p * c) / (sum(p^2) + sum(c^2) - sum(p * c))
            
        Args:
            predicted_fp_probs: Array of shape (fp_bits,) with predicted bit probabilities.
            candidate_smiles: List of candidate SMILES to rank.
            max_k: Max candidates to return (<= 25).
            
        Returns:
            List of up to max_k deduplicated SMILES sorted from highest to lowest score.
        """
        if not candidate_smiles:
            return []

        scores = []
        p = predicted_fp_probs
        p_sq = np.sum(p ** 2)

        valid_cands = []
        for smi in candidate_smiles:
            c = self.compute_morgan_fp(smi)
            if c is None:
                continue
            dot = np.dot(p, c)
            c_sq = np.sum(c ** 2)
            denom = p_sq + c_sq - dot
            tanimoto = dot / max(denom, 1e-8)
            scores.append(tanimoto)
            valid_cands.append(smi)

        if not valid_cands:
            return candidate_smiles[:max_k]

        # Sort descending by similarity score
        sorted_indices = np.argsort(scores)[::-1]
        sorted_smiles = [valid_cands[i] for i in sorted_indices]

        # Deduplicate by InChIKey14 to maximize MRR@25 efficiency
        dedup_smiles, _ = deduplicate_candidates(sorted_smiles, max_k=max_k)
        return dedup_smiles
