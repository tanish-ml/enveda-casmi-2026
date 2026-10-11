"""Dataset loaders and PyTorch Dataset for Enveda CASMI 2026 Parquet data."""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.data.preprocess import filter_and_normalize_peaks, parse_collision_energy, bin_spectrum
from src.utils.chem import calculate_neutral_mass, smiles_to_inchikey14


def load_spectra_dataframe(parquet_path: str) -> pl.DataFrame:
    """Load train or test Parquet file using Polars for high performance.
    
    Args:
        parquet_path: Path to train.parquet or test.parquet.
        
    Returns:
        Polars DataFrame.
    """
    return pl.read_parquet(parquet_path)


class MassSpectraDataset(Dataset):
    """PyTorch Dataset for mass spectrometry data from Enveda CASMI 2026."""

    def __init__(
        self,
        df: Union[pl.DataFrame, str],
        max_peaks: int = 300,
        binned: bool = True,
        min_mz: float = 10.0,
        max_mz: float = 1000.0,
        bin_size: float = 0.5,
        is_test: bool = False,
    ):
        """
        Args:
            df: Polars DataFrame or filepath to parquet file.
            max_peaks: Maximum number of peak tokens to keep.
            binned: If True, return dense binned vector; if False, return peak tokens.
            min_mz: Minimum m/z bound for binning.
            max_mz: Maximum m/z bound for binning.
            bin_size: Bin size in Daltons (e.g. 0.5 Da).
            is_test: True if test set (no ground-truth SMILES available).
        """
        if isinstance(df, str):
            self.df = pl.read_parquet(df)
        else:
            self.df = df

        self.max_peaks = max_peaks
        self.binned = binned
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.bin_size = bin_size
        self.is_test = is_test

        self.has_smiles = "normalized_smiles" in self.df.columns and not is_test
        self.num_bins = int(np.ceil((max_mz - min_mz) / bin_size))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.row(idx, named=True)

        mol_id = str(row.get("molecule_id", f"mol_{idx}"))
        prec_mz = float(row.get("precursor_mz", 0.0))
        adduct = str(row.get("adduct", "[M+H]+"))
        ce = parse_collision_energy(row.get("collision_energy", 30.0))

        raw_mzs = row.get("ms2_mzs", [])
        raw_intens = row.get("ms2_normalized_intensities", [])

        # Filter and normalize peaks
        mzs, intens = filter_and_normalize_peaks(
            raw_mzs, raw_intens, max_peaks=self.max_peaks
        )

        neutral_mass = calculate_neutral_mass(prec_mz, adduct)
        if neutral_mass is None:
            neutral_mass = prec_mz

        meta_tensor = torch.tensor([prec_mz, neutral_mass, ce], dtype=torch.float32)

        if self.binned:
            spectrum_vec = bin_spectrum(
                mzs, intens,
                min_mz=self.min_mz,
                max_mz=self.max_mz,
                bin_size=self.bin_size
            )
            spec_tensor = torch.from_numpy(spectrum_vec)
        else:
            # Token format (N, 2): [[mz_1, int_1], ...] padded to max_peaks
            spec_array = np.zeros((self.max_peaks, 2), dtype=np.float32)
            n = min(len(mzs), self.max_peaks)
            if n > 0:
                spec_array[:n, 0] = mzs[:n]
                spec_array[:n, 1] = intens[:n]
            spec_tensor = torch.from_numpy(spec_array)

        item = {
            "molecule_id": mol_id,
            "spectrum": spec_tensor,
            "metadata": meta_tensor,
            "precursor_mz": prec_mz,
            "adduct": adduct,
            "collision_energy": ce,
        }

        if self.has_smiles:
            smiles = str(row["normalized_smiles"])
            item["smiles"] = smiles
            item["inchikey14"] = smiles_to_inchikey14(smiles) or ""

        return item
