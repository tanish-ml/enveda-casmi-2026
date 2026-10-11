"""Preprocessing utilities for tandem mass spectrometry (MS/MS) data."""

from typing import List, Tuple, Union, Optional
import numpy as np
import re


def parse_collision_energy(ce_val: Union[str, float, int, list, None]) -> float:
    """Parse collision energy values into standard float representation (eV).
    
    Collision energies in the dataset can appear in varied formats:
    e.g. 60, 60.0, '60', '60.0', '[60.0]', 'NCE 35%', '30-50 eV', etc.
    
    Args:
        ce_val: Raw collision energy value.
        
    Returns:
        Extracted float value in eV or default (30.0) if unparseable.
    """
    if ce_val is None:
        return 30.0
    if isinstance(ce_val, (int, float)):
        return float(ce_val)
    if isinstance(ce_val, (list, tuple)):
        if len(ce_val) > 0:
            return parse_collision_energy(ce_val[0])
        return 30.0

    s = str(ce_val).strip()
    # Extract first sequence of digits and optional decimal
    match = re.search(r"[-+]?\d*\.?\d+", s)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return 30.0
    return 30.0


def filter_and_normalize_peaks(
    mzs: Union[List[float], np.ndarray],
    intensities: Union[List[float], np.ndarray],
    max_peaks: int = 300,
    min_relative_intensity: float = 0.005,
    power_scaling: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter and normalize fragment peaks from MS/MS spectra.
    
    Args:
        mzs: Array of fragment mass-to-charge (m/z) values.
        intensities: Array of corresponding peak intensities.
        max_peaks: Maximum number of most intense peaks to retain.
        min_relative_intensity: Minimum intensity relative to base peak (e.g. 0.5%).
        power_scaling: Power transformation exponent (default sqrt / 0.5) to dampen high peaks.
        
    Returns:
        Tuple of (filtered_sorted_mzs, normalized_intensities).
    """
    mzs = np.asarray(mzs, dtype=np.float32)
    intensities = np.asarray(intensities, dtype=np.float32)

    if len(mzs) == 0 or len(intensities) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    # Base peak normalization
    max_intensity = np.max(intensities)
    if max_intensity > 0:
        rel_intensities = intensities / max_intensity
    else:
        rel_intensities = intensities

    # Filter out noise floor
    valid_mask = rel_intensities >= min_relative_intensity
    mzs = mzs[valid_mask]
    intensities = rel_intensities[valid_mask]

    # Retain top max_peaks if exceeded
    if len(mzs) > max_peaks:
        top_indices = np.argsort(intensities)[-max_peaks:]
        mzs = mzs[top_indices]
        intensities = intensities[top_indices]

    # Sort in ascending m/z order
    sort_order = np.argsort(mzs)
    mzs = mzs[sort_order]
    intensities = intensities[sort_order]

    # Power scaling (square root) and L2 normalize
    if power_scaling != 1.0:
        intensities = np.power(np.maximum(intensities, 0), power_scaling)
    norm = np.linalg.norm(intensities)
    if norm > 0:
        intensities = intensities / norm

    return mzs, intensities


def bin_spectrum(
    mzs: np.ndarray,
    intensities: np.ndarray,
    min_mz: float = 10.0,
    max_mz: float = 1000.0,
    bin_size: float = 0.1,
) -> np.ndarray:
    """Bin MS/MS spectrum into a fixed-width 1D vector representation.
    
    Args:
        mzs: Fragment m/z values.
        intensities: Normalized fragment intensities.
        min_mz: Minimum m/z bound.
        max_mz: Maximum m/z bound.
        bin_size: Bin resolution (in Da).
        
    Returns:
        1D numpy array representing binned spectrum intensity.
    """
    num_bins = int(np.ceil((max_mz - min_mz) / bin_size))
    binned = np.zeros(num_bins, dtype=np.float32)

    if len(mzs) == 0:
        return binned

    bin_indices = np.floor((mzs - min_mz) / bin_size).astype(int)
    valid = (bin_indices >= 0) & (bin_indices < num_bins)

    for idx, intensity in zip(bin_indices[valid], intensities[valid]):
        binned[idx] = max(binned[idx], intensity)

    return binned


def preprocess_spectrum(
    precursor_mz: float,
    adduct: str,
    collision_energy: Union[str, float, None],
    ms2_mzs: Union[List[float], np.ndarray],
    ms2_intensities: Union[List[float], np.ndarray],
    max_peaks: int = 300,
) -> dict:
    """End-to-end spectrum preprocessing record.
    
    Args:
        precursor_mz: Monoisotopic precursor m/z.
        adduct: Adduct annotation.
        collision_energy: Collision energy string or float.
        ms2_mzs: Fragment peaks m/z array.
        ms2_intensities: Fragment intensities array.
        max_peaks: Maximum peaks to retain.
        
    Returns:
        Dictionary of preprocessed spectral features.
    """
    clean_ce = parse_collision_energy(collision_energy)
    clean_mzs, clean_intensities = filter_and_normalize_peaks(
        ms2_mzs, ms2_intensities, max_peaks=max_peaks
    )
    return {
        "precursor_mz": float(precursor_mz),
        "adduct": adduct,
        "collision_energy": clean_ce,
        "ms2_mzs": clean_mzs,
        "ms2_intensities": clean_intensities,
        "num_peaks": len(clean_mzs),
    }
