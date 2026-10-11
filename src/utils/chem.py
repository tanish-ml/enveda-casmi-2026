"""Cheminformatics helper utilities using RDKit for Enveda CASMI 2026."""

from typing import List, Optional, Tuple
import logging

try:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Common electrospray ionization (ESI) adducts and mass shifts
# Neutral Mass M = (Precursor m/z * z - adduct_shift) / mult
# Format: adduct_name: (mass_offset, charge)
ADDUCT_MASS_OFFSETS = {
    # Positive ion mode
    "[M+H]+": 1.007276,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823,
    "[M+H-H2O]+": -17.00274,
    "[M+2H]2+": 2.014552 / 2.0,
    # Negative ion mode
    "[M-H]-": -1.007276,
    "[M+Cl]-": 34.969402,
    "[M+HCOO]-": 44.998201,
    "[M+CH3COO]-": 59.013851,
    "[M-2H]2-": -2.014552 / 2.0,
}


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string using RDKit tautomer enumeration.
    
    Args:
        smiles: Raw input SMILES.
        
    Returns:
        Canonicalized SMILES or None if parsing fails.
    """
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        te = rdMolStandardize.TautomerEnumerator()
        canon_mol = te.Canonicalize(mol)
        return Chem.MolToSmiles(canon_mol)
    except Exception as exc:
        logger.debug(f"Failed to canonicalize SMILES '{smiles}': {exc}")
        return None


def smiles_to_inchikey14(smiles: str) -> Optional[str]:
    """Extract the first 14 characters of InChIKey (InChIKey14) following
    competition tautomer canonicalization rules.
    
    The first block (14 characters) encodes molecular skeleton / connectivity,
    ignoring stereochemistry and isotopic variation.
    
    Args:
        smiles: Input SMILES string.
        
    Returns:
        14-character InChIKey skeleton string, or None if invalid.
    """
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        te = rdMolStandardize.TautomerEnumerator()
        canon_mol = te.Canonicalize(mol)
        inchikey = Chem.MolToInchiKey(canon_mol)
        if inchikey and len(inchikey) >= 14:
            return inchikey[:14]
        return None
    except Exception as exc:
        logger.debug(f"Error computing InChIKey14 for '{smiles}': {exc}")
        return None


def deduplicate_candidates(candidates: List[str], max_k: int = 25) -> Tuple[List[str], List[str]]:
    """Deduplicate candidate SMILES based on their InChIKey14 representation.
    
    Submitting stereoisomers or tautomers that map to the same InChIKey14 wastes slots
    in top-25 ranking. This function preserves the highest-ranked unique candidate.
    
    Args:
        candidates: List of candidate SMILES in order of decreasing confidence.
        max_k: Maximum number of candidates to retain (competition max is 25).
        
    Returns:
        Tuple of (deduplicated_smiles_list, corresponding_inchikey14_list).
    """
    seen_inchikey14 = set()
    dedup_smiles = []
    dedup_keys = []

    for smi in candidates:
        if not smi:
            continue
        ik14 = smiles_to_inchikey14(smi)
        if ik14 is not None:
            if ik14 not in seen_inchikey14:
                seen_inchikey14.add(ik14)
                dedup_smiles.append(smi)
                dedup_keys.append(ik14)
        else:
            # Fallback if invalid molecule or RDKit fails: retain raw SMILES if unique
            if smi not in seen_inchikey14:
                seen_inchikey14.add(smi)
                dedup_smiles.append(smi)
                dedup_keys.append("")

        if len(dedup_smiles) >= max_k:
            break

    return dedup_smiles, dedup_keys


def calculate_neutral_mass(precursor_mz: float, adduct: str) -> Optional[float]:
    """Calculate neutral monoisotopic mass from precursor m/z and adduct annotation.
    
    Args:
        precursor_mz: Precursor mass-to-charge ratio.
        adduct: Adduct string representation (e.g., '[M+H]+', '[M-H]-').
        
    Returns:
        Estimated neutral mass in Da, or None if adduct is unknown.
    """
    if adduct in ADDUCT_MASS_OFFSETS:
        offset = ADDUCT_MASS_OFFSETS[adduct]
        return precursor_mz - offset
    return None
