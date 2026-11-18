# Data inventory and preparation

Updated: 2026-11-06 (Asia/Hong_Kong).

## Inputs

- `BigSolDBv2.0.csv`: 103,944 rows. Fields: solute SMILES, temperature in K, solvent name/SMILES, solubility in mole fraction and mol/L, LogS(mol/L), compound name, CAS, PubChem CID, FDA approval flag and source.
- `BigSolDBv2.0_densities.csv`: 2,210 rows. Fields: solvent, temperature in K, density in g/cm³ and source. Not used in LR-001.

Main CSV SHA-256 on the lab machine: `879f55f39db977e295c705e304d011d89d6e10da1daaa3f069181108c1de38b0`.

## LR-001 preparation actually applied

1. Parse temperature and mole-fraction solubility as numeric.
2. Require finite positive temperature and finite solubility strictly between 0 and 1.
3. Require both SMILES strings; parse/canonicalize with RDKit and exclude unparseable structures.
4. Remove duplicate records sharing canonical solute, canonical solvent, temperature, mole-fraction solubility and source. Conflicting measurements remain.
5. Target = log10(mole-fraction solubility).

103,645 valid rows remained before and after deduplication; 299 rows were excluded overall. The combined filter count does not establish a separate count for each exclusion reason. One observed RDKit parse error involved the placeholder `-`.

Retained coverage: 1,448 canonical solutes, 207 canonical solvents, 11,203 pairs, 243.15–425.77 K. These bounds summarize the data; they are not a recommended operating range.

## Evaluation

Canonical solute–solvent pair grouping; seed 42; 20% of pairs held out. Train: 82,911 rows / 8,962 pairs. Test: 20,734 rows / 2,241 pairs. Pair overlap was asserted to be zero. Individual solutes or solvents may appear in both sets. Scaling is fitted on training data only.

## Audit still needed

- Dataset publication/license and formal citation.
- Identity consistency, mixtures, salts and pure-solvent interpretation.
- Source-level dependence, conflicting measurements and measurement uncertainties.
- Per-pair temperature coverage and coverage at common process temperatures.
- Temperature/target outliers, unit consistency and exclusion reasons by category.
- Stronger generalization protocols, including held-out solutes and/or sources.

The raw CSV files were not modified. Molecular descriptors derive solely from SMILES; measured solubility columns are not features.

## 2026-11-06 — Unit/provenance correction and density audit

The mol/L and LogS columns are calculated labels, not independent measurements. The checked formula gives mol solute per litre of pure solvent, distinct from true solution molarity. See [DENSITY_AUDIT.md](DENSITY_AUDIT.md) for author sources, numerical verification, file hashes and coverage. Density has 70 solvent names versus 213 in raw main data; one decimal-comma value requires explicit parsing. No v2.2 data have been substituted.
