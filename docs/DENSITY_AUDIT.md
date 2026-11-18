# DENS-001 — Density coverage and units audit

Completed on lab machine: 2026-11-06. No raw inputs modified; no predictive model trained.

## Findings from our files

| Check | Result |
| --- | ---: |
| Solubility records | 103,944 |
| Solvent names in main data | 213 |
| Density records / solvent names | 2,210 / 70 |
| Decimal-comma density records | 1 |
| Invalid numeric/positive density records after explicit parsing | 0 |
| Main rows matched by normalized solvent name | 100,983 |
| Main rows at an exact tabulated density temperature | 55,969 |
| Main rows within matched density temperature bounds | 81,630 |
| Main rows outside matched density bounds | 19,353 |
| Unmatched main rows | 2,961 |
| Solvent/temperature groups with differing reported densities | 257 |

Exact-temperature rows are included in within-range rows. Matching only trims and lowercases names; identity-aware mapping has not yet been completed. Coverage counts apply to raw main data, not the cleaned LR-001 cohort. Different density values from different sources are not automatically errors.

Density range: 0.61003–1.6095 g/cm³; temperature range: 243.15–373.12 K. No density extrapolation was performed. Within-range values used piecewise-linear interpolation of median density at each measured temperature, solely for this diagnostic.

## Concentration check

Using RDKit solvent molar masses, calculated `x/(1-x) × 1000ρ/M_solvent` on 81,630 covered rows. Against the existing mol/L column, median absolute log10 discrepancy = 2.2118033e-06; 95th percentile = 0.000160939451. This strongly supports a solvent-volume interpretation in these rows; small deviations can reflect interpolation differences. This is a consistency check on derived labels, not independent validation of solution density.

The [v2.0 author record](https://zenodo.org/records/15094979) confirms that concentration labels were density-derived. The [v2.2 author clarification](https://zenodo.org/records/22648301) explicitly distinguishes litres of solvent from litres of solution. Earlier project notes used the CSV's mol/L label without this distinction; use the clarified interpretation going forward. LR-001 predicts mole fraction and therefore its reported numerical metrics are unaffected.

## Provenance and limits

Main CSV MD5: `1e6d8a510387746e9faab5f39dc46317`.
Density CSV MD5: `059ae2deadb2ad1523fd4e24b2c88133`.
Both match the corresponding published v2.0 author-record file MD5s.

Script on lab: `scripts/audit_density.py`; run with `/home/puqing/anaconda3/envs/jupyter/bin/python scripts/audit_density.py` from the lab project root. Outputs: `results/density_audit_v1/audit.json`, `density_coverage.csv`, `density_conflicts.csv`. Local copies are under `lab_baseline/`.

Pending: resolve molecular identity/name aliases, review conflicts and sources, quantify interpolation uncertainty, assess sparse density curves, and choose a supported process cohort. Do not fill missing density from the target-derived concentration.
