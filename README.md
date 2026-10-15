# NPH Epidemiology in US Aging Cohorts

Analysis code for *Emergence of Symptoms Over Time in Individuals With Radiologic Evidence of Possible NPH*.

Evans Index, callosal angle, and DESI were measured with the NPH imaging pipeline:
https://github.com/SiavashShirzad/nph-imaging-pipeline

## Data

Data from BLSA, BIOCARD, NACC, and ADNI are not included and must be requested from each cohort. Set `DATA_DIR` and the file for each input in `FILES` in `config.py`. Each input is a single table (CSV or Excel).

| Input | Contents |
|---|---|
| `blsa` | BLSA visits: imaging measures, SPPB, consensus cognitive status and diagnosis, demographics, death, stroke and TIA history |
| `blsa_amyloid_pet` | BLSA PiB PET status and estimated age at amyloid onset |
| `blsa_cognitive_tests` | BLSA Trail Making A and B, Digit Symbol, letter fluency, and CVLT, one row per visit |
| `biocard` | BIOCARD visits: imaging measures, SPPB, consensus diagnosis, demographics, death, diagnosis flags |
| `biocard_amyloid_pet` | BIOCARD amyloid PET status |
| `biocard_csf` | BIOCARD CSF Aβ42 and Aβ40 (Lumipulse) |
| `biocard_tau_pet` | BIOCARD MK-6240 Braak III and IV SUVR |
| `biocard_cognition` | BIOCARD executive and episodic memory composites |
| `nacc` | NACC visits and MRI: imaging measures, gait examination, CDR, diagnoses, neuropathology, demographics |
| `adni` | ADNI MRI: imaging measures and acquisition date |
| `adni_visits` | ADNI visits: CDR, gait examination, diagnoses, demographics, medical history, death; `_age` and `_date` columns give the age and date of each measure |
| `adni_diagnosis` | ADNI diagnostic summary |
| `adni_amyloid_pet` | ADNI amyloid PET status |
| `adni_csf` | ADNI CSF p-tau181 and Aβ42 (Elecsys) |
| `adni_tau_pet` | ADNI flortaucipir meta-temporal SUVR |
| `adni_cognition` | ADNI-EF and ADNI-MEM composites |
| `scan_review` | Visual review of flagged scans: decision, revised callosal angle, and DESH rating |
| `brain_volumes` | DLMUSE total brain volume and intracranial volume for each MRI |

## Scripts

Run each script from the repository folder. Tables (CSV) and figures (PDF and PNG) are written to `outputs`.

| Script | Output |
|---|---|
| `data.py` | Harmonized imaging and clinical data; scan review summary |
| `table1_demographics.py` | Table 1; eTables 2 and 3 |
| `prevalence_phenotypes.py` | Figures 1 and 2; eTable 4 |
| `incidence.py` | eTable 5; eFigure 4 |
| `progression_competing_risks.py` | Figures 3 and 4; eTables 6 to 8; eFigures 5 and 6 |
| `imaging_definitions.py` | eFigure 3 |
| `biomarkers.py` | eFigure 7 |
| `cognitive_profile.py` | eFigure 8 |
| `coexisting_conditions.py` | eTable 9; eFigure 9 |
| `us_projection.py` | eTable 10 |
| `participant_flow.py` | eFigure 1 |

`config.py` holds the inputs, thresholds, and shared functions. `covariates.py` builds amyloid, tau, parkinsonism, stroke, and brain volume covariates.

## Requirements

Python 3.11 or later with `pandas`, `numpy`, `scipy`, `statsmodels`, `lifelines`, `matplotlib`, and `openpyxl`.

```
pip install pandas numpy scipy statsmodels lifelines matplotlib openpyxl
```
