# NPH Epidemiology in US Aging Cohorts

Analysis code for *Prevalence, Incidence, and Progression of Normal Pressure Hydrocephalus in US Aging Cohorts*.

## Data

The analyses use participant-level data from four aging cohorts: BLSA, BIOCARD, NACC, and ADNI. These data are not distributed here and must be requested from each cohort directly. Each script expects the corresponding CSV in the working directory:

```
BLSA_epidemiology.csv
BIOCARD_epidemiology.csv
NACC_epidemiology.csv
ADNI_epidemiology.csv
```

Each file contains one row per participant-visit, with the imaging indices (Evans Index, callosal angle, DESI) and the cohort's native gait and cognitive assessments.

## Scripts

| Script | Output |
|---|---|
| `table1_demographics.py` | Table 1 — demographics and follow-up duration |
| `prevalence_phenotypes.py` | Period prevalence by age band and cohort |
| `prevalence_figure.py` | Figure 1 — stacked-bar prevalence |
| `prevalence_desh.py` | Sensitivity analysis using the DESH (DESI) definition |
| `prevalence_sex_stratified.py` | Age- and sex-stratified prevalence |
| `prevalence_by_race.py` | Race-stratified prevalence with marginal standardization |
| `incidence_figure.py` | Figure 2 — stacked-bar incidence |
| `progression_competing_risks.py` | Figure 3 — cumulative incidence of symptomatic conversion |
| `followup_duration.py` | Follow-up duration by cohort |
| `rate_of_change.py` | Mixed-effects cognitive and gait trajectories |

`incidence_figure.py` reads `NPH_Incidence_Rates_Final.csv`, so run `incidence_rates.py` first. The other scripts are independent.

## Definitions

Radiologic NPH (RNPH) is defined as an Evans Index > 0.30 with a callosal angle < 90°. Participants with RNPH are classified by clinical features into asymptomatic (RNPH-A), gait impairment only (RNPH-G), cognitive impairment only (RNPH-C), or both, corresponding to possible NPH (RNPH-GC).

## Requirements

Python 3.11 with `pandas`, `numpy`, `scipy`, `statsmodels`, `lifelines`, and `matplotlib`.

```
pip install pandas numpy scipy statsmodels lifelines matplotlib
```

## Imaging pipeline

Code for the automated imaging measurements is in separate repositories:

- Evans Index: https://github.com/SiavashShirzad/AutomatedEvansIndex
- Callosal angle: https://github.com/SiavashShirzad/AutomatedCallosalAngle
- DESI: https://github.com/SiavashShirzad/Disproportionately-Elevated-Sulcal-Index-DESI-

Trained model weights and the full integrated pipeline are available on reasonable request.
