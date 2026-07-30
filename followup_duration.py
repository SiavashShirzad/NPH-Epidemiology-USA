"""Distribution of visit and MRI follow-up duration by cohort."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300 
plt.rcParams['axes.grid'] = False
sns.set_style("ticks")

COHORT_COLORS = {
    'BLSA':    '#4E79A7',
    'BIOCARD': '#59A14F',
    'NACC':    '#E15759',
    'ADNI':    '#B07AA1'
}

# 1. PREPROCESSING (Ensuring Clean Data)
def preprocess_blsa(df):
    df = df.copy()
    df['visitdate'] = pd.to_datetime(df.get('visitdate'), errors='coerce')
    df['birthdate'] = pd.to_datetime(df.get('birthdate'), errors='coerce')
    df['age'] = (df['visitdate'] - df['birthdate']).dt.days / 365.25
    for c in ['EI', 'sppb', 'cogn_status']:
        if c in df: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def preprocess_biocard(df):
    df = df.copy().rename(columns={
        'JHUANONID':'pid', 'EI':'evans_index', 'VISITYEAR':'visit_year',
        'BIRTHYEAR':'birth_year', 'SPPB':'sppb', 'DIAGNOSIS':'DIAGNOSIS'
    })
    for c in ['evans_index', 'sppb', 'visit_year', 'birth_year']:
        if c in df: df[c] = pd.to_numeric(df[c], errors='coerce')
    df['age'] = df['visit_year'] - df['birth_year']
    return df

def preprocess_nacc(df):
    df = df.copy().rename(columns={
        'NACCID':'pid', 'EI':'evans_index', 'MRIYR':'mri_year',
        'BIRTHYR':'birth_year', 'GAIT':'gait_raw', 'CDRGLOB':'CDRGLOB'
    })
    for c in ['evans_index', 'gait_raw', 'mri_year', 'birth_year', 'CDRGLOB']:
        if c in df: df[c] = pd.to_numeric(df[c], errors='coerce')
    df['age'] = df['mri_year'] - df['birth_year']
    return df

def preprocess_adni(df):
    df = df.copy().rename(columns={
        'Subject':'pid', 'EI':'evans_index', 'Age':'age',
        'Gait':'NXGAIT', 'CDGLOBAL':'CDGLOBAL'
    })
    for c in ['evans_index', 'age', 'NXGAIT', 'CDGLOBAL']:
        if c in df: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# Apply Processing (Handling missing vars gracefully)
try:
    BLSA_proc = preprocess_blsa(BLSA)
    BIOCARD_proc = preprocess_biocard(biocard)
    NACC_proc = preprocess_nacc(NACC)
    ADNI_proc = preprocess_adni(ADNI)
except NameError:
    print("WARNING: Raw dataframes not found. Ensure data is loaded.")
    BLSA_proc, BIOCARD_proc, NACC_proc, ADNI_proc = [pd.DataFrame()]*4


def followup_years(df, pid_col, time_col, is_datetime=False, cohort_name="", followup_type=""):
    """
    Compute per-participant follow-up duration (years).
    Logic: Max(Time) - Min(Time) for visits matching criteria.
    """
    d = df[[pid_col, time_col]].dropna().copy()
    if d.empty:
        return pd.DataFrame(columns=['pid','years','cohort','type'])

    if is_datetime:
        d[time_col] = pd.to_datetime(d[time_col], errors='coerce')
        d = d.dropna(subset=[time_col])
        grouped = d.groupby(pid_col)[time_col]
        dur = (grouped.max() - grouped.min()).dt.days / 365.25
    else:
        d[time_col] = pd.to_numeric(d[time_col], errors='coerce')
        d = d.dropna(subset=[time_col])
        grouped = d.groupby(pid_col)[time_col]
        dur = grouped.max() - grouped.min()

    out = pd.DataFrame({
        'pid': dur.index,
        'years': dur.values,
        'cohort': cohort_name,
        'type': followup_type
    })
    # Clip negative values (rare data errors) and ensure float
    out['years'] = out['years'].clip(lower=0).astype(float)
    return out

# 3. INCLUSION MASKS (Strict Age ≥ 60)
def mask_imaging(df, age_col='age', ei_col='EI'):
    """Age >= 60 AND Imaging Available"""
    return (df[age_col] >= 60) & df[ei_col].notna()

def mask_gait(df, age_col='age', ei_col='EI', gait_col=None):
    """Age >= 60 AND Imaging Available AND Gait Available"""
    mask = (df[age_col] >= 60) & df[ei_col].notna()
    if gait_col and gait_col in df.columns:
        mask &= df[gait_col].notna()
    return mask

def mask_cognitive(df, age_col='age', ei_col='EI', cog_col=None):
    """Age >= 60 AND Imaging Available AND Cognition Available"""
    mask = (df[age_col] >= 60) & df[ei_col].notna()
    if cog_col and cog_col in df.columns:
        mask &= df[cog_col].notna()
    return mask

frames = []

# --- BLSA ---
frames.append(followup_years(
    BLSA_proc[mask_imaging(BLSA_proc, 'age', 'EI')],
    'pid', 'visitdate', True, 'BLSA', 'Imaging'))
frames.append(followup_years(
    BLSA_proc[mask_gait(BLSA_proc, 'age', 'EI', 'sppb')],
    'pid', 'visitdate', True, 'BLSA', 'Gait'))
frames.append(followup_years(
    BLSA_proc[mask_cognitive(BLSA_proc, 'age', 'EI', 'cogn_status')],
    'pid', 'visitdate', True, 'BLSA', 'Cognitive'))

# --- BIOCARD ---
frames.append(followup_years(
    BIOCARD_proc[mask_imaging(BIOCARD_proc, 'age', 'evans_index')],
    'pid', 'visit_year', False, 'BIOCARD', 'Imaging'))
frames.append(followup_years(
    BIOCARD_proc[mask_gait(BIOCARD_proc, 'age', 'evans_index', 'sppb')],
    'pid', 'visit_year', False, 'BIOCARD', 'Gait'))
frames.append(followup_years(
    BIOCARD_proc[mask_cognitive(BIOCARD_proc, 'age', 'evans_index', 'DIAGNOSIS')],
    'pid', 'visit_year', False, 'BIOCARD', 'Cognitive'))

# --- NACC ---
frames.append(followup_years(
    NACC_proc[mask_imaging(NACC_proc, 'age', 'evans_index')],
    'pid', 'mri_year', False, 'NACC', 'Imaging'))
frames.append(followup_years(
    NACC_proc[mask_gait(NACC_proc, 'age', 'evans_index', 'gait_raw')],
    'pid', 'mri_year', False, 'NACC', 'Gait'))
frames.append(followup_years(
    NACC_proc[mask_cognitive(NACC_proc, 'age', 'evans_index', 'CDRGLOB')],
    'pid', 'mri_year', False, 'NACC', 'Cognitive'))

# --- ADNI ---
frames.append(followup_years(
    ADNI_proc[mask_imaging(ADNI_proc, 'age', 'evans_index')],
    'pid', 'age', False, 'ADNI', 'Imaging'))
frames.append(followup_years(
    ADNI_proc[mask_gait(ADNI_proc, 'age', 'evans_index', 'NXGAIT')],
    'pid', 'age', False, 'ADNI', 'Gait'))
frames.append(followup_years(
    ADNI_proc[mask_cognitive(ADNI_proc, 'age', 'evans_index', 'CDGLOBAL')],
    'pid', 'age', False, 'ADNI', 'Cognitive'))

FOLLOWUP_ALL = pd.concat(frames, ignore_index=True)


def plot_followup_panel(df):
    """Creates a 3-panel horizontal boxplot figure"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    types = ['Imaging', 'Gait', 'Cognitive']
    titles = ['A. Imaging Follow-up', 'B. Gait Follow-up', 'C. Cognitive Follow-up']
    
    # Get cohort order present in data
    present_cohorts = [c for c in ['BLSA','BIOCARD','NACC','ADNI'] if c in df['cohort'].unique()]
    palette = [COHORT_COLORS[c] for c in present_cohorts]

    for idx, (ftype, title) in enumerate(zip(types, titles)):
        ax = axes[idx]
        sub = df[df['type'] == ftype]
        
        if sub.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
            continue

        sns.boxplot(
            data=sub,
            y='cohort', x='years',
            order=present_cohorts,
            palette=palette,
            ax=ax,
            fliersize=2, # Small outliers
            width=0.6,
            linewidth=1.0
        )
        
        ax.set_title(title, fontweight='bold', pad=15)
        ax.set_xlabel("Years of Follow-up (Age ≥ 60)")
        ax.set_ylabel("") # Shared Y label
        
        # Add mean/median annotation text
        medians = sub.groupby('cohort')['years'].median()
        counts = sub.groupby('cohort')['years'].count()
        
        # Tufte-style: Clean up spines
        sns.despine(ax=ax)
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('Fig_Followup_Duration.tiff', dpi=300)
    plt.show()

# Stats Table
def summarize(df):
    return df.groupby(['type','cohort'])['years'].agg(
        N='count',
        Median='median',
        Mean='mean',
        IQR_25=lambda x: x.quantile(0.25),
        IQR_75=lambda x: x.quantile(0.75),
        Max='max'
    ).round(2)

print("\n=== FOLLOW-UP STATISTICS (Participants ≥ 60) ===")
summary_stats = summarize(FOLLOWUP_ALL)
print(summary_stats)

print("\n=== GENERATING PLOT ===")
plot_followup_panel(FOLLOWUP_ALL)
