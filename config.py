import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

DATA_DIR = ''
OUTPUT_DIR = 'outputs'

FILES = {
    'blsa': '',
    'blsa_amyloid_pet': '',
    'blsa_cognitive_tests': '',
    'biocard': '',
    'biocard_amyloid_pet': '',
    'biocard_csf': '',
    'biocard_tau_pet': '',
    'biocard_cognition': '',
    'nacc': '',
    'adni': '',
    'adni_visits': '',
    'adni_diagnosis': '',
    'adni_amyloid_pet': '',
    'adni_csf': '',
    'adni_tau_pet': '',
    'adni_cognition': '',
    'scan_review': '',
    'brain_volumes': '',
}

EVANS_THRESH = 0.30
CA_THRESH = 90
DESI_THRESH = 4.11
SPPB_THRESH = 10
MIN_AGE = 60
LINK_WINDOW = {'BLSA': 0.5, 'BIOCARD': 0.5, 'NACC': 0.5, 'ADNI': 0.5}
AMYLOID_WINDOW = 2.0
TAU_WINDOW = 2.0
ELECSYS_RATIO_CUT = 0.025
LUMIPULSE_RATIO_POSITIVE = 0.058
LUMIPULSE_RATIO_NEGATIVE = 0.073
FTP_META_TEMPORAL_CUT = 1.23
TAU_REFERENCE_SD = 2.0
DEATH_WINDOW = 2.0
FOLLOWUP_CAP = 12.0
MIN_AT_RISK = 10
MIN_IMAGING_N = 15
N_BOOTSTRAP = 1000

COHORTS = ['BLSA', 'BIOCARD', 'NACC', 'ADNI']
PROG_COHORTS = ['BLSA', 'ADNI']
COGNITION_COHORTS = ['BLSA', 'ADNI', 'NACC']
# RNPH is the internal label for NPH imaging features (Evans Index > 0.30 with callosal angle < 90 degrees or with DESH).
# It is an imaging finding only, not a diagnosis of NPH. Suffixes give the clinical presentation:
# A, no gait or cognitive impairment; G, gait impairment only; C, cognitive impairment only;
# GC, gait and cognitive impairment; All, any presentation.
INCIDENCE_OUTCOME_COHORTS = {'RNPH': COHORTS, 'RNPH-G': PROG_COHORTS, 'RNPH-C': COGNITION_COHORTS, 'RNPH-GC': PROG_COHORTS}

US_AGE_2024 = {'60-69': 41085348, '70-79': 28190749, '≥80': 13697651}
US_POP_STRATA = {('M', '60-69'): 18800000, ('F', '60-69'): 20380000, ('M', '70-79'): 12200000,
                 ('F', '70-79'): 14550000, ('M', '80+'): 5700000, ('F', '80+'): 8680000}
US_60_2024 = {'Women': 53.9, 'Asian': 5.2, 'Black or African American': 10.0, 'White': 72.6, "Bachelor's degree or higher": 31.6}

RACE_NACC = {1: 'White', 2: 'Black or African American', 3: 'American Indian or Alaska Native',
             4: 'Native Hawaiian or Pacific Islander', 5: 'Asian', 50: 'Other', 99: 'Unknown'}
RACE_BLSA = {6: 'White', 3: 'Black or African American', 2: 'American Indian or Alaska Native',
             10: 'Native Hawaiian or Pacific Islander', 9: 'Asian', 7: 'More than one race', 11: 'Other', 8: 'Unknown'}
RACE_ADNI = {1: 'American Indian or Alaska Native', 2: 'Asian', 3: 'Native Hawaiian or Pacific Islander',
             4: 'Black or African American', 5: 'White', 6: 'More than one race', 7: 'Unknown'}

COLORS = {'RNPH-A': '#ADB8BE', 'RNPH-G': '#4C82AF', 'RNPH-C': '#D6952F', 'RNPH-GC': '#A6323B', 'RNPH-All': '#1F3A5F',
          'control': '#7F7F7F'}
LABELS = {'RNPH-A': 'No impairment', 'RNPH-G': 'Gait only', 'RNPH-C': 'Cognition only', 'RNPH-GC': 'Gait and cognition'}
DEFINITION_COLORS = {'Callosal angle': '#1F3A5F', 'DESH': '#9E2B35'}
INK = '#1a1a19'

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'Arimo'],
                     'font.size': 8, 'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
os.makedirs(OUTPUT_DIR, exist_ok=True)

num = lambda s: pd.to_numeric(s, errors='coerce')


def read(key, **kwargs):
    f = os.path.join(DATA_DIR, FILES[key])
    if f.endswith(('.xlsx', '.xls')):
        return pd.read_excel(f, **kwargs)
    return pd.read_csv(f, low_memory=False, **kwargs)


def decimal_year(dt):
    dt = pd.to_datetime(dt, errors='coerce')
    return dt.dt.year + (dt.dt.dayofyear - 0.5) / 365.25


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, max(0, ctr - half), min(1, ctr + half)


def fmt_p(p):
    if pd.isna(p):
        return ''
    return '<.001' if p < 0.001 else (f'{p:.3f}' if p < 0.01 else f'{p:.2f}').lstrip('0')


def fmt_hr(h):
    return '' if pd.isna(h[0]) else f'{h[0]:.2f} ({h[1]:.2f}-{h[2]:.2f})'


def fmt_ci(x):
    return '' if pd.isna(x[0]) else f'{x[0]:.1f}' if pd.isna(x[1]) else f'{x[0]:.1f} ({x[1]:.1f}-{x[2]:.1f})'


def n_pct(k, n):
    return f'{int(k)} ({100 * k / n:.1f})' if n else ''


def mean_sd(x, d=1):
    x = num(pd.Series(x)).dropna()
    return f'{x.mean():.{d}f} ({x.std():.{d}f})' if len(x) else ''


def median_iqr(x, d=1):
    x = pd.Series(x).dropna()
    return f'{x.median():.{d}f} ({x.quantile(.25):.{d}f}-{x.quantile(.75):.{d}f})' if len(x) else ''


def save_table(df, name):
    df.to_csv(os.path.join(OUTPUT_DIR, f'{name}.csv'))
    return df


def save_figure(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
