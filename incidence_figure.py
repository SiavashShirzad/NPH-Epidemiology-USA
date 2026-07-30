"""Figure 2: stacked-bar incidence of RNPH phenotypes by cohort."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import re
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.grid'] = False

COHORT_ORDER = ['BLSA', 'BIOCARD', 'NACC', 'ADNI']
SUBTYPES = ['RNPH-G', 'RNPH-C', 'RNPH-GC (Possible NPH)']   # stacked (symptomatic composition)
OVERALL = 'RNPH'                                             # overlaid reference (radiologic, overall)

# Category colors - same coordinated palette as the prevalence figure (Figure 1).
# RNPH-G steel blue, RNPH-C amber, RNPH-GC JAMA red; RNPH-All dark slate for the overall reference.
PHENO_COLORS = {'RNPH': '#1F3A5F',                    # deep slate (overall reference, high contrast)
                'RNPH-G': '#4C82AF',                  # steel blue (isolated gait)
                'RNPH-C': '#E3A24A',                  # amber (isolated cognitive)
                'RNPH-GC (Possible NPH)': '#A6323B'}  # JAMA red (possible NPH)
SHORT = {'RNPH': 'RNPH-All', 'RNPH-G': 'RNPH-G', 'RNPH-C': 'RNPH-C',
         'RNPH-GC (Possible NPH)': 'RNPH-GC'}

df = pd.read_csv("NPH_Incidence_Rates_Final.csv")

def parse_ci(s):
    m = re.match(r'\s*([\d.]+)\s*\[\s*([\d.]+)\s*-\s*([\d.]+)\s*\]', str(s))
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (np.nan, np.nan, np.nan)

parsed = df['Adjusted (gamma 95% CI)'].apply(parse_ci)
df['rate'] = [p[0] for p in parsed]
df['lo'] = [p[1] for p in parsed]
df['hi'] = [p[2] for p in parsed]
R = df.set_index(['Cohort', 'Phenotype'])['rate']
LO = df.set_index(['Cohort', 'Phenotype'])['lo']
HI = df.set_index(['Cohort', 'Phenotype'])['hi']

fig, ax = plt.subplots(figsize=(11, 7.5))
x = np.arange(len(COHORT_ORDER))
bar_w = 0.55
bottoms = np.zeros(len(COHORT_ORDER))

for cat in SUBTYPES:
    vals = np.array([R.get((c, cat), np.nan) for c in COHORT_ORDER])
    ax.bar(x, vals, bar_w, bottom=bottoms, color=PHENO_COLORS[cat],
           edgecolor='white', linewidth=1.0, label=SHORT[cat], zorder=3)
    for xi, (v, b0) in enumerate(zip(vals, bottoms)):
        if not np.isnan(v) and v >= 0.6:
            txtcol = 'white' if cat == 'RNPH-GC (Possible NPH)' else 'black'
            strokecol = PHENO_COLORS[cat] if txtcol == 'white' else 'white'
            ax.text(xi, b0 + v / 2, f'{v:.1f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=txtcol, zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground=strokecol)])
    bottoms += np.nan_to_num(vals)

allr = np.array([R.get((c, OVERALL), np.nan) for c in COHORT_ORDER])
allo = allr - np.array([LO.get((c, OVERALL), np.nan) for c in COHORT_ORDER])
ahi = np.array([HI.get((c, OVERALL), np.nan) for c in COHORT_ORDER]) - allr
x_off = x + bar_w / 2 + 0.06
ax.errorbar(x_off, allr, yerr=[allo, ahi], fmt='D', color=PHENO_COLORS[OVERALL],
            markersize=10, capsize=6, capthick=2.2, elinewidth=2.5,
            markeredgecolor='white', markeredgewidth=1.6, linestyle='none',
            label='RNPH-All', zorder=7)

ax.set_xticks(x)
ax.set_xticklabels(COHORT_ORDER, fontweight='bold', fontsize=14)
ax.set_ylabel("Incidence per 1,000 Person-Years\n(Age & Sex Standardized)",
              fontweight='bold', fontsize=14)
ax.set_xlim(-0.6, len(COHORT_ORDER) - 1 + 0.7)
ax.set_ylim(bottom=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_axisbelow(True)
ax.tick_params(axis='y', labelsize=12)

# legend at the bottom, same format as Figure 1
handles, labels = ax.get_legend_handles_labels()
order = ['RNPH-G', 'RNPH-C', 'RNPH-GC', 'RNPH-All']
hd = dict(zip(labels, handles))
handles = [hd[k] for k in order if k in hd]
labels = [k for k in order if k in hd]
fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
           fontsize=12, bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("Fig_Incidence_Stacked.tiff", dpi=300, format='tiff', bbox_inches='tight')
print("done")
