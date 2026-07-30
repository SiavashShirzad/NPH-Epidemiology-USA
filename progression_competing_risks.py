"""Figure 3: cumulative incidence of symptomatic conversion (Aalen-Johansen, death as competing risk)."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lifelines import AalenJohansenFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.grid'] = False


THEME_COLORS = {
    'RNPH': '#1F3A5F',                    # deep slate (overall reference, high contrast)
                'RNPH-G': '#4C82AF',                  # steel blue (isolated gait)
                'RNPH-C': '#E3A24A',                  # amber (isolated cognitive)
                'RNPH-GC': '#A6323B'
}

# Map the columns/scenarios to their specific outcome color
TARGET_COLORS = [
    THEME_COLORS['RNPH-G'],   # Col 0: Conversion to RNPH-G
    THEME_COLORS['RNPH-C'],   # Col 1: Conversion to RNPH-C
    THEME_COLORS['RNPH-GC'],  # Col 2: Conversion to RNPH-GC
    THEME_COLORS['RNPH-GC']   # Col 3: Conversion from G to GC
]
CONTROL_COLOR = '#888888'     # Neutral grey for the control group

EVANS_THRESH = 0.3; CA_ACUTE_THRESH = 90; SPPB_ABN_THRESH = 10

def finalize(o):
    o['rnph'] = ((o['ei'] > EVANS_THRESH) | (o['ca'] > CA_ACUTE_THRESH)).fillna(False)
    o['control_visit'] = (o['ei'] < EVANS_THRESH) & o['ei'].notna()
    o['arm_eligible'] = o['rnph'] | o['control_visit']
    o['comb_abn'] = o['gait_abn'] & o['cog_abn']
    o['comb_avail'] = o['gait_avail'] & o['cog_avail']
    return o[o['age'] >= 60].sort_values(['pid', 'age'])

def load_blsa():
    B = pd.read_csv("BLSA_epidemiology.csv", low_memory=False)
    B = B.loc[:, ~B.columns.duplicated()].groupby(['pid', 'visitdate'], as_index=False).first()
    bd = pd.to_datetime(B['birthdate'], errors='coerce'); vd = pd.to_datetime(B['visitdate'], errors='coerce')
    dod = pd.to_datetime(B.get('dateofdeath'), errors='coerce')
    o = pd.DataFrame({'pid': 'BLSA_' + B['pid'].astype(str), 'age': (vd - bd).dt.days / 365.25,
                      'death_age': (dod - bd).dt.days / 365.25, 'is_dead': dod.notna(),
                      'sex_male': np.where(B['sex'].astype(str).str.upper().str.startswith('M'), 1, 0),
                      'ei': pd.to_numeric(B['EI'], errors='coerce'), 'ca': pd.to_numeric(B['CA'], errors='coerce')})
    g = pd.to_numeric(B['sppb'], errors='coerce').where(lambda x: (x >= 0) & (x <= 12))
    c = pd.to_numeric(B['cogn_status'], errors='coerce').replace({3: 0})
    o['gait_avail'] = g.notna().values; o['gait_abn'] = ((g <= SPPB_ABN_THRESH) & g.notna()).values
    o['cog_avail'] = c.notna().values; o['cog_abn'] = (c.isin([1, 2]) & c.notna()).values
    return finalize(o)

def load_adni():
    A = pd.read_csv("ADNI_epidemiology.csv", low_memory=False)
    A = A[A['Sex'] != 'X'].copy()
    _by = pd.to_datetime(A['PTDOBYY'], errors='coerce').dt.year
    _dy = pd.to_numeric(A['NPDODYR'], errors='coerce')
    _sub = pd.DataFrame({'Subject': A['Subject'].values, 'by': _by.values, 'dy': _dy.values}).groupby('Subject').first()
    sub_death = _sub['dy'] - _sub['by']
    A = A.groupby(['Subject', 'Visit'], as_index=False).first()
    sex = A['Sex'].astype(str).str.upper().str[0]
    sex = sex.where(sex.isin(['M', 'F']), A['PTGENDER'].astype(str).str.upper().str[0])
    death_age = A['Subject'].map(sub_death)
    o = pd.DataFrame({'pid': 'ADNI_' + A['Subject'].astype(str), 'age': pd.to_numeric(A['Age'], errors='coerce'),
                      'death_age': death_age.values, 'is_dead': death_age.notna().values, 'sex_male': (sex == 'M').astype(int),
                      'ei': pd.to_numeric(A['EI'], errors='coerce'), 'ca': pd.to_numeric(A['CA'], errors='coerce')})
    g = pd.to_numeric(A['NXGAIT'], errors='coerce'); c = pd.to_numeric(A['CDGLOBAL'], errors='coerce')
    o['gait_avail'] = g.isin([1, 2]).values; o['gait_abn'] = (g == 2).values
    o['cog_avail'] = c.notna().values; o['cog_abn'] = (c >= 0.5).fillna(False).values
    return finalize(o)

PROC = {'BLSA': load_blsa(), 'ADNI': load_adni()}

def build(df, outcome, avail, base_filter=None):
    rows = []
    for pid, g in df.groupby('pid'):
        g = g.sort_values('age')
        cand = g[g['arm_eligible']]
        if base_filter is not None:
            cand = cand[base_filter(g).reindex(cand.index).fillna(False)]
        if cand.empty:
            continue
        b = cand.iloc[0]; base_age = b['age']
        arm = 'Target' if b['rnph'] else 'Control'
        if bool(b[outcome]):
            continue
        fut = g[(g['age'] > base_age) & (g[avail])]
        ev = fut[fut[outcome] == True]
        if not ev.empty:
            dur, st = ev['age'].min() - base_age, 1
        elif b['is_dead'] and pd.notna(b['death_age']) and b['death_age'] > base_age:
            dur, st = b['death_age'] - base_age, 2
        elif not fut.empty:
            dur, st = fut['age'].max() - base_age, 0
        else:
            continue
        if dur > 0:
            rows.append({'pid': pid, 'arm': arm, 'duration': dur, 'event': st,
                         't0_age': base_age, 'sex_male': int(b['sex_male'])})
    return pd.DataFrame(rows)

def cif_at(ajf, t):
    d = ajf.cumulative_density_; b = ajf.confidence_interval_
    if d.empty or t > d.index.max(): return 'N/A'
    est = d.asof(t).iloc[0] * 100; lo = b.iloc[:, 0].asof(t) * 100; hi = b.iloc[:, 1].asof(t) * 100
    return f"{est:.1f}%" if (pd.isna(lo) or pd.isna(hi)) else f"{est:.1f}% ({lo:.1f}-{hi:.1f})"

# Panel titles templated as "Conversion from X to Y"
SCENARIOS = [
    ('A. Conversion from\nRNPH to RNPH-G', 'gait_abn', 'gait_avail', None),
    ('B. Conversion from\nRNPH to RNPH-C', 'cog_abn', 'cog_avail', None),
    ('C. Conversion from\nRNPH to RNPH-GC', 'comb_abn', 'comb_avail', None),
    ('D. Conversion from\nRNPH-G to RNPH-GC', 'cog_abn', 'cog_avail',
     lambda g: g['gait_abn'] & ~g['cog_abn'] & g['gait_avail'] & g['cog_avail']),
]

results_table = []
fig, axes = plt.subplots(2, 4, figsize=(22, 12), sharey=False, sharex=False)
row_axes = {0: [], 1: []}

for row, cohort in enumerate(['BLSA', 'ADNI']):
    for col, (title, outcome, avail, bf) in enumerate(SCENARIOS):
        # Assign the target color based on the column's specific outcome
        tcolor = TARGET_COLORS[col]
        ax = axes[row, col]
        if row == 1 and col == 3:
            ax.set_visible(False)
            continue
        drawn_axes = row_axes[row]
        drawn_axes.append(ax)

        d = build(PROC[cohort], outcome, avail, bf)
        tgt, ctl = d[d['arm'] == 'Target'], d[d['arm'] == 'Control']

        hr_txt, stars, p_str = "n/a", "", "n/a"
        if len(tgt) >= 10 and (tgt['event'] == 1).sum() >= 5 and (ctl['event'] == 1).sum() >= 5:
            m = d.copy(); m['rnph'] = (m['arm'] == 'Target').astype(int); m['cs'] = (m['event'] == 1).astype(int)
            try:
                cph = CoxPHFitter().fit(m[['duration', 'cs', 'rnph', 't0_age', 'sex_male']], 'duration', 'cs')
                r = cph.summary.loc['rnph']; p = r['p']
                hr_txt = f"aHR {np.exp(r['coef']):.2f} ({np.exp(r['coef lower 95%']):.2f}-{np.exp(r['coef upper 95%']):.2f})"
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
            except Exception:
                pass

        lr_str = "n/a"
        if len(tgt) >= 10 and (tgt['event'] == 1).sum() >= 3 and (ctl['event'] == 1).sum() >= 3:
            lr = logrank_test(tgt['duration'], ctl['duration'],
                              event_observed_A=(tgt['event'] == 1).astype(int),
                              event_observed_B=(ctl['event'] == 1).astype(int))
            lr_str = "p<0.001" if lr.p_value < 0.001 else f"p={lr.p_value:.3f}"

        for arm, color, ls in [('Control', CONTROL_COLOR, '--'), ('Target', tcolor, '-')]:
            g = d[d['arm'] == arm]
            n_tot = len(g); n_ev = int((g['event'] == 1).sum()); n_dead = int((g['event'] == 2).sum())
            ci3 = ci5 = ci10 = 'N/A'
            if n_tot >= 10 and n_ev > 0:
                ajf = AalenJohansenFitter(calculate_variance=True)
                ajf.fit(g['duration'], g['event'], event_of_interest=1, label=arm)
                ajf.plot(ax=ax, color=color, linestyle=ls, linewidth=3.0, ci_alpha=0.15)
                ci3, ci5, ci10 = cif_at(ajf, 3), cif_at(ajf, 5), cif_at(ajf, 10)

            results_table.append({'Cohort': cohort, 'Scenario': title.replace('\n', ' '), 'Arm': arm,
                                  'N': n_tot, 'Events': n_ev, 'Deaths': n_dead,
                                  'Baseline age': round(g['t0_age'].mean(), 1) if n_tot else np.nan,
                                  'CIF 3y': ci3, 'CIF 5y': ci5, 'CIF 10y': ci10,
                                  'aHR age+sex': hr_txt if arm == 'Target' else '-',
                                  'aHR p': f"{p_str} {stars}" if arm == 'Target' else '-',
                                  'Log-rank p': lr_str if arm == 'Target' else '-'})

        ax.text(0.04, 0.96, f"{hr_txt}\n{stars}", transform=ax.transAxes, ha='left', va='top',
                fontsize=13, fontweight='bold', color='#222')

        if row == 0:
            ax.set_title(title, fontweight='bold', fontsize=16, pad=15)
        else:
            ax.set_title(title, fontweight='bold', fontsize=16, pad=15)

        ax.set_xlabel("Years", fontweight='bold', fontsize=14)
        ax.tick_params(labelbottom=True, labelleft=True, labelsize=13)

        if col == 0:
            ax.set_ylabel(f"{cohort}\nCumulative Incidence", fontweight='bold', fontsize=15)

        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        if ax.get_legend(): ax.get_legend().remove()

        lines = [Line2D([0], [0], color=tcolor, lw=3.0, ls='-'), Line2D([0], [0], color=CONTROL_COLOR, lw=3.0, ls='--')]
        labels = (['RNPH', 'Not RNPH'] if col < 3 else ['RNPH-G', 'Not RNPH + isolated gait'])
        ax.legend(lines, labels, frameon=False, loc='lower right', fontsize=12)

# common y-scale within each cohort row, with tick numbers shown on all panels
for r, axs in row_axes.items():
    if not axs:
        continue
    ymax = max(a.get_ylim()[1] for a in axs)
    for a in axs:
        a.set_ylim(0, ymax)
        a.tick_params(axis='y', labelleft=True, labelsize=13)

plt.tight_layout()
plt.savefig('Fig_RNPH_Progression_BLSA_ADNI.tiff', dpi=300, bbox_inches='tight')

TAB = pd.DataFrame(results_table)
TAB.to_csv("Progression_Summary_BLSA_ADNI.csv", index=False)
print("row ymax:", {r: round(max(a.get_ylim()[1] for a in axs),3) for r,axs in row_axes.items() if axs})
print("done")
