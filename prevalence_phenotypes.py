from data import *

AGE_BANDS = [(60, 70, '60-69'), (70, 80, '70-79'), (80, 200, '≥80'), (60, 200, '≥60')]
CATS = ['RNPH-All', 'RNPH-A', 'RNPH-G', 'RNPH-C', 'RNPH-GC']
PRESENTATIONS = {'RNPH-All': 'NPH imaging features, any', 'RNPH-A': 'No impairment', 'RNPH-G': 'Gait impairment only',
                 'RNPH-C': 'Cognitive impairment only', 'RNPH-GC': 'Gait and cognitive impairment'}


def prevalence(scans, rad, classifiable):
    rows = []
    for cohort in COHORTS:
        s = scans[(scans['cohort'] == cohort) & scans[classifiable]]
        clin = CLIN[CLIN['cohort'] == cohort]
        for lo, hi, band in AGE_BANDS:
            b = s[(s['age'] >= lo) & (s['age'] < hi)]
            rad_n, rad_k = b['pid'].nunique(), b.loc[b[rad], 'pid'].nunique()
            c = clin[(clin['age'] >= lo) & (clin['age'] < hi) & clin['pid'].isin(b['pid'])]
            symptoms = c.groupby('pid').agg(g=('gait_abn', lambda x: x.astype(float).max()), c=('cog_abn', lambda x: x.astype(float).max()))
            agg = b.groupby('pid')[rad].any().to_frame('rad').join(symptoms, how='inner').dropna(subset=['g', 'c'])
            p = agg[agg['rad']]
            counts = {'RNPH-A': ((p.g == 0) & (p.c == 0)).sum(), 'RNPH-G': ((p.g == 1) & (p.c == 0)).sum(),
                      'RNPH-C': ((p.g == 0) & (p.c == 1)).sum(), 'RNPH-GC': ((p.g == 1) & (p.c == 1)).sum()}
            for cat in CATS:
                k, n = (rad_k, rad_n) if cat == 'RNPH-All' else (int(counts[cat]), len(agg))
                p_, lo_ci, hi_ci = wilson(k, n)
                rows.append({'cohort': cohort, 'band': band, 'category': cat, 'k': k, 'n': n, 'prev': p_, 'lo': lo_ci, 'hi': hi_ci})
    return pd.DataFrame(rows)


def prevalence_table(results):
    cell = lambda x: '' if x['n'] == 0 else f"{100 * x['prev']:.1f} ({100 * x['lo']:.1f}-{100 * x['hi']:.1f})"
    rows = []
    for c in COHORTS:
        for definition, res in results.items():
            r = res.set_index(['cohort', 'band', 'category'])
            for cat, presentation in PRESENTATIONS.items():
                rows.append({'Cohort': c, 'Definition': definition, 'Clinical presentation': presentation,
                             **{f'{band} y': cell(r.loc[(c, band, cat)]) for band in [b[2] for b in AGE_BANDS]}})
    return pd.DataFrame(rows).set_index(['Cohort', 'Definition', 'Clinical presentation'])


STACK = ['RNPH-A', 'RNPH-G', 'RNPH-C', 'RNPH-GC']
FIG_BANDS = ['60-69', '70-79', '≥80']
BAR_X, BAR_W, DIAMOND_X = -0.04, 0.42, 0.34
SEGMENT_FONT, SIDE_FONT, VALUE_FONT = 7.0, 6.8, 7.4
SIDE_COLORS = {'RNPH-A': '#6b767c', 'RNPH-G': '#3b6a91', 'RNPH-C': '#a8691a', 'RNPH-GC': '#8e2a32'}


def label_positions(targets, min_gap, floor):
    placed = []
    for y in sorted(targets):
        y = max(y, floor)
        placed.append(max(y, placed[-1] + min_gap) if placed else y)
    return placed


def plot_prevalence(res, name):
    r = res.set_index(['cohort', 'band', 'category'])
    top = max(100 * r.loc[(c, b, 'RNPH-All'), 'hi'] for c in COHORTS for b in FIG_BANDS)
    top = float(5 * np.ceil(top * 1.12 / 5))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.95, bottom=0.1, hspace=0.32, wspace=0.12)
    units_per_pt = top / (axes[0, 0].get_position().height * fig.get_figheight() * 72)
    inside_min = (SEGMENT_FONT + 3.0) * units_per_pt
    gap = (SIDE_FONT + 2.5) * units_per_pt
    x = np.arange(len(FIG_BANDS))
    for i, (ax, cohort) in enumerate(zip(axes.ravel(), COHORTS)):
        bottom = np.zeros(len(FIG_BANDS))
        outside = {xi: [] for xi in range(len(FIG_BANDS))}
        for cat in STACK:
            vals = np.array([100 * r.loc[(cohort, b, cat), 'prev'] for b in FIG_BANDS])
            ax.bar(x + BAR_X, vals, BAR_W, bottom=bottom, color=COLORS[cat], edgecolor='white', lw=0.8,
                   label=LABELS[cat] if i == 0 else None, zorder=2)
            for xi, (v, b0) in enumerate(zip(vals, bottom)):
                if v <= 0:
                    continue
                if v >= inside_min:
                    ax.text(xi + BAR_X, b0 + v / 2, f'{v:.1f}', ha='center', va='center', fontsize=SEGMENT_FONT,
                            color='white' if cat in ('RNPH-G', 'RNPH-GC') else INK, zorder=4)
                else:
                    outside[xi].append((b0 + v / 2, v, cat))
            bottom += vals
        for xi, items in outside.items():
            items.sort()
            for (y, v, cat), ly in zip(items, label_positions([y for y, _, _ in items], gap, floor=gap * 0.6)):
                ax.text(xi + BAR_X - BAR_W / 2 - 0.05, ly, f'{v:.1f}', ha='right', va='center', fontsize=SIDE_FONT,
                        color=SIDE_COLORS[cat], fontweight='bold', zorder=4)
        overall = np.array([[100 * r.loc[(cohort, b, 'RNPH-All'), k] for k in ('prev', 'lo', 'hi')] for b in FIG_BANDS])
        ax.errorbar(x + DIAMOND_X, overall[:, 0], yerr=[overall[:, 0] - overall[:, 1], overall[:, 2] - overall[:, 0]], fmt='D',
                    ms=4.8, color=COLORS['RNPH-All'], mec='white', mew=0.6, capsize=0, elinewidth=1.0, zorder=3,
                    label='All NPH imaging features (95% CI)' if i == 0 else None)
        for xi, (p, lo, hi) in enumerate(overall):
            ax.text(xi + DIAMOND_X, hi + 2.5 * units_per_pt, f'{p:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT,
                    color=COLORS['RNPH-All'], fontweight='bold', zorder=4)
        ax.set_ylim(0, top)
        ax.set_xlim(-0.72, len(FIG_BANDS) - 0.42)
        ax.set_xticks(x + 0.05)
        ax.set_xticklabels([f'{b} y' for b in FIG_BANDS], fontsize=8)
        ax.tick_params(axis='x', length=0, pad=4)
        ax.tick_params(axis='y', labelsize=8, length=3, width=0.7)
        ax.set_yticks(np.arange(0, top + 0.1, 5))
        ax.yaxis.grid(True, color='#e6e5df', lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f'{chr(65 + i)}  {cohort}', loc='left', fontsize=9.5, fontweight='bold', pad=6)
        if i % 2 == 0:
            ax.set_ylabel('Period prevalence, %', fontsize=8.5, labelpad=6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, fontsize=7.6, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.3, handleheight=1.0, columnspacing=1.3)
    save_figure(fig, name)


PREV_CA = prevalence(SCANS60, 'rad_ca', 'classifiable_ca')
PREV_DESH = prevalence(SCANS60, 'rad_desh', 'classifiable_desh')

if __name__ == '__main__':
    save_table(prevalence_table({'CA': PREV_CA, 'DESH': PREV_DESH}), 'eTable4_prevalence')
    plot_prevalence(PREV_CA, 'Figure1_prevalence_CA')
    plot_prevalence(PREV_DESH, 'Figure2_prevalence_DESH')
