from prevalence_phenotypes import CATS
from data import *

BANDS = [(60, 70, '60-69'), (70, 80, '70-79'), (80, 200, '≥80')]
rng = np.random.default_rng(2026)


def band_table(cohort, rad, classifiable):
    s = SCANS60[(SCANS60['cohort'] == cohort) & SCANS60[classifiable]]
    out = []
    for lo, hi, band in BANDS:
        b = s[(s['age'] >= lo) & (s['age'] < hi)]
        c = CLIN[(CLIN['cohort'] == cohort) & (CLIN['age'] >= lo) & (CLIN['age'] < hi) & CLIN['pid'].isin(b['pid'])]
        symptoms = c.groupby('pid').agg(gait=('gait_abn', lambda x: x.astype(float).max()), cog=('cog_abn', lambda x: x.astype(float).max()))
        g = b.groupby('pid')[rad].any().to_frame('rad').join(symptoms, how='left')
        g['complete'] = g['gait'].notna() & g['cog'].notna()
        g['RNPH-All'] = g['rad']
        for cat, (gv, cv) in {'RNPH-A': (0, 0), 'RNPH-G': (1, 0), 'RNPH-C': (0, 1), 'RNPH-GC': (1, 1)}.items():
            g[cat] = g['rad'] & g['complete'] & (g['gait'] == gv) & (g['cog'] == cv)
        out.append(g.assign(band=band).reset_index())
    return pd.concat(out, ignore_index=True)


def project(t):
    totals = {}
    for cat in CATS:
        count = 0.0
        for band, pop in US_AGE_2024.items():
            x = t[t['band'] == band]
            denom = len(x) if cat == 'RNPH-All' else int(x['complete'].sum())
            if denom:
                count += pop * x[cat].sum() / denom
        totals[cat] = count
    return totals


def projections(rad, classifiable):
    rows = []
    for cohort in COHORTS:
        t = band_table(cohort, rad, classifiable)
        point = project(t)
        ids = t['pid'].unique()
        by_pid = {p: g for p, g in t.groupby('pid')}
        boots = pd.DataFrame([project(pd.concat([by_pid[p] for p in rng.choice(ids, len(ids))], ignore_index=True)) for _ in range(N_BOOTSTRAP)])
        for cat in CATS:
            lo, hi = np.percentile(boots[cat], [2.5, 97.5])
            rows.append({'cohort': cohort, 'category': cat, 'millions': point[cat] / 1e6, 'lo': lo / 1e6, 'hi': hi / 1e6})
    res = pd.DataFrame(rows)
    res['cell'] = res.apply(lambda r: f"{r['millions']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})", axis=1)
    t = res.pivot(index='category', columns='cohort', values='cell').reindex(index=CATS, columns=COHORTS)
    return t.assign(**{'Range across cohorts': res.groupby('category')['millions'].agg(lambda m: f'{m.min():.1f}-{m.max():.1f}')})


save_table(pd.concat({'Callosal angle': projections('rad_ca', 'classifiable_ca'), 'DESH': projections('rad_desh', 'classifiable_desh')},
                     names=['Definition']), 'eTable10_us_projection')
