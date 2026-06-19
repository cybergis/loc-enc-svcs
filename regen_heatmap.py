"""Single-panel fig_main_heatmap.pdf: Pearson r only, BL/EC/EO.

Fixes the BL column to be the TRUE coords-only baseline (the `none` encoder's
value, constant per scale) instead of the per-encoder dim-4 'baseline' runs that
aggregate_metrics mislabels. Significance stars are recomputed as
"significantly greater than coords-only" via paired Wilcoxon over the 25 reps.
Amplitude (OLS slope) panel dropped.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

ROOT, OUT = './results', './plots'
ENCODER_ORDER = [
    'Sphere2Vec-dfs', 'Sphere2Vec-sphereM+', 'wrap_ffn',
    'Sphere2Vec-sphereM', 'Sphere2Vec-sphereC', 'Sphere2Vec-sphereC+',
    'Space2Vec-grid', 'Space2Vec-theory', 'NeRF', 'rff', 'tile_ffn',
]
SCALE_LABELS = {'global': 'Global', 'county': 'County', 'grid': 'Grid'}
SCALES = ['global', 'county', 'grid']
EFF = 'SVC_X2_Smooth'

reps_df = pd.read_csv(os.path.join(ROOT, 'combined_all_reps.csv'))
reps_df['encoder_trained'] = reps_df['encoder_trained'].astype(str)
reps_df = reps_df[(reps_df['dgp'] == 'simple') & (reps_df['spatial_effect'] == EFF)
                  & (reps_df['encoder_trained'] == 'False')]


def rep_series(scale, fc, enc):
    m = ((reps_df['scale'] == scale) & (reps_df['feature_config'] == fc)
         & (reps_df['encoder'] == enc))
    s = reps_df.loc[m, ['repetition', 'pearson_r']].dropna()
    return s.sort_values('repetition')['pearson_r'].to_numpy()


# coords-only reference = none @ emb+coords, per scale
bl_ref = {s: rep_series(s, 'emb+coords', 'none') for s in SCALES}

conditions = [('BL', None), ('EC', 'emb+coords'), ('EO', 'emb_only')]
n_cond, n_enc = len(conditions), len(ENCODER_ORDER)
n_cols = len(SCALES) * n_cond
col_labels = [f'{SCALE_LABELS[s]}\n{short}' for s in SCALES for short, _ in conditions]

matrix = np.full((n_enc, n_cols), np.nan)
annot = np.full((n_enc, n_cols), '', dtype=object)

for i, enc in enumerate(ENCODER_ORDER):
    for j, scale in enumerate(SCALES):
        ref = bl_ref[scale]
        for k, (short, fc) in enumerate(conditions):
            col = j * n_cond + k
            if short == 'BL':
                vals = ref                      # coords-only, constant across rows
            else:
                vals = rep_series(scale, fc, enc)
            if len(vals) == 0:
                annot[i, col] = '—'
                continue
            mean = float(np.mean(vals))
            matrix[i, col] = mean
            star = ''
            if short != 'BL' and len(vals) == len(ref) and len(ref) > 0:
                diff = vals - ref
                if np.any(diff != 0):
                    try:
                        _, p = wilcoxon(vals, ref, alternative='greater')
                        if p < 0.05:
                            star = '*'
                    except ValueError:
                        pass
            annot[i, col] = f'{mean:.2f}{star}'

fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.8))
plt.subplots_adjust(left=0.22, right=0.99, top=0.90, bottom=0.16)
df_m = pd.DataFrame(matrix, index=ENCODER_ORDER, columns=col_labels)
sns.heatmap(df_m, ax=ax, cmap='RdYlGn', vmin=0.3, vmax=1.0,
            annot=annot, fmt='', annot_kws={'size': 7.5},
            linewidths=0.4, linecolor='white',
            cbar_kws={'shrink': 0.7, 'pad': 0.02})
for sep in [n_cond, 2 * n_cond]:
    ax.axvline(sep, color='white', linewidth=3)
ax.axhline(3, color='white', linewidth=3)
ax.set_yticklabels(ax.get_yticklabels(), color='black', fontsize=8.5)
ax.set_xlabel(''); ax.set_ylabel('')
ax.tick_params(axis='x', labelsize=8)
ax.set_title(r'$\beta_2$ SVC Recovery (Pattern, Pearson $r$)',
             fontsize=11, fontweight='bold', pad=10)
fig.text(0.5, 0.035,
         'BL = Baseline (coords only)   EC = Embedding + Coordinates   '
         'EO = Embedding Only   * significantly > coords-only baseline (p<0.05)',
         ha='center', fontsize=7, style='italic', color='#555555')

out = os.path.join(OUT, 'fig_main_heatmap.pdf')
fig.savefig(out)
plt.close(fig)
print(f'Saved: {out}')
print('BL (coords-only) per scale:', {s: round(float(np.mean(v)), 3) for s, v in bl_ref.items()})
