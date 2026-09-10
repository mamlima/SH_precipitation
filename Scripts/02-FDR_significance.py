#%% 0. Init
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import spearmanr


#%% 1. Configuration
IN_PATTERN  = '../Data/Output_data/RidgePrecipImpact_{reg}_{season}.nc'
OUT_PATTERN = '../Data/Output_data/RidgePrecipFDR_{reg}_{season}.nc'

REGIONS = ['SAm', 'SAf', 'Oce']
SEASONS = ['winter', 'summer']

N_BOOT     = 1000     # bootstrap ensemble size used in scripts 01-03
ALPHA_FDR  = 0.10     # 2 x nominal level, as recommended for correlated fields
ALPHA_PW   = 0.05     # pointwise level, for the comparison only

THRESH = {'winter': {'clim': 0.5, 'ext': 3.0},
          'summer': {'clim': 1.5, 'ext': 9.0}}

# Denominator for the reported fraction.
#   'valid' : land points passing the climatological mask
#   'land'  : all land points in the domain
DENOMINATOR = 'valid'


#%% 2. Functions
def two_tailed_p(pval_upper, n=N_BOOT):
    """Two-tailed empirical p from the stored upper-tail proportion.

    pval_upper = fraction of null samples strictly greater than the observation,
    so n_gt = round(pval_upper * n) recovers the count. The (r+1)/(n+1) form
    avoids p = 0, which is not a meaningful value for a finite ensemble.
    """
    pu = np.clip(np.asarray(pval_upper, dtype=float), 0.0, 1.0)
    n_gt = np.rint(pu * n)
    n_le = n - n_gt
    p = 2.0 * np.minimum((n_gt + 1.0) / (n + 1.0), (n_le + 1.0) / (n + 1.0))
    return np.minimum(p, 1.0)


def bh_fdr(p, mask, alpha=ALPHA_FDR):
    """Benjamini-Hochberg over the grid points in `mask`. Returns (rej, p_crit)."""
    sel = mask & np.isfinite(p)
    pv = np.sort(p[sel])
    N = pv.size
    if N == 0:
        return np.zeros_like(sel, dtype=bool), 0.0
    below = np.where(pv <= alpha * np.arange(1, N + 1) / N)[0]
    p_crit = float(pv[below[-1]]) if below.size else 0.0
    return sel & (p <= p_crit), p_crit


#%% 3. Apply FDR
rows = []
for season in SEASONS:
    for reg in REGIONS:
        ds = xr.open_dataset(IN_PATTERN.format(reg=reg, season=season))

        land = ds['precip_clima'].isel(reg=0).notnull().values
        valid = {'clim': land & (ds['precip_clima'].isel(reg=0).values > THRESH[season]['clim']),
                 'ext':  land & (ds['precip_ext'].isel(reg=0).values  > THRESH[season]['ext'])}

        nreg = ds.sizes['reg']
        out = {}
        for metric, pvar in [('clim', 'pval_llb'), ('ext', 'pval_llb_ext')]:
            sig = np.zeros((nreg,) + land.shape, dtype=np.int8)
            pcrit = np.zeros(nreg)
            for k in range(nreg):
                pu = ds[pvar].isel(reg=k).values
                p2 = two_tailed_p(pu)
                rej, pc = bh_fdr(p2, valid[metric])
                # sign: small upper-tail p means the observation sits above the null
                sig[k][rej & (pu < 0.5)] = 1
                sig[k][rej & (pu > 0.5)] = -1
                pcrit[k] = pc

                denom = valid[metric].sum() if DENOMINATOR == 'valid' else land.sum()
                pw = ((pu < ALPHA_PW / 2) | (pu > 1 - ALPHA_PW / 2)) & valid[metric]
                rows.append({'region': reg, 'season': season, 'metric': metric,
                             'sector_idx': k,
                             'sector_lon': float(ds['reg'].values[k]),
                             'pct_pointwise': pw.sum() / denom * 100,
                             'pct_fdr': (sig[k] != 0).sum() / denom * 100,
                             'p_crit': pc})
            out[f'sig_{metric}'] = (['reg', 'lat', 'lon'], sig)
            out[f'pcrit_{metric}'] = (['reg'], pcrit)

        xr.Dataset(data_vars=out,
                   coords=dict(reg=ds['reg'], lat=ds['lat'], lon=ds['lon']),
                   attrs=dict(description=(f'FDR-controlled significance for {reg} {season}; '
                                           f'Benjamini-Hochberg per sector map, alpha_FDR={ALPHA_FDR}; '
                                           '+1 significant positive, -1 significant negative'),
                              alpha_fdr=ALPHA_FDR, n_boot=N_BOOT,
                              denominator=DENOMINATOR)
                   ).to_netcdf(OUT_PATTERN.format(reg=reg, season=season))
        print(f'{reg} {season}: written')

res = pd.DataFrame(rows)
res.to_csv('../Data/Output_data/R2_2_FDR_comparison.csv', index=False)


#%% 4. Numbers for the reply
pd.set_option('display.width', 180)
print('\n=== Pointwise vs FDR, summary per region / season / metric ===')
summary = []
for (reg, season, metric), g in res.groupby(['region', 'season', 'metric']):
    g = g.sort_values('sector_lon')
    rho = spearmanr(g.pct_pointwise, g.pct_fdr).correlation
    summary.append({'region': reg, 'season': season, 'metric': metric,
                    'peak_pointwise': g.pct_pointwise.max(),
                    'peak_fdr': g.pct_fdr.max(),
                    'drop_pp': g.pct_pointwise.max() - g.pct_fdr.max(),
                    'peak_sector_pw': g.loc[g.pct_pointwise.idxmax(), 'sector_lon']+15,
                    'peak_sector_fdr': g.loc[g.pct_fdr.idxmax(), 'sector_lon']+15,
                    'spearman_r': rho,
                    'median_p_crit': g.p_crit.median()})
summary = pd.DataFrame(summary)
print(summary.round(3).to_string(index=False))

print('\n--- Sentences this supports ---')
for _, r in summary.iterrows():
    same = 'unchanged' if r.peak_sector_pw == r.peak_sector_fdr else 'MOVED'
    print(f'{r.region:4s} {r.season:6s} {r.metric:4s}: peak {r.peak_pointwise:5.1f}% -> '
          f'{r.peak_fdr:5.1f}% ({r.drop_pp:4.1f} pp), peak sector {same}, '
          f'rank corr. across sectors {r.spearman_r:.3f}')
print('\nA Spearman correlation near 1 is the quantitative form of "the longitudinal '
      'structure is unchanged" — that is the sentence the reply needs.')
print(f'\nDenominator used: {DENOMINATOR}. Note that the current Fig. 2 script divides '
      'by ALL land points, not by the masked-in points; see the notes file.')