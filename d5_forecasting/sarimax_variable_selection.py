# =============================================================================
# SARIMAX: Variable Selection Scenario
# Dataset  : statsmodels Macrodata (US quarterly macroeconomic data)
# Target   : realgdp
# Scenario : Start with all candidate X variables → stepwise rejection
#            → final model with selected variables
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from itertools import product
import scipy.stats as stats

# =============================================================================
# 1. DATA LOADING & PREPARATION
# =============================================================================
print("=" * 65)
print("1. DATA LOADING & PREPARATION")
print("=" * 65)

macro = sm.datasets.macrodata.load_pandas().data

quarters = pd.period_range(start='1959Q1', periods=len(macro), freq='Q')
macro.index = quarters.to_timestamp()
macro.index.freq = 'QS'

target = macro['realgdp']

# All candidate exogenous variables (excluding year, quarter, realgdp)
all_candidates = ['realcons', 'realinv', 'realgovt', 'realdpi',
                  'cpi', 'm1', 'tbilrate', 'unemp', 'pop', 'infl', 'realint']

print(f"Date range     : {macro.index[0].date()} ~ {macro.index[-1].date()}")
print(f"Observations   : {len(macro)}")
print(f"Target (Y)     : realgdp")
print(f"Candidates (X) : {all_candidates}")

# =============================================================================
# 2. CANDIDATE VARIABLE OVERVIEW
# =============================================================================
print("\n" + "=" * 65)
print("2. CANDIDATE VARIABLE OVERVIEW")
print("=" * 65)

desc = macro[all_candidates + ['realgdp']].describe().T[['mean','std','min','max']]
print(desc.round(2).to_string())

# =============================================================================
# 3. CORRELATION WITH TARGET
# =============================================================================
print("\n" + "=" * 65)
print("3. CORRELATION WITH TARGET (realgdp)")
print("=" * 65)

corr_with_target = macro[all_candidates].corrwith(target).sort_values(ascending=False)
print("\n  Variable       Correlation   Verdict")
print("  " + "-" * 45)
for var, corr in corr_with_target.items():
    verdict = "Strong ✓" if abs(corr) > 0.7 else ("Moderate" if abs(corr) > 0.4 else "Weak ✗")
    print(f"  {var:<14} {corr:>8.4f}     {verdict}")

# Correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

corr_matrix = macro[all_candidates + ['realgdp']].corr()
im = axes[0].imshow(corr_matrix.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
axes[0].set_xticks(range(len(corr_matrix.columns)))
axes[0].set_yticks(range(len(corr_matrix.columns)))
axes[0].set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
axes[0].set_yticklabels(corr_matrix.columns, fontsize=8)
axes[0].set_title('Correlation Heatmap (All Variables)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=axes[0])
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        axes[0].text(j, i, f'{corr_matrix.values[i,j]:.2f}',
                     ha='center', va='center', fontsize=6,
                     color='black' if abs(corr_matrix.values[i,j]) < 0.7 else 'white')

# Correlation with target bar
colors = ['#2196F3' if c > 0 else '#F44336' for c in corr_with_target.values]
axes[1].barh(corr_with_target.index, corr_with_target.values, color=colors, edgecolor='white')
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].axvline(0.7, color='green', linewidth=1.5, linestyle='--', label='|r|=0.7 threshold')
axes[1].axvline(-0.7, color='green', linewidth=1.5, linestyle='--')
axes[1].set_title('Correlation with realgdp', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Pearson Correlation', fontsize=10)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('01_correlation.png', dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: 01_correlation.png")

# =============================================================================
# 4. REJECTION STEP 1 — LOW CORRELATION
# =============================================================================
print("\n" + "=" * 65)
print("4. REJECTION STEP 1 — Low Correlation with Target (|r| < 0.4)")
print("=" * 65)

threshold_corr = 0.4
rejected_corr = corr_with_target[abs(corr_with_target) < threshold_corr].index.tolist()
after_corr    = [v for v in all_candidates if v not in rejected_corr]

print(f"\n  Rejected (|r| < {threshold_corr}): {rejected_corr}")
print(f"  Remaining              : {after_corr}")

# =============================================================================
# 5. REJECTION STEP 2 — MULTICOLLINEARITY (VIF)
# =============================================================================
print("\n" + "=" * 65)
print("5. REJECTION STEP 2 — Multicollinearity (VIF > 10)")
print("=" * 65)

def compute_vif(df):
    vif_data = pd.DataFrame()
    vif_data['Variable'] = df.columns
    vif_data['VIF']      = [variance_inflation_factor(df.values, i)
                             for i in range(df.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

X_after_corr = macro[after_corr].dropna()
vif_df = compute_vif(X_after_corr)

print("\n  VIF Results:")
print("  Variable       VIF        Verdict")
print("  " + "-" * 40)
rejected_vif = []
for _, row in vif_df.iterrows():
    verdict = "REJECT (multicollinear) ✗" if row['VIF'] > 10 else "Keep ✓"
    print(f"  {row['Variable']:<14} {row['VIF']:>8.2f}   {verdict}")
    if row['VIF'] > 10:
        rejected_vif.append(row['Variable'])

# Iterative VIF removal (remove highest VIF one at a time)
remaining = after_corr.copy()
print("\n  Iterative VIF removal:")
iteration = 1
while True:
    X_tmp  = macro[remaining].dropna()
    vif_tmp = compute_vif(X_tmp)
    max_vif = vif_tmp['VIF'].max()
    max_var = vif_tmp.loc[vif_tmp['VIF'].idxmax(), 'Variable']
    if max_vif > 10:
        print(f"  Iter {iteration}: Remove '{max_var}' (VIF={max_vif:.2f})")
        remaining.remove(max_var)
        iteration += 1
    else:
        print(f"  Iter {iteration}: All VIF <= 10. Stop.")
        break

after_vif = remaining
print(f"\n  After VIF filtering: {after_vif}")

# VIF bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

vif_df_plot = compute_vif(macro[after_corr].dropna())
colors_vif = ['#F44336' if v > 10 else '#4CAF50' for v in vif_df_plot['VIF']]
axes[0].barh(vif_df_plot['Variable'], vif_df_plot['VIF'], color=colors_vif, edgecolor='white')
axes[0].axvline(10, color='red', linewidth=2, linestyle='--', label='VIF=10 threshold')
axes[0].set_title('VIF — Before Removal', fontsize=12, fontweight='bold')
axes[0].set_xlabel('VIF', fontsize=10)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, axis='x')

vif_df_after = compute_vif(macro[after_vif].dropna())
colors_after = ['#4CAF50' for _ in vif_df_after['VIF']]
axes[1].barh(vif_df_after['Variable'], vif_df_after['VIF'], color=colors_after, edgecolor='white')
axes[1].axvline(10, color='red', linewidth=2, linestyle='--', label='VIF=10 threshold')
axes[1].set_title('VIF — After Removal', fontsize=12, fontweight='bold')
axes[1].set_xlabel('VIF', fontsize=10)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('02_vif.png', dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: 02_vif.png")

# =============================================================================
# 6. REJECTION STEP 3 — STATIONARITY (ADF Test on X variables)
# =============================================================================
print("\n" + "=" * 65)
print("6. REJECTION STEP 3 — ADF Test on X Variables (level)")
print("=" * 65)

print("\n  Variable       ADF Stat    p-value    Stationary?   Action")
print("  " + "-" * 62)

rejected_adf = []
after_adf    = []
for var in after_vif:
    result = adfuller(macro[var].dropna(), autolag='AIC')
    stat, pval = result[0], result[1]
    is_stat = pval < 0.05
    # Check if 1st difference is stationary
    diff_result = adfuller(macro[var].diff().dropna(), autolag='AIC')
    diff_stat   = diff_result[1] < 0.05
    if is_stat:
        action = "Use level ✓"
        after_adf.append(var)
    elif diff_stat:
        action = "Use d=1 ✓"
        after_adf.append(var)
    else:
        action = "REJECT ✗"
        rejected_adf.append(var)
    print(f"  {var:<14} {stat:>8.4f}  {pval:>8.4f}   {'Yes' if is_stat else 'No':<12}  {action}")

print(f"\n  After ADF check : {after_adf}")
if rejected_adf:
    print(f"  Rejected        : {rejected_adf}")

# Final selected variables
final_vars = after_adf
print(f"\n  >>> Final selected X variables: {final_vars}")

# =============================================================================
# 7. VARIABLE SELECTION SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("7. VARIABLE SELECTION SUMMARY")
print("=" * 65)

all_vars_info = []
for var in all_candidates:
    corr_val = corr_with_target[var]
    step1 = "REJECT" if var in rejected_corr else "Pass"
    step2 = "REJECT" if var not in after_vif and var not in rejected_corr else ("Pass" if var in after_vif else "—")
    step3 = "REJECT" if var in rejected_adf else ("Pass" if var in final_vars else "—")
    final = "SELECTED ✓" if var in final_vars else "Rejected ✗"
    all_vars_info.append({
        'Variable': var,
        'Corr': f'{corr_val:.3f}',
        'Step1(Corr)': step1,
        'Step2(VIF)': step2,
        'Step3(ADF)': step3,
        'Final': final
    })

summary_df = pd.DataFrame(all_vars_info)
print(summary_df.to_string(index=False))

# Funnel visualization
fig, ax = plt.subplots(figsize=(12, 6))

stages = [
    f'All Candidates\n({len(all_candidates)} vars)',
    f'After Correlation\n({len(after_corr)} vars)',
    f'After VIF\n({len(after_vif)} vars)',
    f'Final Selected\n({len(final_vars)} vars)'
]
counts = [len(all_candidates), len(after_corr), len(after_vif), len(final_vars)]
colors_funnel = ['#1565C0', '#1976D2', '#42A5F5', '#90CAF9']

for i, (stage, count, color) in enumerate(zip(stages, counts, colors_funnel)):
    width = 0.6 + (count / len(all_candidates)) * 0.4
    rect = plt.Rectangle((i - width/2, 0), width, 0.8,
                          color=color, alpha=0.85, zorder=3)
    ax.add_patch(rect)
    ax.text(i, 0.4, stage, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=4)
    if i < len(stages) - 1:
        rejected_n = counts[i] - counts[i+1]
        ax.annotate(f'−{rejected_n} rejected',
                    xy=(i + 0.5, 0.4), fontsize=9, color='#F44336',
                    ha='center', va='bottom',
                    xytext=(i + 0.5, 0.95),
                    arrowprops=dict(arrowstyle='->', color='#F44336'))

ax.set_xlim(-0.7, len(stages) - 0.3)
ax.set_ylim(-0.2, 1.3)
ax.set_xticks(range(len(stages)))
ax.set_xticklabels([f'Step {i}' for i in range(len(stages))], fontsize=10)
ax.set_yticks([])
ax.set_title('Variable Selection Funnel', fontsize=13, fontweight='bold')
ax.spines[['top','right','left']].set_visible(False)
plt.tight_layout()
plt.savefig('03_selection_funnel.png', dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: 03_selection_funnel.png")

# =============================================================================
# 8. TRAIN / TEST SPLIT
# =============================================================================
print("\n" + "=" * 65)
print("8. TRAIN / TEST SPLIT")
print("=" * 65)

n_test  = 12
exog    = macro[final_vars]
train_y = target.iloc[:-n_test]
test_y  = target.iloc[-n_test:]
train_X = exog.iloc[:-n_test]
test_X  = exog.iloc[-n_test:]

print(f"  Train: {train_y.index[0].date()} ~ {train_y.index[-1].date()}  ({len(train_y)} obs)")
print(f"  Test : {test_y.index[0].date()} ~ {test_y.index[-1].date()}   ({len(test_y)} obs)")
print(f"  Final exog variables: {final_vars}")

# =============================================================================
# 9. GRID SEARCH — Best (p,d,q)(P,D,Q,s)
# =============================================================================
print("\n" + "=" * 65)
print("9. GRID SEARCH (AIC-based)")
print("=" * 65)

p_range = range(0, 3)
d_range = [1]
q_range = range(0, 3)
P_range = range(0, 2)
D_range = [0, 1]
Q_range = range(0, 2)
s       = 4

best_aic    = np.inf
best_order  = None
best_sorder = None
results_grid = []

combos = list(product(p_range, d_range, q_range, P_range, D_range, Q_range))
print(f"  Testing {len(combos)} combinations...", end='', flush=True)

for p, d, q, P, D, Q in combos:
    try:
        model = SARIMAX(
            train_y, exog=train_X,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        results_grid.append({
            'order': (p, d, q), 'seasonal_order': (P, D, Q, s),
            'aic': res.aic, 'bic': res.bic
        })
        if res.aic < best_aic:
            best_aic    = res.aic
            best_order  = (p, d, q)
            best_sorder = (P, D, Q, s)
    except Exception:
        pass

print(" Done.")
print(f"\n  Best order          : ARIMA{best_order}")
print(f"  Best seasonal order : SARIMA{best_sorder}")
print(f"  Best AIC            : {best_aic:.2f}")
print(f"\n  Top 5 by AIC:")
print(pd.DataFrame(results_grid).sort_values('aic').head(5).to_string(index=False))

# =============================================================================
# 10. FINAL MODEL FITTING
# =============================================================================
print("\n" + "=" * 65)
print("10. FINAL MODEL FITTING")
print("=" * 65)

final_model = SARIMAX(
    train_y, exog=train_X,
    order=best_order,
    seasonal_order=best_sorder,
    enforce_stationarity=False,
    enforce_invertibility=False
)
final_result = final_model.fit(disp=False)
print(final_result.summary())

# =============================================================================
# 11. EXOGENOUS VARIABLE SIGNIFICANCE (p-value check)
# =============================================================================
print("\n" + "=" * 65)
print("11. EXOGENOUS VARIABLE SIGNIFICANCE (p-value)")
print("=" * 65)

print(f"\n  Variable       Coef        Std Err    p-value    Significant?")
print("  " + "-" * 60)
coef_rows = []
for var in final_vars:
    if var in final_result.params.index:
        coef  = final_result.params[var]
        se    = final_result.bse[var]
        pval  = final_result.pvalues[var]
        sig   = pval < 0.05
        stars = '***' if pval<0.001 else ('**' if pval<0.01 else ('*' if pval<0.05 else 'n.s.'))
        print(f"  {var:<14} {coef:>10.4f}  {se:>9.4f}  {pval:>9.4f}   {'Yes '+stars if sig else 'No  '+stars}")
        coef_rows.append({'var': var, 'coef': coef, 'pval': pval, 'sig': sig, 'stars': stars})

# Coefficient plot
fig, ax = plt.subplots(figsize=(9, max(4, len(coef_rows)*0.7 + 1)))
vars_   = [r['var'] for r in coef_rows]
coefs_  = [r['coef'] for r in coef_rows]
pvals_  = [r['pval'] for r in coef_rows]
colors_ = ['#2196F3' if r['sig'] else '#BDBDBD' for r in coef_rows]

bars = ax.barh(vars_, coefs_, color=colors_, edgecolor='white', height=0.5)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Exogenous Variable Coefficients\n(Blue=Significant p<0.05, Gray=Not significant)',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Coefficient Value', fontsize=10)
for bar, row in zip(bars, coef_rows):
    x_pos = row['coef'] + (max(coefs_) * 0.02 if row['coef'] >= 0 else min(coefs_) * 0.02)
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f"{row['coef']:.4f} {row['stars']}",
            va='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('04_coefficients.png', dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: 04_coefficients.png")

# =============================================================================
# 12. RESIDUAL DIAGNOSTICS
# =============================================================================
print("\n" + "=" * 65)
print("12. RESIDUAL DIAGNOSTICS")
print("=" * 65)

residuals = final_result.resid
lb_test   = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(f"  Ljung-Box p-value (lag=10): {lb_test['lb_pvalue'].values[0]:.4f}  "
      f"{'✓ White noise' if lb_test['lb_pvalue'].values[0] > 0.05 else '✗ Autocorrelation remains'}")

fig = final_result.plot_diagnostics(figsize=(14, 8))
fig.suptitle(f'Residual Diagnostics — SARIMAX{best_order}x{best_sorder}',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('05_residuals.png', dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: 05_residuals.png")

# =============================================================================
# 13. FORECAST & EVALUATION
# =============================================================================
print("\n" + "=" * 65)
print("13. FORECAST & EVALUATION")
print("=" * 65)

forecast      = final_result.get_forecast(steps=n_test, exog=test_X)
forecast_mean = forecast.predicted_mean
forecast_ci   = forecast.conf_int(alpha=0.05)

mae  = np.mean(np.abs(forecast_mean - test_y))
rmse = np.sqrt(np.mean((forecast_mean - test_y)**2))
mape = np.mean(np.abs((forecast_mean - test_y) / test_y)) * 100

print(f"  MAE  : {mae:,.2f}")
print(f"  RMSE : {rmse:,.2f}")
print(f"  MAPE : {mape:.2f}%")

forecast_df = pd.DataFrame({
    'Actual'   : test_y.values,
    'Forecast' : forecast_mean.values,
    'Lower_95' : forecast_ci.iloc[:, 0].values,
    'Upper_95' : forecast_ci.iloc[:, 1].values,
    'Error'    : forecast_mean.values - test_y.values
}, index=test_y.index)
print(f"\n  Forecast vs Actual:")
print(forecast_df.round(2).to_string())

# =============================================================================
# 14. FORECAST VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f'SARIMAX{best_order}x{best_sorder} — Real GDP Forecast\n'
             f'Final X vars: {final_vars}',
             fontsize=12, fontweight='bold')

ax = axes[0]
ax.plot(train_y.index, train_y, label='Train', color='steelblue', linewidth=1.5)
ax.plot(test_y.index, test_y, label='Actual (Test)', color='black', linewidth=2, linestyle='--')
ax.plot(forecast_mean.index, forecast_mean, label='Forecast', color='tomato', linewidth=2)
ax.fill_between(forecast_ci.index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1],
                alpha=0.2, color='tomato', label='95% CI')
ax.axvline(test_y.index[0], color='gray', linestyle=':', linewidth=1.5)
ax.set_title('Full Period: Train + Test Forecast', fontsize=11)
ax.set_ylabel('Real GDP (Billion USD)', fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(test_y.index, test_y, label='Actual', color='black',
         linewidth=2.5, marker='o', markersize=5)
ax2.plot(forecast_mean.index, forecast_mean, label='Forecast', color='tomato',
         linewidth=2.5, marker='s', markersize=5)
ax2.fill_between(forecast_ci.index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1],
                 alpha=0.25, color='tomato', label='95% CI')
for idx, row in forecast_df.iterrows():
    ax2.annotate(f'{row["Error"]:+,.0f}',
                 xy=(idx, row['Forecast']),
                 xytext=(0, 12), textcoords='offset points',
                 ha='center', fontsize=7.5, color='tomato')
ax2.set_title(f'Test Period Zoom  |  MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:.2f}%',
              fontsize=11)
ax2.set_ylabel('Real GDP (Billion USD)', fontsize=10)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('06_forecast.png', dpi=120, bbox_inches='tight')
plt.show()
print("  Saved: 06_forecast.png")

# =============================================================================
# 15. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("15. FINAL SUMMARY")
print("=" * 65)
print(f"\n  [Variable Selection]")
print(f"  All candidates  : {len(all_candidates)} vars  {all_candidates}")
print(f"  After corr(|r|<{threshold_corr}): {len(after_corr)} vars  {after_corr}")
print(f"  After VIF(>10)  : {len(after_vif)} vars  {after_vif}")
print(f"  After ADF check : {len(final_vars)} vars  {final_vars}")
print(f"\n  [Model]")
print(f"  SARIMAX{best_order}x{best_sorder}")
print(f"  AIC : {best_aic:.2f}")
print(f"\n  [Forecast Performance]")
print(f"  MAE  : {mae:,.2f}")
print(f"  RMSE : {rmse:,.2f}")
print(f"  MAPE : {mape:.2f}%")
print(f"\n  [Exogenous Coefficients]")
for row in coef_rows:
    sig_str = 'Significant' if row['sig'] else 'Not significant'
    print(f"  {row['var']:<14}: coef={row['coef']:>10.4f}  p={row['pval']:.4f}  {sig_str} {row['stars']}")
print("\nDone.")
