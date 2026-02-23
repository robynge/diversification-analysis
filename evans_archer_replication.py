"""
Evans & Archer (1968) Replication
"Diversification and the Reduction of Dispersion: An Empirical Analysis"
The Journal of Finance, Vol. 23, No. 5 (Dec., 1968), pp. 761-767

This script generates TWO sets of output:
1. With Dividends (Total Return) - output/with_dividends/
2. Price Only (No Dividends)     - output/price_only/

Data Source: CRSP/WRDS (1958-1967, S&P 500 constituents)
Simulation: 1000 runs, portfolio sizes 1-40
Model: Y = B/X + A (rectangular hyperbola)

Original Results (Evans & Archer 1968):
- A (asymptote) = 0.1191
- B (coefficient) = 0.08625
- R² = 0.9863
- Systematic variation = 0.1166
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Publication Style Configuration
# ============================================================================
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "red_strong": "#B64342",
    "green_main": "#2E7D32",
    "green_secondary": "#66BB6A",
}

def apply_publication_style(font_size: int = 16, axes_linewidth: float = 2.5):
    """Apply publication-ready matplotlib style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 2,
        "axes.linewidth": axes_linewidth,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "legend.fontsize": font_size - 2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.usetex": False,
        "mathtext.fontset": "stix",  # High-quality math rendering
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    })


# ============================================================================
# Data Processing - Price Only Returns
# ============================================================================

def compute_price_only_returns(data_dir: Path) -> pd.DataFrame:
    """
    Compute semi-annual log returns using PRICE ONLY (retx, excluding dividends).

    CRSP fields:
    - ret: Total return (includes dividends)
    - retx: Return excluding dividends (price change only)
    """
    # Load monthly raw data
    monthly_data = pd.read_csv(data_dir / 'evans_archer_monthly_raw.csv')

    print(f"Loaded monthly data: {len(monthly_data)} rows")
    print(f"Columns: {list(monthly_data.columns)}")

    # Check for missing retx values
    missing_retx = monthly_data['retx'].isna().sum()
    print(f"Missing retx values: {missing_retx}")

    # Compute semi-annual value relatives using retx (price only)
    def compute_semiannual_value_relative_price_only(group):
        """Compute semi-annual value relative using price returns only (no dividends)."""
        valid_returns = group['retx'].dropna()

        if len(valid_returns) == 0:
            return np.nan

        # Compound returns: (1+r1) * (1+r2) * ... * (1+rn)
        value_relative = np.prod(1 + valid_returns)

        return value_relative

    # Group by permno and semiannual_period
    semiannual_data = monthly_data.groupby(['permno', 'semiannual_period']).apply(
        lambda x: pd.Series({
            'value_relative_price_only': compute_semiannual_value_relative_price_only(x),
            'n_months': x['retx'].notna().sum()
        })
    ).reset_index()

    # Filter for complete data (6 months per semi-annual period)
    semiannual_data = semiannual_data[semiannual_data['n_months'] >= 5]
    semiannual_data = semiannual_data[semiannual_data['value_relative_price_only'] > 0]

    # Compute log returns
    semiannual_data['log_return_price_only'] = np.log(semiannual_data['value_relative_price_only'])

    # Filter for securities with all 19 periods
    periods_per_security = semiannual_data.groupby('permno')['semiannual_period'].count()
    valid_permnos = periods_per_security[periods_per_security >= 19].index.tolist()

    semiannual_final = semiannual_data[semiannual_data['permno'].isin(valid_permnos)]

    print(f"Securities with complete data: {len(valid_permnos)}")
    print(f"Total observations: {len(semiannual_final)}")

    # Create pivot table (periods x securities)
    log_returns_matrix = semiannual_final.pivot(
        index='semiannual_period',
        columns='permno',
        values='log_return_price_only'
    )

    print(f"Log returns matrix shape: {log_returns_matrix.shape}")

    return log_returns_matrix


# ============================================================================
# Portfolio Analysis (Same as original)
# ============================================================================

def compute_portfolio_stats(log_returns: pd.DataFrame, selected_columns: list) -> tuple[float, float]:
    """Compute portfolio geometric mean return and standard deviation."""
    portfolio_log_returns = log_returns[selected_columns].mean(axis=1)
    mean_log_return = portfolio_log_returns.mean()
    geo_mean_return = np.exp(mean_log_return)
    std_dev = portfolio_log_returns.std(ddof=1)
    return geo_mean_return, std_dev


def run_simulation(
    log_returns: pd.DataFrame,
    max_portfolio_size: int = 40,
    n_runs: int = 1000,
    random_seed: int = 42
) -> pd.DataFrame:
    """Run the Evans & Archer simulation."""
    np.random.seed(random_seed)

    all_columns = list(log_returns.columns)
    n_securities = len(all_columns)
    max_portfolio_size = min(max_portfolio_size, n_securities)

    print(f"\nRunning simulation (PRICE ONLY):")
    print(f"  Number of runs: {n_runs}")
    print(f"  Max portfolio size: {max_portfolio_size}")
    print(f"  Available securities: {n_securities}")

    results = []

    for run in range(n_runs):
        if (run + 1) % 100 == 0:
            print(f"  Completed run {run + 1}/{n_runs}")

        shuffled_columns = np.random.permutation(all_columns).tolist()

        for size in range(1, max_portfolio_size + 1):
            selected = shuffled_columns[:size]
            geo_mean, std_dev = compute_portfolio_stats(log_returns, selected)

            results.append({
                'portfolio_size': size,
                'run': run,
                'std_dev': std_dev,
                'mean_return': geo_mean
            })

    print(f"  Total portfolios evaluated: {len(results)}")

    return pd.DataFrame(results)


def fit_hyperbola(results: pd.DataFrame) -> tuple[float, float, float]:
    """Fit the model: Y = B/X + A"""
    mean_std = results.groupby('portfolio_size')['std_dev'].mean()

    X = mean_std.index.values.astype(float)
    Y = mean_std.values

    X_inv = 1.0 / X
    X_design = np.column_stack([np.ones_like(X_inv), X_inv])
    params, _, _, _ = np.linalg.lstsq(X_design, Y, rcond=None)
    A, B = params

    Y_pred = A + B / X
    SS_res = np.sum((Y - Y_pred) ** 2)
    SS_tot = np.sum((Y - Y.mean()) ** 2)
    R_squared = 1 - SS_res / SS_tot

    return A, B, R_squared


def compute_systematic_risk(log_returns: pd.DataFrame) -> float:
    """Compute systematic risk (market portfolio SD)."""
    market_portfolio_returns = log_returns.mean(axis=1)
    return market_portfolio_returns.std(ddof=1)


# ============================================================================
# Statistical Tests (Following Evans & Archer 1968, Journal p.766)
# ============================================================================

def compute_statistical_tests(results: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Compute t-tests and F-tests following Evans & Archer (1968) methodology.

    From the original paper (p.766):
    "(1) t-tests on successive mean portfolio standard deviations, which
    indicated on the average the significance of successive increases in
    portfolio size; and (2) F-tests on successive standard deviations about
    the mean portfolio standard deviation, which tend to indicate convergence
    of the individual observations on the mean values."

    Original findings:
    - Portfolio size 2: adding 1 security → significant reduction
    - Portfolio size 8: need to add 5 securities for significance
    - Portfolio size 16: need to add 19 securities for significance
    - Portfolio size > 19: no significant reduction possible within 40 securities

    Returns:
        Dictionary with test results
    """
    from scipy import stats

    sizes = sorted(results['portfolio_size'].unique())
    max_size = max(sizes)

    test_results = {
        'basic_stats': [],
        'securities_needed_t_test': [],
        'securities_needed_f_test': [],
    }

    # 1. Basic statistics for each portfolio size
    for size in sizes:
        data = results[results['portfolio_size'] == size]['std_dev']
        test_results['basic_stats'].append({
            'portfolio_size': size,
            'mean_std': data.mean(),
            'std_of_std': data.std(),
            'variance_of_std': data.var(),
            'n_obs': len(data)
        })

    # 2. Find minimum securities needed for significant reduction (t-test)
    # This replicates the key finding in the original paper
    for start_size in sizes:
        if start_size >= max_size:
            continue

        start_data = results[results['portfolio_size'] == start_size]['std_dev']
        securities_needed = None

        for end_size in range(start_size + 1, max_size + 1):
            end_data = results[results['portfolio_size'] == end_size]['std_dev']

            # One-tailed t-test: is mean SD at end_size significantly LOWER?
            t_stat, t_pval_two = stats.ttest_ind(start_data, end_data)
            t_pval = t_pval_two / 2 if t_stat > 0 else 1 - t_pval_two / 2

            if t_pval < alpha:
                securities_needed = end_size - start_size
                break

        test_results['securities_needed_t_test'].append({
            'starting_size': start_size,
            'securities_needed': securities_needed if securities_needed else None,
            'no_significance_within_range': securities_needed is None
        })

    # 3. Find minimum securities needed for significant variance reduction (F-test)
    for start_size in sizes:
        if start_size >= max_size:
            continue

        start_data = results[results['portfolio_size'] == start_size]['std_dev']
        start_var = start_data.var()
        securities_needed = None

        for end_size in range(start_size + 1, max_size + 1):
            end_data = results[results['portfolio_size'] == end_size]['std_dev']
            end_var = end_data.var()

            # F-test: is variance at end_size significantly LOWER?
            if end_var > 0:
                f_stat = start_var / end_var
                df1 = len(start_data) - 1
                df2 = len(end_data) - 1
                f_pval = 1 - stats.f.cdf(f_stat, df1, df2)

                if f_pval < alpha:
                    securities_needed = end_size - start_size
                    break

        test_results['securities_needed_f_test'].append({
            'starting_size': start_size,
            'securities_needed': securities_needed if securities_needed else None,
            'no_significance_within_range': securities_needed is None
        })

    return test_results


def print_statistical_tests(test_results: dict, output_dir: Path, label: str = ""):
    """Print and save statistical test results matching Evans & Archer (1968) style."""

    print("\n" + "="*75)
    print(f"STATISTICAL TESTS - {label} (Evans & Archer 1968 Method)")
    print("="*75)

    # Key finding table: Securities needed for significant reduction
    print("\n┌" + "─"*73 + "┐")
    print("│" + " SECURITIES NEEDED FOR SIGNIFICANT REDUCTION IN MEAN SD (t-test, α=0.05)".ljust(73) + "│")
    print("├" + "─"*20 + "┬" + "─"*26 + "┬" + "─"*25 + "┤")
    print("│" + " Starting Size".ljust(20) + "│" + " Securities to Add".center(26) + "│" + " Resulting Size".center(25) + "│")
    print("├" + "─"*20 + "┼" + "─"*26 + "┼" + "─"*25 + "┤")

    key_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 16, 19, 20, 25, 30, 35]
    for row in test_results['securities_needed_t_test']:
        if row['starting_size'] in key_sizes:
            start = row['starting_size']
            if row['securities_needed'] is not None:
                needed = str(row['securities_needed'])
                end = str(start + row['securities_needed'])
            else:
                needed = f">{40 - start}"
                end = "N/A (>40)"
            print(f"│{start:^20}│{needed:^26}│{end:^25}│")

    print("└" + "─"*20 + "┴" + "─"*26 + "┴" + "─"*25 + "┘")

    # F-test results
    print("\n┌" + "─"*73 + "┐")
    print("│" + " SECURITIES NEEDED FOR SIGNIFICANT VARIANCE REDUCTION (F-test, α=0.05)".ljust(73) + "│")
    print("├" + "─"*20 + "┬" + "─"*26 + "┬" + "─"*25 + "┤")
    print("│" + " Starting Size".ljust(20) + "│" + " Securities to Add".center(26) + "│" + " Resulting Size".center(25) + "│")
    print("├" + "─"*20 + "┼" + "─"*26 + "┼" + "─"*25 + "┤")

    for row in test_results['securities_needed_f_test']:
        if row['starting_size'] in key_sizes:
            start = row['starting_size']
            if row['securities_needed'] is not None:
                needed = str(row['securities_needed'])
                end = str(start + row['securities_needed'])
            else:
                needed = f">{40 - start}"
                end = "N/A (>40)"
            print(f"│{start:^20}│{needed:^26}│{end:^25}│")

    print("└" + "─"*20 + "┴" + "─"*26 + "┴" + "─"*25 + "┘")

    # Comparison with original paper
    print("\n" + "─"*75)
    print("COMPARISON WITH ORIGINAL EVANS & ARCHER (1968) FINDINGS:")
    print("─"*75)
    print("Original paper (p.766):")
    print("  • Size 2:  need 1 security  for significant reduction")
    print("  • Size 8:  need 5 securities for significant reduction")
    print("  • Size 16: need 19 securities for significant reduction")
    print("  • Size >19: no significant reduction within 40 securities")

    # Save to CSV
    pd.DataFrame(test_results['basic_stats']).to_csv(
        output_dir / f'stats_basic_{label.lower().replace(" ", "_")}.csv', index=False)
    pd.DataFrame(test_results['securities_needed_t_test']).to_csv(
        output_dir / f'stats_t_test_{label.lower().replace(" ", "_")}.csv', index=False)
    pd.DataFrame(test_results['securities_needed_f_test']).to_csv(
        output_dir / f'stats_f_test_{label.lower().replace(" ", "_")}.csv', index=False)


# ============================================================================
# Visualization - Comparison Plot
# ============================================================================

def plot_figure1_single(
    results: pd.DataFrame,
    params: dict,
    title: str,
    output_path: Path,
    color_main: str,
    color_secondary: str,
    figsize: tuple = (10, 7)
):
    """
    Create Figure 1 style plot for a single dataset.
    Matches the style from evans_archer_replication.ipynb
    """
    apply_publication_style(font_size=14, axes_linewidth=2.2)

    fig, ax = plt.subplots(figsize=figsize)

    # Compute stats
    stats = results.groupby('portfolio_size')['std_dev'].agg(['mean', 'std', 'count']).reset_index()
    X = stats['portfolio_size'].values
    Y = stats['mean'].values
    Ystd = stats['std'].values
    n_runs = stats['count'].values[0]

    # Smooth curve
    Xs = np.linspace(1, X.max(), 200)

    # 95% CI
    ci_u = Y + 1.96 * Ystd / np.sqrt(n_runs)
    ci_l = Y - 1.96 * Ystd / np.sqrt(n_runs)

    # Plot elements (matching notebook style)
    ax.fill_between(X, ci_l, ci_u, color=color_secondary, alpha=0.25)
    ax.plot(Xs, params['B']/Xs + params['A'], color=color_main, linewidth=2.5)
    ax.scatter(X, Y, color=color_main, s=60, zorder=5, edgecolors='white')
    ax.axhline(y=params['sys_risk'], color=PALETTE["red_strong"], linewidth=1.5)

    # Systematic risk label
    ax.text(2, params['sys_risk'] * 1.02,
            f'Systematic variation = {params["sys_risk"]:.4f}',
            fontsize=10)

    # Legend and parameters (matching notebook style exactly)
    xs, ys, lh = 0.52, 0.96, 0.045

    # Line for predicted Y
    ax.plot([xs, xs+0.06], [ys, ys], color=color_main, lw=2.5,
            transform=ax.transAxes, clip_on=False)
    ax.text(xs+0.08, ys, '= predicted Y', fontsize=10,
            transform=ax.transAxes, va='center')

    # Dot for actual Y
    ax.plot(xs+0.03, ys-lh, 'o', color=color_main, ms=8,
            transform=ax.transAxes, clip_on=False)
    ax.text(xs+0.08, ys-lh, '= actual Y', fontsize=10,
            transform=ax.transAxes, va='center')

    # Rectangle for 95% CI
    ax.add_patch(Rectangle((xs, ys-2*lh-0.015), 0.06, 0.03,
                           facecolor=color_secondary, alpha=0.4,
                           transform=ax.transAxes, clip_on=False))
    ax.text(xs+0.08, ys-2*lh, '= 95% CI', fontsize=10,
            transform=ax.transAxes, va='center')

    # Parameters (using LaTeX-style math)
    ax.text(xs, ys-3.5*lh, r'$Y = B/X + A$', fontsize=10,
            transform=ax.transAxes, va='top')
    ax.text(xs, ys-4.5*lh, f'A={params["A"]:.4f}, B={params["B"]:.5f}', fontsize=10,
            transform=ax.transAxes, va='top')
    ax.text(xs, ys-5.5*lh, f'$R^2$={params["R2"]:.4f}', fontsize=10,
            transform=ax.transAxes, va='top')

    # Add return information if available
    if 'semi_annual_ret' in params and 'annual_ret' in params:
        ax.text(xs, ys-7*lh, f'Mean Return (1 security):', fontsize=10,
                transform=ax.transAxes, va='top')
        ax.text(xs, ys-8*lh, f'  Semi-annual: {params["semi_annual_ret"]:.2f}%', fontsize=10,
                transform=ax.transAxes, va='top')
        ax.text(xs, ys-9*lh, f'  Annualized:  {params["annual_ret"]:.2f}%', fontsize=10,
                transform=ax.transAxes, va='top')

    ax.set_xlabel('Portfolio size')
    ax.set_ylabel('Mean portfolio std dev')
    ax.set_xlim(0, X.max() + 2)
    ax.set_ylim(min(params['sys_risk'], Y.min()) * 0.95, Y.max() * 1.08)

    fig.tight_layout(pad=2)
    fig.savefig(output_path.with_suffix('.png'), dpi=600, facecolor='white')
    plt.close(fig)

    print(f"Saved: {output_path.with_suffix('.png')}")


def plot_comparison(
    results_with_div: pd.DataFrame,
    results_price_only: pd.DataFrame,
    params_with_div: dict,
    params_price_only: dict,
    output_path: Path,
    figsize: tuple = (14, 8)
):
    """
    Create comparison plot: With Dividends vs Price Only
    """
    apply_publication_style(font_size=13, axes_linewidth=2.0)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ===== Left plot: With Dividends (Original) =====
    ax1 = axes[0]
    stats1 = results_with_div.groupby('portfolio_size')['std_dev'].agg(['mean', 'std']).reset_index()
    X1 = stats1['portfolio_size'].values
    Y1 = stats1['mean'].values

    X_smooth = np.linspace(1, 40, 200)
    Y_pred1 = params_with_div['B'] / X_smooth + params_with_div['A']

    n_runs = len(results_with_div[results_with_div['portfolio_size']==1])
    ax1.fill_between(X1, Y1 - 1.96*stats1['std']/np.sqrt(n_runs),
                     Y1 + 1.96*stats1['std']/np.sqrt(n_runs),
                     color=PALETTE["blue_secondary"], alpha=0.25)
    ax1.plot(X_smooth, Y_pred1, color=PALETTE["blue_main"], linewidth=2.5)
    ax1.scatter(X1, Y1, color=PALETTE["blue_main"], s=40, zorder=5, edgecolors='white', linewidth=0.5)
    ax1.axhline(y=params_with_div['sys_risk'], color=PALETTE["red_strong"], linewidth=1.5)

    ax1.set_xlabel('Portfolio size')
    ax1.set_ylabel('Mean portfolio std dev')
    ax1.set_title('With Dividends (Total Return)', fontweight='bold', fontsize=11)
    ax1.set_xlim(0, 42)
    ax1.set_ylim(min(params_with_div['sys_risk'], Y1.min())*0.95, Y1.max()*1.08)

    # Systematic risk label
    ax1.text(2, params_with_div['sys_risk'] * 1.02,
             f'Systematic = {params_with_div["sys_risk"]:.4f}', fontsize=9)

    # Legend (matching notebook style)
    xs, ys, lh = 0.52, 0.96, 0.05
    ax1.plot([xs, xs+0.06], [ys, ys], color=PALETTE["blue_main"], lw=2.5,
             transform=ax1.transAxes, clip_on=False)
    ax1.text(xs+0.08, ys, '= predicted Y', fontsize=9, transform=ax1.transAxes, va='center')
    ax1.plot(xs+0.03, ys-lh, 'o', color=PALETTE["blue_main"], ms=6,
             transform=ax1.transAxes, clip_on=False)
    ax1.text(xs+0.08, ys-lh, '= actual Y', fontsize=9, transform=ax1.transAxes, va='center')
    ax1.add_patch(Rectangle((xs, ys-2*lh-0.015), 0.06, 0.03,
                            facecolor=PALETTE["blue_secondary"], alpha=0.4,
                            transform=ax1.transAxes, clip_on=False))
    ax1.text(xs+0.08, ys-2*lh, '= 95% CI', fontsize=9, transform=ax1.transAxes, va='center')

    # Parameters
    ax1.text(xs, ys-3.5*lh, r'$Y = B/X + A$', fontsize=9, transform=ax1.transAxes, va='top')
    ax1.text(xs, ys-4.5*lh, f'A={params_with_div["A"]:.4f}, B={params_with_div["B"]:.5f}',
             fontsize=9, transform=ax1.transAxes, va='top')
    ax1.text(xs, ys-5.5*lh, f'$R^2$={params_with_div["R2"]:.4f}',
             fontsize=9, transform=ax1.transAxes, va='top')

    # ===== Right plot: Price Only =====
    ax2 = axes[1]
    stats2 = results_price_only.groupby('portfolio_size')['std_dev'].agg(['mean', 'std']).reset_index()
    X2 = stats2['portfolio_size'].values
    Y2 = stats2['mean'].values

    Y_pred2 = params_price_only['B'] / X_smooth + params_price_only['A']

    ax2.fill_between(X2, Y2 - 1.96*stats2['std']/np.sqrt(n_runs),
                     Y2 + 1.96*stats2['std']/np.sqrt(n_runs),
                     color=PALETTE["green_secondary"], alpha=0.25)
    ax2.plot(X_smooth, Y_pred2, color=PALETTE["green_main"], linewidth=2.5)
    ax2.scatter(X2, Y2, color=PALETTE["green_main"], s=40, zorder=5, edgecolors='white', linewidth=0.5)
    ax2.axhline(y=params_price_only['sys_risk'], color=PALETTE["red_strong"], linewidth=1.5)

    ax2.set_xlabel('Portfolio size')
    ax2.set_ylabel('Mean portfolio std dev')
    ax2.set_title('Price Only (No Dividends)', fontweight='bold', fontsize=11)
    ax2.set_xlim(0, 42)
    ax2.set_ylim(min(params_price_only['sys_risk'], Y2.min())*0.95, Y2.max()*1.08)

    # Systematic risk label
    ax2.text(2, params_price_only['sys_risk'] * 1.02,
             f'Systematic = {params_price_only["sys_risk"]:.4f}', fontsize=9)

    # Legend (matching notebook style)
    xs, ys, lh = 0.52, 0.96, 0.05
    ax2.plot([xs, xs+0.06], [ys, ys], color=PALETTE["green_main"], lw=2.5,
             transform=ax2.transAxes, clip_on=False)
    ax2.text(xs+0.08, ys, '= predicted Y', fontsize=9, transform=ax2.transAxes, va='center')
    ax2.plot(xs+0.03, ys-lh, 'o', color=PALETTE["green_main"], ms=6,
             transform=ax2.transAxes, clip_on=False)
    ax2.text(xs+0.08, ys-lh, '= actual Y', fontsize=9, transform=ax2.transAxes, va='center')
    ax2.add_patch(Rectangle((xs, ys-2*lh-0.015), 0.06, 0.03,
                            facecolor=PALETTE["green_secondary"], alpha=0.4,
                            transform=ax2.transAxes, clip_on=False))
    ax2.text(xs+0.08, ys-2*lh, '= 95% CI', fontsize=9, transform=ax2.transAxes, va='center')

    # Parameters
    ax2.text(xs, ys-3.5*lh, r'$Y = B/X + A$', fontsize=9, transform=ax2.transAxes, va='top')
    ax2.text(xs, ys-4.5*lh, f'A={params_price_only["A"]:.4f}, B={params_price_only["B"]:.5f}',
             fontsize=9, transform=ax2.transAxes, va='top')
    ax2.text(xs, ys-5.5*lh, f'$R^2$={params_price_only["R2"]:.4f}',
             fontsize=9, transform=ax2.transAxes, va='top')

    fig.suptitle('Evans & Archer (1968) Replication: Effect of Dividends',
                 fontsize=14, fontweight='bold', y=1.02)

    fig.tight_layout()
    fig.savefig(output_path.with_suffix('.png'), dpi=600, facecolor='white', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path.with_suffix('.png')}")


def plot_return_by_portfolio_size(
    results: pd.DataFrame,
    params: dict,
    title: str,
    output_path: Path,
    color_main: str,
    color_secondary: str,
    figsize: tuple = (10, 7)
):
    """
    Plot return vs portfolio size showing ALL simulation points (scatter plot).
    Each portfolio size shows all 1000 Monte Carlo simulation results.
    """
    apply_publication_style(font_size=14, axes_linewidth=2.2)
    fig, ax = plt.subplots(figsize=figsize)

    # Convert geometric mean return to semi-annual percentage
    results_copy = results.copy()
    results_copy['return_pct'] = (results_copy['mean_return'] - 1) * 100

    # Get all data points for scatter plot
    X_all = results_copy['portfolio_size'].values
    Y_all = results_copy['return_pct'].values

    # Scatter plot with small points and transparency
    ax.scatter(X_all, Y_all, c=color_main, s=8, alpha=0.15, edgecolors='none')

    # Compute stats for reference
    stats = results_copy.groupby('portfolio_size')['return_pct'].agg(['mean', 'min', 'max']).reset_index()
    X = stats['portfolio_size'].values
    Y_mean = stats['mean'].values
    Y_min = stats['min'].values
    Y_max = stats['max'].values

    # Find max mean return
    max_idx = np.argmax(Y_mean)
    max_size = X[max_idx]
    max_return = Y_mean[max_idx]

    # Legend (right upper corner)
    n_runs = len(results_copy[results_copy['portfolio_size'] == 1])
    legend_text = (
        f'N = {n_runs} simulations\n'
        f'Max mean: {max_return:.2f}% @ {max_size} stocks\n'
        f'Mean @ 1: {Y_mean[0]:.2f}%\n'
        f'Mean @ {X[-1]}: {Y_mean[-1]:.2f}%'
    )
    if 'annual_ret' in params:
        legend_text += f'\nAnnualized: {params["annual_ret"]:.2f}%'

    ax.text(0.98, 0.98, legend_text, transform=ax.transAxes, fontsize=10,
            ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Portfolio size (number of securities)')
    ax.set_ylabel('Return (%, semi-annual)')
    ax.set_title(title, fontweight='bold', loc='left')
    ax.set_xlim(0, X.max() + 2)
    y_range = Y_max.max() - Y_min.min()
    ax.set_ylim(Y_min.min() - y_range * 0.05, Y_max.max() + y_range * 0.05)

    fig.tight_layout(pad=2)
    fig.savefig(output_path.with_suffix('.png'), dpi=600, facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path.with_suffix('.png')}")


def plot_return_comparison(
    results_with_div: pd.DataFrame,
    results_price_only: pd.DataFrame,
    params_with_div: dict,
    params_price_only: dict,
    output_path: Path,
    figsize: tuple = (14, 8)
):
    """
    Create comparison plot of returns: With Dividends vs Price Only
    Shows ALL simulation points as scatter plot.
    """
    apply_publication_style(font_size=13, axes_linewidth=2.0)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for idx, (results, params, color, color_sec, title) in enumerate([
        (results_with_div, params_with_div, PALETTE["blue_main"], PALETTE["blue_secondary"], "With Dividends"),
        (results_price_only, params_price_only, PALETTE["green_main"], PALETTE["green_secondary"], "Price Only")
    ]):
        ax = axes[idx]
        results_copy = results.copy()
        results_copy['return_pct'] = (results_copy['mean_return'] - 1) * 100

        # Get all data points for scatter plot
        X_all = results_copy['portfolio_size'].values
        Y_all = results_copy['return_pct'].values

        # Scatter plot with small points and transparency
        ax.scatter(X_all, Y_all, c=color, s=6, alpha=0.12, edgecolors='none')

        # Compute stats for reference
        stats = results_copy.groupby('portfolio_size')['return_pct'].agg(['mean', 'min', 'max']).reset_index()
        X = stats['portfolio_size'].values
        Y_mean = stats['mean'].values
        Y_min = stats['min'].values
        Y_max = stats['max'].values

        # Find max mean return
        max_idx = np.argmax(Y_mean)
        max_size = X[max_idx]
        max_return = Y_mean[max_idx]

        ax.set_xlabel('Portfolio size')
        ax.set_ylabel('Return (%, semi-annual)')
        ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
        ax.set_xlim(0, X.max() + 2)
        y_range = Y_max.max() - Y_min.min()
        ax.set_ylim(Y_min.min() - y_range * 0.05, Y_max.max() + y_range * 0.05)

        # Legend (right upper corner)
        n_runs = len(results_copy[results_copy['portfolio_size'] == 1])
        legend_text = f'N={n_runs}\nMax: {max_return:.2f}%\nAnnual: {params["annual_ret"]:.2f}%'
        ax.text(0.98, 0.98, legend_text, transform=ax.transAxes, fontsize=9,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Return by Portfolio Size - Effect of Dividends',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout(pad=2)
    fig.savefig(output_path.with_suffix('.png'), dpi=600, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path.with_suffix('.png')}")


def plot_volatility_distribution(
    results: pd.DataFrame,
    params: dict,
    title: str,
    output_path: Path,
    color_main: str,
    color_secondary: str,
    figsize: tuple = (10, 7)
):
    """Plot volatility distribution by portfolio size (annualized)."""
    apply_publication_style(font_size=14, axes_linewidth=2.2)
    fig, ax = plt.subplots(figsize=figsize)

    # Semi-annual std dev * sqrt(2) for annualization
    ann_factor = np.sqrt(2)

    stats = results.groupby('portfolio_size')['std_dev'].agg(
        ['mean', 'std', 'min', 'max', 'count']
    ).reset_index()

    X = stats['portfolio_size'].values
    Y = stats['mean'].values * ann_factor * 100  # Annualized, percentage
    Ystd = stats['std'].values * ann_factor * 100
    n_runs = stats['count'].iloc[0]
    ci_u = Y + 1.96 * Ystd / np.sqrt(n_runs)
    ci_l = Y - 1.96 * Ystd / np.sqrt(n_runs)
    Y_min = stats['min'].values * ann_factor * 100
    Y_max = stats['max'].values * ann_factor * 100

    ax.fill_between(X, Y_min, Y_max, color=color_secondary, alpha=0.15)
    ax.fill_between(X, ci_l, ci_u, color=color_secondary, alpha=0.35)
    ax.plot(X, Y, color=color_main, linewidth=2.5, marker='o', markersize=6,
            markerfacecolor='white', markeredgecolor=color_main, markeredgewidth=1.5)

    # Systematic risk line (annualized)
    sys_risk_pct = params['sys_risk'] * ann_factor * 100
    ax.axhline(y=sys_risk_pct, color=PALETTE["red_strong"], linewidth=1.5)

    # Legend (right upper corner)
    xs, ys, lh = 0.70, 0.96, 0.045
    ax.plot([xs, xs+0.06], [ys, ys], color=color_main, lw=2.5,
            transform=ax.transAxes, clip_on=False)
    ax.text(xs+0.08, ys, 'mean volatility', fontsize=10,
            transform=ax.transAxes, va='center')
    ax.add_patch(Rectangle((xs, ys-lh-0.015), 0.06, 0.03,
                           facecolor=color_secondary, alpha=0.4,
                           transform=ax.transAxes, clip_on=False))
    ax.text(xs+0.08, ys-lh, '95% CI', fontsize=10,
            transform=ax.transAxes, va='center')
    ax.add_patch(Rectangle((xs, ys-2*lh-0.015), 0.06, 0.03,
                           facecolor=color_secondary, alpha=0.15,
                           transform=ax.transAxes, clip_on=False))
    ax.text(xs+0.08, ys-2*lh, 'range (min-max)', fontsize=10,
            transform=ax.transAxes, va='center')

    # Stats (right lower corner)
    xs2, ys2, lh2 = 0.98, 0.02, 0.045
    ax.text(xs2, ys2, f'Systematic variation = {sys_risk_pct:.2f}%', fontsize=10,
            transform=ax.transAxes, va='bottom', ha='right')
    ax.text(xs2, ys2+lh2, f'Vol @ {X[-1]} stocks: {Y[-1]:.2f}%', fontsize=10,
            transform=ax.transAxes, va='bottom', ha='right')
    ax.text(xs2, ys2+2*lh2, f'Vol @ 1 stock: {Y[0]:.2f}%', fontsize=10,
            transform=ax.transAxes, va='bottom', ha='right')

    ax.set_xlabel('Portfolio size (number of securities)')
    ax.set_ylabel('Annualized Volatility (%)')
    ax.set_title(title, fontweight='bold', loc='left')
    ax.set_xlim(0, X.max() + 2)
    y_range = Y_max.max() - Y_min.min()
    ax.set_ylim(max(0, Y_min.min() - y_range * 0.1), Y_max.max() + y_range * 0.1)

    fig.tight_layout(pad=2)
    fig.savefig(output_path.with_suffix('.png'), dpi=600, facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path.with_suffix('.png')}")


def plot_volatility_comparison(
    results_with_div: pd.DataFrame,
    results_price_only: pd.DataFrame,
    params_with_div: dict,
    params_price_only: dict,
    output_path: Path,
    figsize: tuple = (14, 8)
):
    """Create comparison plot of volatility distribution: With Dividends vs Price Only."""
    apply_publication_style(font_size=13, axes_linewidth=2.0)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ann_factor = np.sqrt(2)

    for idx, (results, params, color, color_sec, title) in enumerate([
        (results_with_div, params_with_div, PALETTE["blue_main"], PALETTE["blue_secondary"], "With Dividends"),
        (results_price_only, params_price_only, PALETTE["green_main"], PALETTE["green_secondary"], "Price Only")
    ]):
        ax = axes[idx]

        stats = results.groupby('portfolio_size')['std_dev'].agg(
            ['mean', 'std', 'min', 'max', 'count']
        ).reset_index()

        X = stats['portfolio_size'].values
        Y = stats['mean'].values * ann_factor * 100
        Ystd = stats['std'].values * ann_factor * 100
        n_runs = stats['count'].iloc[0]
        ci_u = Y + 1.96 * Ystd / np.sqrt(n_runs)
        ci_l = Y - 1.96 * Ystd / np.sqrt(n_runs)
        Y_min = stats['min'].values * ann_factor * 100
        Y_max = stats['max'].values * ann_factor * 100

        ax.fill_between(X, Y_min, Y_max, color=color_sec, alpha=0.15)
        ax.fill_between(X, ci_l, ci_u, color=color_sec, alpha=0.35)
        ax.plot(X, Y, color=color, linewidth=2, marker='o', markersize=4,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=1)

        # Systematic risk line
        sys_risk_pct = params['sys_risk'] * ann_factor * 100
        ax.axhline(y=sys_risk_pct, color=PALETTE["red_strong"], linewidth=1.2)

        ax.set_xlabel('Portfolio size')
        ax.set_ylabel('Annualized Volatility (%)')
        ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
        ax.set_xlim(0, X.max() + 2)
        y_range = Y_max.max() - Y_min.min()
        ax.set_ylim(max(0, Y_min.min() - y_range * 0.1), Y_max.max() + y_range * 0.1)

        # Stats
        ax.text(0.98, 0.02, f'Sys = {sys_risk_pct:.1f}%', fontsize=9,
                transform=ax.transAxes, va='bottom', ha='right')

    fig.suptitle('Volatility Distribution - Effect of Dividends',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout(pad=2)
    fig.savefig(output_path.with_suffix('.png'), dpi=600, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path.with_suffix('.png')}")


def plot_overlay(
    results_with_div: pd.DataFrame,
    results_price_only: pd.DataFrame,
    params_with_div: dict,
    params_price_only: dict,
    output_path: Path,
    figsize: tuple = (12, 8)
):
    """
    Create overlay plot showing both curves on the same axes.
    """
    apply_publication_style(font_size=14, axes_linewidth=2.2)

    fig, ax = plt.subplots(figsize=figsize)

    # Compute stats
    stats1 = results_with_div.groupby('portfolio_size')['std_dev'].mean()
    stats2 = results_price_only.groupby('portfolio_size')['std_dev'].mean()

    X = stats1.index.values
    X_smooth = np.linspace(1, 40, 200)

    # Fitted curves
    Y_pred1 = params_with_div['B'] / X_smooth + params_with_div['A']
    Y_pred2 = params_price_only['B'] / X_smooth + params_price_only['A']

    # Plot with dividends
    ax.plot(X_smooth, Y_pred1, color=PALETTE["blue_main"], linewidth=2.5,
            label='With Dividends (Total Return)')
    ax.scatter(X, stats1.values, color=PALETTE["blue_main"], s=40, zorder=5,
               edgecolors='white', linewidth=0.5, alpha=0.7)

    # Plot price only
    ax.plot(X_smooth, Y_pred2, color=PALETTE["green_main"], linewidth=2.5,
            label='Price Only (No Dividends)')
    ax.scatter(X, stats2.values, color=PALETTE["green_main"], s=40, zorder=5,
               edgecolors='white', linewidth=0.5, alpha=0.7)

    # Systematic risk lines
    ax.axhline(y=params_with_div['sys_risk'], color=PALETTE["blue_main"],
               linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axhline(y=params_price_only['sys_risk'], color=PALETTE["green_main"],
               linestyle='--', linewidth=1.2, alpha=0.7)

    ax.set_xlabel('Portfolio size')
    ax.set_ylabel('Mean portfolio standard deviation')
    ax.set_title('Effect of Dividends on Diversification Analysis', fontweight='bold')
    ax.set_xlim(0, 42)

    ax.legend(loc='upper right', fontsize=11)

    # Add difference annotation
    diff_A = params_price_only['A'] - params_with_div['A']
    diff_pct = diff_A / params_with_div['A'] * 100

    ax.text(0.02, 0.02,
            f"Asymptote difference: {diff_A:.4f} ({diff_pct:+.1f}%)",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_path.with_suffix('.png'), dpi=600, facecolor='white')
    plt.close(fig)

    print(f"Saved: {output_path.with_suffix('.png')}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""

    print("="*70)
    print("EVANS & ARCHER (1968) REPLICATION - PRICE ONLY vs WITH DIVIDENDS")
    print("="*70)

    # Setup paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "wrds"

    # Create separate output directories
    output_dir_with_div = base_dir / "output" / "with_dividends"
    output_dir_price_only = base_dir / "output" / "price_only"
    output_dir_comparison = base_dir / "output" / "comparison"

    output_dir_with_div.mkdir(parents=True, exist_ok=True)
    output_dir_price_only.mkdir(parents=True, exist_ok=True)
    output_dir_comparison.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directories:")
    print(f"  With dividends:  {output_dir_with_div}")
    print(f"  Price only:      {output_dir_price_only}")
    print(f"  Comparison:      {output_dir_comparison}")

    # ========== LOAD ORIGINAL DATA (WITH DIVIDENDS) ==========
    print("\n" + "-"*70)
    print("PART 1: Loading Original Data (With Dividends)")
    print("-"*70)

    log_returns_with_div = pd.read_pickle(data_dir / 'evans_archer_log_returns_matrix.pkl')
    print(f"With dividends data shape: {log_returns_with_div.shape}")

    # ========== COMPUTE PRICE-ONLY DATA ==========
    print("\n" + "-"*70)
    print("PART 2: Computing Price-Only Returns (No Dividends)")
    print("-"*70)

    log_returns_price_only = compute_price_only_returns(data_dir)

    # Ensure same securities in both datasets
    common_cols = list(set(log_returns_with_div.columns) & set(log_returns_price_only.columns))
    print(f"Common securities: {len(common_cols)}")

    log_returns_with_div = log_returns_with_div[common_cols]
    log_returns_price_only = log_returns_price_only[common_cols]

    # ========== RUN SIMULATIONS ==========
    print("\n" + "-"*70)
    print("PART 3: Running Simulations")
    print("-"*70)

    print("\n>>> With Dividends:")
    results_with_div = run_simulation(log_returns_with_div, random_seed=42)

    print("\n>>> Price Only:")
    results_price_only = run_simulation(log_returns_price_only, random_seed=42)

    # ========== FIT MODELS ==========
    print("\n" + "-"*70)
    print("PART 4: Fitting Models")
    print("-"*70)

    A1, B1, R2_1 = fit_hyperbola(results_with_div)
    sys_risk_1 = compute_systematic_risk(log_returns_with_div)

    A2, B2, R2_2 = fit_hyperbola(results_price_only)
    sys_risk_2 = compute_systematic_risk(log_returns_price_only)

    # Compute mean returns for params
    mean_ret_1 = results_with_div[results_with_div['portfolio_size']==1]['mean_return'].mean()
    mean_ret_2 = results_price_only[results_price_only['portfolio_size']==1]['mean_return'].mean()

    params_with_div = {
        'A': A1, 'B': B1, 'R2': R2_1, 'sys_risk': sys_risk_1,
        'semi_annual_ret': (mean_ret_1 - 1) * 100,
        'annual_ret': (mean_ret_1 ** 2 - 1) * 100
    }
    params_price_only = {
        'A': A2, 'B': B2, 'R2': R2_2, 'sys_risk': sys_risk_2,
        'semi_annual_ret': (mean_ret_2 - 1) * 100,
        'annual_ret': (mean_ret_2 ** 2 - 1) * 100
    }

    # ========== COMPARISON ==========
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)

    print(f"\n{'Parameter':<25} {'With Dividends':<18} {'Price Only':<18} {'Difference':<15}")
    print("-"*76)
    print(f"{'A (asymptote)':<25} {A1:<18.4f} {A2:<18.4f} {A2-A1:<15.4f}")
    print(f"{'B (coefficient)':<25} {B1:<18.5f} {B2:<18.5f} {B2-B1:<15.5f}")
    print(f"{'R²':<25} {R2_1:<18.4f} {R2_2:<18.4f} {R2_2-R2_1:<15.4f}")
    print(f"{'Systematic risk':<25} {sys_risk_1:<18.4f} {sys_risk_2:<18.4f} {sys_risk_2-sys_risk_1:<15.4f}")

    # Percentage differences
    print(f"\n{'Percentage Differences:'}")
    print(f"  A: {(A2-A1)/A1*100:+.2f}%")
    print(f"  B: {(B2-B1)/B1*100:+.2f}%")
    print(f"  Systematic risk: {(sys_risk_2-sys_risk_1)/sys_risk_1*100:+.2f}%")

    # Mean return comparison (semi-annual and annualized)
    mean_ret_1 = results_with_div[results_with_div['portfolio_size']==1]['mean_return'].mean()
    mean_ret_2 = results_price_only[results_price_only['portfolio_size']==1]['mean_return'].mean()

    # Annualized returns: (1 + semi-annual)^2 - 1 or equivalently geometric_mean^2
    annual_ret_1 = mean_ret_1 ** 2  # geometric mean, so square it for annual
    annual_ret_2 = mean_ret_2 ** 2

    print(f"\n{'='*76}")
    print("RETURN COMPARISON (1 security average)")
    print(f"{'='*76}")
    print(f"\n{'Metric':<30} {'With Dividends':<22} {'Price Only':<22}")
    print("-"*76)
    print(f"{'Semi-annual Return':<30} {(mean_ret_1-1)*100:>18.2f}%    {(mean_ret_2-1)*100:>18.2f}%")
    print(f"{'Annualized Return':<30} {(annual_ret_1-1)*100:>18.2f}%    {(annual_ret_2-1)*100:>18.2f}%")
    print(f"{'Dividend Contribution (semi)':<30} {(mean_ret_1-mean_ret_2)*100:>18.2f}%    {'--':>18}")
    print(f"{'Dividend Contribution (annual)':<30} {(annual_ret_1-annual_ret_2)*100:>18.2f}%    {'--':>18}")

    # ========== CREATE PLOTS ==========
    print("\n" + "-"*70)
    print("PART 5: Creating Plots")
    print("-"*70)

    # Individual Figure 1 for With Dividends
    print("\n>>> With Dividends Figure 1:")
    plot_figure1_single(
        results_with_div, params_with_div,
        title="S&P 500 (1958-1967) - With Dividends (Total Return)",
        output_path=output_dir_with_div / "figure1_with_dividends",
        color_main=PALETTE["blue_main"],
        color_secondary=PALETTE["blue_secondary"]
    )

    # Individual Figure 1 for Price Only
    print("\n>>> Price Only Figure 1:")
    plot_figure1_single(
        results_price_only, params_price_only,
        title="S&P 500 (1958-1967) - Price Only (No Dividends)",
        output_path=output_dir_price_only / "figure1_price_only",
        color_main=PALETTE["green_main"],
        color_secondary=PALETTE["green_secondary"]
    )

    # Comparison plots
    print("\n>>> Comparison Plots:")
    plot_comparison(
        results_with_div, results_price_only,
        params_with_div, params_price_only,
        output_path=output_dir_comparison / "comparison_side_by_side"
    )

    plot_overlay(
        results_with_div, results_price_only,
        params_with_div, params_price_only,
        output_path=output_dir_comparison / "comparison_overlay"
    )

    # Return plots
    print("\n>>> Return Plots:")
    plot_return_by_portfolio_size(
        results_with_div, params_with_div,
        title="S&P 500 (1958-1967) - Mean Return - With Dividends",
        output_path=output_dir_with_div / "returns_with_dividends",
        color_main=PALETTE["blue_main"],
        color_secondary=PALETTE["blue_secondary"]
    )

    plot_return_by_portfolio_size(
        results_price_only, params_price_only,
        title="S&P 500 (1958-1967) - Mean Return - Price Only",
        output_path=output_dir_price_only / "returns_price_only",
        color_main=PALETTE["green_main"],
        color_secondary=PALETTE["green_secondary"]
    )

    plot_return_comparison(
        results_with_div, results_price_only,
        params_with_div, params_price_only,
        output_path=output_dir_comparison / "returns_comparison"
    )

    # Volatility plots
    print("\n>>> Volatility Plots:")
    plot_volatility_distribution(
        results_with_div, params_with_div,
        title="S&P 500 (1958-1967) - Volatility - With Dividends",
        output_path=output_dir_with_div / "volatility_with_dividends",
        color_main=PALETTE["blue_main"],
        color_secondary=PALETTE["blue_secondary"]
    )

    plot_volatility_distribution(
        results_price_only, params_price_only,
        title="S&P 500 (1958-1967) - Volatility - Price Only",
        output_path=output_dir_price_only / "volatility_price_only",
        color_main=PALETTE["green_main"],
        color_secondary=PALETTE["green_secondary"]
    )

    plot_volatility_comparison(
        results_with_div, results_price_only,
        params_with_div, params_price_only,
        output_path=output_dir_comparison / "volatility_comparison"
    )

    # Save simulation results to respective folders
    results_with_div.to_csv(output_dir_with_div / "simulation_results.csv", index=False)
    results_price_only.to_csv(output_dir_price_only / "simulation_results.csv", index=False)
    print(f"\nSaved simulation results to respective folders")

    # ========== STATISTICAL TESTS (Evans & Archer 1968 Method) ==========
    print("\n" + "-"*70)
    print("PART 6: Statistical Tests (Evans & Archer 1968 Method)")
    print("-"*70)

    # Compute and print statistical tests for both versions
    test_results_div = compute_statistical_tests(results_with_div)
    print_statistical_tests(test_results_div, output_dir_with_div, "With Dividends")

    test_results_price = compute_statistical_tests(results_price_only)
    print_statistical_tests(test_results_price, output_dir_price_only, "Price Only")

    # ========== SUMMARY BY PORTFOLIO SIZE ==========
    print("\n" + "-"*70)
    print("PART 7: Summary by Portfolio Size")
    print("-"*70)

    # Standard deviation summary
    summary1_sd = results_with_div.groupby('portfolio_size')['std_dev'].mean()
    summary2_sd = results_price_only.groupby('portfolio_size')['std_dev'].mean()

    print(f"\n{'Size':<8} {'With Div SD':<15} {'Price Only SD':<15} {'Difference':<12} {'% Diff':<10}")
    print("-"*60)
    for size in [1, 2, 5, 8, 10, 15, 20, 30, 40]:
        sd1 = summary1_sd.loc[size]
        sd2 = summary2_sd.loc[size]
        diff = sd2 - sd1
        pct = diff / sd1 * 100
        print(f"{size:<8} {sd1:<15.4f} {sd2:<15.4f} {diff:<12.4f} {pct:<+10.2f}%")

    # Return summary (semi-annual and annualized)
    summary1_ret = results_with_div.groupby('portfolio_size')['mean_return'].mean()
    summary2_ret = results_price_only.groupby('portfolio_size')['mean_return'].mean()

    print(f"\n" + "="*90)
    print("RETURN SUMMARY BY PORTFOLIO SIZE")
    print("="*90)
    print(f"\n{'Size':<6} {'--- With Dividends ---':<30} {'--- Price Only ---':<30} {'Div Contrib':<12}")
    print(f"{'':6} {'Semi-Ann':>12} {'Annual':>12}    {'Semi-Ann':>12} {'Annual':>12}    {'(Annual)':<12}")
    print("-"*90)
    for size in [1, 2, 5, 8, 10, 15, 20, 30, 40]:
        ret1 = summary1_ret.loc[size]
        ret2 = summary2_ret.loc[size]
        # Semi-annual return (as percentage)
        semi1 = (ret1 - 1) * 100
        semi2 = (ret2 - 1) * 100
        # Annualized return: geometric_mean^2 - 1
        ann1 = (ret1 ** 2 - 1) * 100
        ann2 = (ret2 ** 2 - 1) * 100
        # Dividend contribution (annual)
        div_contrib = ann1 - ann2
        print(f"{size:<6} {semi1:>11.2f}% {ann1:>11.2f}%    {semi2:>11.2f}% {ann2:>11.2f}%    {div_contrib:>+10.2f}%")

    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    # Recalculate for final summary
    mean_ret_1 = results_with_div[results_with_div['portfolio_size']==1]['mean_return'].mean()
    mean_ret_2 = results_price_only[results_price_only['portfolio_size']==1]['mean_return'].mean()
    annual_ret_1 = mean_ret_1 ** 2
    annual_ret_2 = mean_ret_2 ** 2

    print(f"""
Impact of Excluding Dividends:

1. ASYMPTOTE (A):
   - With Dividends: {A1:.4f}
   - Price Only:     {A2:.4f}
   - Change:         {(A2-A1)/A1*100:+.2f}%

2. SYSTEMATIC RISK:
   - With Dividends: {sys_risk_1:.4f}
   - Price Only:     {sys_risk_2:.4f}
   - Change:         {(sys_risk_2-sys_risk_1)/sys_risk_1*100:+.2f}%

3. RETURN COMPARISON (1 security average):
   ┌─────────────────────┬──────────────────┬──────────────────┐
   │                     │  With Dividends  │   Price Only     │
   ├─────────────────────┼──────────────────┼──────────────────┤
   │ Semi-annual Return  │     {(mean_ret_1-1)*100:>6.2f}%      │     {(mean_ret_2-1)*100:>6.2f}%      │
   │ Annualized Return   │     {(annual_ret_1-1)*100:>6.2f}%      │     {(annual_ret_2-1)*100:>6.2f}%      │
   └─────────────────────┴──────────────────┴──────────────────┘

4. DIVIDEND CONTRIBUTION:
   - Semi-annual: {(mean_ret_1-mean_ret_2)*100:.2f}%
   - Annualized:  {(annual_ret_1-annual_ret_2)*100:.2f}%

5. KEY INSIGHT:
   Dividends contribute approximately {(annual_ret_1-annual_ret_2)*100:.1f}% annually
   to total returns during 1958-1967.

   The core diversification finding (risk decreases with more securities)
   remains valid regardless of dividend inclusion.

Output Files:
   - With Dividends: {output_dir_with_div}
   - Price Only:     {output_dir_price_only}
   - Comparison:     {output_dir_comparison}
""")

    return results_with_div, results_price_only, params_with_div, params_price_only


if __name__ == "__main__":
    results_with_div, results_price_only, params_with_div, params_price_only = main()
