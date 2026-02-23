"""
ARK ETF Diversification Analysis
Replication of Evans & Archer (1968) Methodology

This script analyzes diversification effects for ARK ETF holdings.
EXACTLY follows ark_etf_diversification_analysis.ipynb

Input Data:
- ARK ETF holdings data from ark_etf/ folder
- Backup prices from prices.xlsx

Output:
- output/ark_etf/{etf_name}/ - Individual ETF figures
- output/ark_etf/comparison/ - Comparison figures across ETFs
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numba import jit


# ============================================================================
# Configuration (EXACTLY from notebook cell-3)
# ============================================================================

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "red_strong": "#B64342", "teal": "#42949E", "violet": "#9A4D8E",
    "green": "#2E7D32", "orange": "#E65100"
}

ETF_COLORS = {
    'ARKK': '#0F4D92',  # Blue
    'ARKF': '#2E7D32',  # Green
    'ARKG': '#9A4D8E',  # Violet
    'ARKQ': '#E65100',  # Orange
    'ARKW': '#42949E',  # Teal
    'ARKX': '#B64342',  # Red
}

ETF_COLORS_SECONDARY = {
    'ARKK': '#3775BA',
    'ARKF': '#66BB6A',
    'ARKG': '#BA68C8',
    'ARKQ': '#FF8A65',
    'ARKW': '#80CBC4',
    'ARKX': '#EF5350',
}

# Cash/currency tickers to exclude (EXACTLY from notebook cell-5)
CASH_TICKERS = {
    # Currency holdings
    'CAD', 'EUR', 'HKD', 'JPY', 'USD', 'ZAR',
    # Money market funds
    'FIRXX', 'FTOXX',
}

# ETFs to analyze
ETF_LIST = ['ARKK', 'ARKF', 'ARKG', 'ARKQ', 'ARKW', 'ARKX']

# Time periods (EXACTLY from notebook cell-14)
TIME_PERIODS = [
    {'name': '60 Days Daily', 'days': 60, 'freq': 'daily'},
    {'name': '60 Days Weekly', 'days': 60, 'freq': 'weekly'},
    {'name': '120 Days Daily', 'days': 120, 'freq': 'daily'},
    {'name': '120 Days Weekly', 'days': 120, 'freq': 'weekly'},
    {'name': '250 Days Daily', 'days': 250, 'freq': 'daily'},
    {'name': '250 Days Weekly', 'days': 250, 'freq': 'weekly'},
]

# Annualization factors (EXACTLY from notebook cell-9)
ANNUALIZATION_FACTOR = {
    'daily': np.sqrt(252),
    'weekly': np.sqrt(52),
    'monthly': np.sqrt(12),
}


def apply_publication_style(font_size=14, axes_linewidth=2.2):
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
        "text.usetex": False,
        "mathtext.fontset": "stix",
    })


# ============================================================================
# Data Loading Functions (EXACTLY from notebook cell-5 and cell-6)
# ============================================================================

def load_ark_etf_data(etf_name, data_dir='ark_etf'):
    """
    Load ARK ETF data including holdings and historical prices.
    Returns current holdings, latest date, and raw dataframe.
    EXACTLY from notebook cell-5.
    """
    file_path = Path(data_dir) / f"{etf_name}_Transformed_Data.xlsx"
    if not file_path.exists():
        print(f"  [ERROR] File not found: {file_path}")
        return None, None, None

    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Get most recent holdings
    latest_date = df['Date'].max()
    current_holdings = df[df['Date'] == latest_date][['Ticker', 'Weight']].copy()
    current_holdings = current_holdings.dropna(subset=['Ticker'])
    current_holdings['Ticker'] = current_holdings['Ticker'].str.strip()

    # Filter out cash/currency holdings
    excluded = current_holdings[current_holdings['Ticker'].isin(CASH_TICKERS)]['Ticker'].tolist()
    current_holdings = current_holdings[~current_holdings['Ticker'].isin(CASH_TICKERS)].reset_index(drop=True)

    if excluded:
        print(f"  [INFO] Excluded {len(excluded)} cash/currency holdings: {excluded}")

    return current_holdings, latest_date, df


def extract_prices_from_etf_data(df, tickers, start_date, end_date):
    """
    Extract historical prices from ARK ETF data file.
    Uses Stock_Price column which already has the price data.
    EXACTLY from notebook cell-5.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter date range
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    df_filtered = df[mask]

    # Build price matrix
    price_data = df_filtered[df_filtered['Ticker'].isin(tickers)][['Date', 'Ticker', 'Stock_Price']]

    # Pivot to get tickers as columns
    prices = price_data.pivot_table(index='Date', columns='Ticker', values='Stock_Price', aggfunc='first')
    prices = prices.sort_index()

    return prices


def get_weekly_prices(prices):
    """Resample daily prices to weekly (Friday close). EXACTLY from notebook cell-5."""
    return prices.resample('W-FRI').last()


def load_backup_prices(data_dir='ark_etf'):
    """
    Load price data from prices.xlsx as a backup source.
    EXACTLY from notebook cell-6.
    """
    file_path = Path(data_dir) / "prices.xlsx"
    if not file_path.exists():
        print("  [WARNING] prices.xlsx not found")
        return None

    df = pd.read_excel(file_path, header=None)

    ticker_row = df.iloc[0].values

    price_data = {}

    col = 0
    while col < len(ticker_row):
        ticker = ticker_row[col]
        if pd.notna(ticker) and isinstance(ticker, str):
            close_col = col + 4
            date_col = col

            if close_col < len(df.columns):
                dates = df.iloc[2:, date_col].values
                closes = df.iloc[2:, close_col].values

                valid_mask = pd.notna(dates)
                if valid_mask.any():
                    ticker_dates = pd.to_datetime(dates[valid_mask])
                    ticker_closes = pd.to_numeric(closes[valid_mask], errors='coerce')
                    price_data[ticker] = pd.Series(ticker_closes, index=ticker_dates)

            col += 6
        else:
            col += 1

    if not price_data:
        print("  [WARNING] No price data found in prices.xlsx")
        return None

    prices_df = pd.DataFrame(price_data)
    prices_df = prices_df.sort_index()

    return prices_df


# ============================================================================
# Analysis Functions (EXACTLY from notebook cell-9)
# ============================================================================

def compute_log_returns(prices):
    """Compute log returns from price series."""
    return np.log(prices / prices.shift(1))


def compute_portfolio_stats(log_returns, selected_tickers):
    """Compute portfolio geometric mean return and standard deviation."""
    port_ret = log_returns[selected_tickers].mean(axis=1)
    return np.exp(port_ret.mean()), port_ret.std()


@jit(nopython=True)
def _run_simulation_core(log_ret, n_tickers, max_size, n_runs, ann_factor):
    """Numba-accelerated core loop for equal weighted simulation."""
    total_results = n_runs * max_size
    portfolio_sizes = np.zeros(total_results, dtype=np.int32)
    runs = np.zeros(total_results, dtype=np.int32)
    std_devs = np.zeros(total_results)
    mean_returns = np.zeros(total_results)

    idx = 0
    for run in range(n_runs):
        shuffled = np.random.permutation(n_tickers)
        for size in range(1, max_size + 1):
            selected = shuffled[:size]
            port_ret = np.zeros(log_ret.shape[0])
            for i in range(log_ret.shape[0]):
                s = 0.0
                for j in range(size):
                    s += log_ret[i, selected[j]]
                port_ret[i] = s / size

            portfolio_sizes[idx] = size
            runs[idx] = run
            std_devs[idx] = np.std(port_ret) * ann_factor
            mean_returns[idx] = np.exp(np.mean(port_ret))
            idx += 1

    return portfolio_sizes, runs, std_devs, mean_returns


def run_simulation(prices, max_size=40, n_runs=1000, seed=42, freq='daily'):
    """
    Run Monte Carlo simulation for diversification analysis (equal weighted).
    Uses Numba for acceleration.
    """
    np.random.seed(seed)
    log_ret = compute_log_returns(prices).dropna().values
    n_tickers = log_ret.shape[1]
    max_size = min(max_size, n_tickers)
    ann_factor = ANNUALIZATION_FACTOR.get(freq, 1.0)

    print(f"  Running {n_runs} simulations (equal weighted), max size {max_size}, {n_tickers} tickers")

    portfolio_sizes, runs, std_devs, mean_returns = _run_simulation_core(
        log_ret, n_tickers, max_size, n_runs, ann_factor
    )

    print(f"    Completed {n_runs} simulations (equal weighted)")

    return pd.DataFrame({
        'portfolio_size': portfolio_sizes,
        'run': runs,
        'std_dev': std_devs,
        'mean_return': mean_returns
    })


@jit(nopython=True)
def _run_simulation_weighted_core(log_ret, weights_array, n_tickers, max_size, n_runs, ann_factor):
    """Numba-accelerated core loop for weighted simulation."""
    total_results = n_runs * max_size
    portfolio_sizes = np.zeros(total_results, dtype=np.int32)
    runs = np.zeros(total_results, dtype=np.int32)
    std_devs = np.zeros(total_results)
    mean_returns = np.zeros(total_results)

    idx = 0
    for run in range(n_runs):
        shuffled = np.random.permutation(n_tickers)
        for size in range(1, max_size + 1):
            selected = shuffled[:size]

            # Get weights and normalize
            weight_sum = 0.0
            for j in range(size):
                weight_sum += weights_array[selected[j]]

            port_ret = np.zeros(log_ret.shape[0])
            for i in range(log_ret.shape[0]):
                s = 0.0
                if weight_sum > 0:
                    for j in range(size):
                        s += log_ret[i, selected[j]] * weights_array[selected[j]] / weight_sum
                else:
                    for j in range(size):
                        s += log_ret[i, selected[j]] / size
                port_ret[i] = s

            portfolio_sizes[idx] = size
            runs[idx] = run
            std_devs[idx] = np.std(port_ret) * ann_factor
            mean_returns[idx] = np.exp(np.mean(port_ret))
            idx += 1

    return portfolio_sizes, runs, std_devs, mean_returns


def run_simulation_weighted(prices, holdings, max_size=40, n_runs=1000, seed=42, freq='daily'):
    """
    Run Monte Carlo simulation using ARK actual weights.
    Uses Numba for acceleration.
    """
    np.random.seed(seed)
    log_ret = compute_log_returns(prices).dropna().values
    ticker_list = list(prices.columns)
    n_tickers = len(ticker_list)
    max_size = min(max_size, n_tickers)
    ann_factor = ANNUALIZATION_FACTOR.get(freq, 1.0)

    # Build weight array aligned with prices columns
    weight_map = {}
    for _, row in holdings.iterrows():
        ticker = row['Ticker']
        if ticker in ticker_list:
            weight_map[ticker] = row['Weight']
    weights_array = np.array([weight_map.get(t, 0.0) for t in ticker_list])

    print(f"  Running {n_runs} simulations (MarketCap weighted), max size {max_size}, {n_tickers} tickers")

    portfolio_sizes, runs, std_devs, mean_returns = _run_simulation_weighted_core(
        log_ret, weights_array, n_tickers, max_size, n_runs, ann_factor
    )

    print(f"    Completed {n_runs} simulations (MarketCap weighted)")

    return pd.DataFrame({
        'portfolio_size': portfolio_sizes,
        'run': runs,
        'std_dev': std_devs,
        'mean_return': mean_returns
    })


def fit_hyperbola(results):
    """Fit hyperbolic model Y = B/X + A to simulation results. EXACTLY from notebook cell-9."""
    mean_std = results.groupby('portfolio_size')['std_dev'].mean()
    X, Y = mean_std.index.values, mean_std.values
    params, _, _, _ = np.linalg.lstsq(np.column_stack([np.ones_like(X), 1/X]), Y, rcond=None)
    A, B = params
    R2 = 1 - np.sum((Y - (A + B/X))**2) / np.sum((Y - Y.mean())**2)
    return A, B, R2


def compute_systematic_risk(prices, freq='daily'):
    """Compute systematic risk (market portfolio std dev), annualized. EXACTLY from notebook cell-9."""
    ann_factor = ANNUALIZATION_FACTOR.get(freq, 1.0)
    return compute_log_returns(prices).dropna().mean(axis=1).std() * ann_factor


def compute_annualized_return(mean_return: float, freq: str) -> float:
    """Convert period return to annualized return."""
    if freq == 'daily':
        return (mean_return ** 252 - 1) * 100
    elif freq == 'weekly':
        return (mean_return ** 52 - 1) * 100
    else:
        return (mean_return - 1) * 100


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


def print_statistical_tests(test_results: dict, etf_name: str, period_name: str, weight_label: str, max_size: int = 40):
    """Print statistical test results matching Evans & Archer (1968) style."""

    print(f"\n  [{weight_label}] Securities needed for significant reduction (α=0.05):")
    print(f"  {'Start':<8} {'t-test':<12} {'F-test':<12}")
    print(f"  {'-'*32}")

    key_sizes = [1, 2, 5, 8, 10, 15, 20]
    key_sizes = [s for s in key_sizes if s < max_size]

    t_test_dict = {r['starting_size']: r for r in test_results['securities_needed_t_test']}
    f_test_dict = {r['starting_size']: r for r in test_results['securities_needed_f_test']}

    for start in key_sizes:
        t_row = t_test_dict.get(start, {})
        f_row = f_test_dict.get(start, {})

        t_needed = t_row.get('securities_needed')
        f_needed = f_row.get('securities_needed')

        t_str = str(t_needed) if t_needed else f">{max_size - start}"
        f_str = str(f_needed) if f_needed else f">{max_size - start}"

        print(f"  {start:<8} +{t_str:<11} +{f_str:<11}")


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_diversification_curve(
    results, params, etf_name, period_name, color, color_secondary,
    output_path: Path
):
    """Plot diversification curve for a single ETF and time period."""
    apply_publication_style(14, 2.2)
    fig, ax = plt.subplots(figsize=(10, 7))

    stats = results.groupby('portfolio_size')['std_dev'].agg(['mean', 'std', 'count']).reset_index()
    X = stats['portfolio_size'].values
    Y = stats['mean'].values
    Ystd = stats['std'].values
    n_runs = stats['count'].iloc[0]

    Xs = np.linspace(1, X.max(), 200)
    ci_u = Y + 1.96 * Ystd / np.sqrt(n_runs)
    ci_l = Y - 1.96 * Ystd / np.sqrt(n_runs)

    ax.fill_between(X, ci_l, ci_u, color=color_secondary, alpha=0.25)
    ax.plot(Xs, params['B']/Xs + params['A'], color=color, linewidth=2.5)
    ax.scatter(X, Y, color=color, s=60, zorder=5, edgecolors='white')
    ax.axhline(y=params['sys_risk'], color=PALETTE["red_strong"], linewidth=1.5)

    # Systematic variation label (right lower corner, no overlap)
    ax.text(0.98, 0.02, f'Systematic variation = {params["sys_risk"]:.4f}', fontsize=10,
            transform=ax.transAxes, ha='right', va='bottom')

    xs, ys, lh = 0.52, 0.96, 0.045
    ax.plot([xs, xs+0.06], [ys, ys], color=color, lw=2.5,
            transform=ax.transAxes, clip_on=False)
    ax.text(xs+0.08, ys, '= predicted Y', fontsize=10,
            transform=ax.transAxes, va='center')
    ax.plot(xs+0.03, ys-lh, 'o', color=color, ms=8,
            transform=ax.transAxes, clip_on=False)
    ax.text(xs+0.08, ys-lh, '= actual Y', fontsize=10,
            transform=ax.transAxes, va='center')
    ax.add_patch(Rectangle((xs, ys-2*lh-0.015), 0.06, 0.03,
                           facecolor=color_secondary, alpha=0.4,
                           transform=ax.transAxes, clip_on=False))
    ax.text(xs+0.08, ys-2*lh, '= 95% CI', fontsize=10,
            transform=ax.transAxes, va='center')
    ax.text(xs, ys-3.5*lh, r'$Y = B/X + A$', fontsize=10,
            transform=ax.transAxes, va='top')
    ax.text(xs, ys-4.5*lh, f'A={params["A"]:.4f}, B={params["B"]:.5f}', fontsize=10,
            transform=ax.transAxes, va='top')
    ax.text(xs, ys-5.5*lh, f'$R^2$={params["R2"]:.4f}', fontsize=10,
            transform=ax.transAxes, va='top')

    ax.set_xlabel('Portfolio size')
    ax.set_ylabel('Mean portfolio std dev')
    ax.set_title(f'{etf_name} - {period_name}', fontweight='bold', loc='left')
    ax.set_xlim(0, X.max() + 2)
    ax.set_ylim(min(params['sys_risk'], Y.min()) * 0.95, Y.max() * 1.08)

    fig.tight_layout(pad=2)
    fig.savefig(output_path, dpi=600, facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_volatility_distribution(
    results, params, etf_name, period_name, color, color_secondary,
    output_path: Path
):
    """Plot annualized volatility distribution by portfolio size."""
    apply_publication_style(14, 2.2)
    fig, ax = plt.subplots(figsize=(10, 7))

    # std_dev is already annualized in run_simulation
    stats = results.groupby('portfolio_size')['std_dev'].agg(
        ['mean', 'std', 'min', 'max', 'count']
    ).reset_index()

    X = stats['portfolio_size'].values
    Y = stats['mean'].values * 100  # Convert to percentage
    Ystd = stats['std'].values * 100
    n_runs = stats['count'].iloc[0]
    ci_u = Y + 1.96 * Ystd / np.sqrt(n_runs)
    ci_l = Y - 1.96 * Ystd / np.sqrt(n_runs)
    Y_min = stats['min'].values * 100
    Y_max = stats['max'].values * 100

    ax.fill_between(X, Y_min, Y_max, color=color_secondary, alpha=0.15)
    ax.fill_between(X, ci_l, ci_u, color=color_secondary, alpha=0.35)
    ax.plot(X, Y, color=color, linewidth=2.5, marker='o', markersize=6,
            markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5)

    # Systematic risk line
    sys_risk_pct = params['sys_risk'] * 100
    ax.axhline(y=sys_risk_pct, color=PALETTE["red_strong"], linewidth=1.5)

    # Legend (right upper corner)
    xs, ys, lh = 0.70, 0.96, 0.045
    ax.plot([xs, xs+0.06], [ys, ys], color=color, lw=2.5,
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
    ax.set_title(f'{etf_name} - {period_name}\nVolatility Distribution by Portfolio Size', fontweight='bold', loc='left')
    ax.set_xlim(0, X.max() + 2)
    y_range = Y_max.max() - Y_min.min()
    ax.set_ylim(max(0, Y_min.min() - y_range * 0.1), Y_max.max() + y_range * 0.1)

    fig.tight_layout(pad=2)
    fig.savefig(output_path, dpi=600, facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_etf_comparison(etf_results, period_name, output_path: Path):
    """Plot comparison of all ETFs for a given time period."""
    apply_publication_style(12, 2.0)
    n_etfs = len(etf_results)
    ncols = min(3, n_etfs)
    nrows = (n_etfs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, (etf_name, data) in enumerate(etf_results.items()):
        ax = axes[idx]
        color = ETF_COLORS.get(etf_name, PALETTE["blue_main"])
        A, B, R2 = data['A'], data['B'], data['R_squared']
        sys_risk = data['systematic_risk']

        stats = data['results'].groupby('portfolio_size')['std_dev'].agg(['mean', 'std', 'count']).reset_index()
        X = stats['portfolio_size'].values
        Y = stats['mean'].values
        Ystd = stats['std'].values
        n_runs = stats['count'].iloc[0]
        ci_u = Y + 1.96 * Ystd / np.sqrt(n_runs)
        ci_l = Y - 1.96 * Ystd / np.sqrt(n_runs)

        ax.fill_between(X, ci_l, ci_u, color=color, alpha=0.2)
        ax.plot(np.linspace(1, X.max(), 100), B/np.linspace(1, X.max(), 100)+A, color=color, lw=2)
        ax.scatter(X, Y, color=color, s=35, zorder=5, edgecolors='white')
        ax.axhline(y=sys_risk, color=PALETTE["red_strong"], lw=1.2)

        ax.set_xlabel('Portfolio size')
        ax.set_ylabel('Mean std dev' if idx % ncols == 0 else '')
        ax.set_title(f'{etf_name}', loc='left', fontweight='bold')
        ax.text(0.95, 0.95, f'$R^2$={R2:.4f}', transform=ax.transAxes, ha='right', va='top', fontsize=10)

    for idx in range(len(etf_results), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Diversification Analysis - {period_name}', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout(pad=2)
    fig.savefig(output_path, dpi=600, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_etf_volatility_comparison(etf_results, period_name, output_path: Path):
    """Plot annualized volatility comparison of all ETFs for a given time period."""
    apply_publication_style(12, 2.0)
    n_etfs = len(etf_results)
    ncols = min(3, n_etfs)
    nrows = (n_etfs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, (etf_name, data) in enumerate(etf_results.items()):
        ax = axes[idx]
        color = ETF_COLORS.get(etf_name, PALETTE["blue_main"])
        color_secondary = ETF_COLORS_SECONDARY.get(etf_name, PALETTE["blue_secondary"])
        sys_risk = data['systematic_risk']

        # std_dev is already annualized
        stats = data['results'].groupby('portfolio_size')['std_dev'].agg(
            ['mean', 'std', 'min', 'max', 'count']
        ).reset_index()

        X = stats['portfolio_size'].values
        Y = stats['mean'].values * 100  # Convert to percentage
        Ystd = stats['std'].values * 100
        n_runs = stats['count'].iloc[0]
        ci_u = Y + 1.96 * Ystd / np.sqrt(n_runs)
        ci_l = Y - 1.96 * Ystd / np.sqrt(n_runs)
        Y_min = stats['min'].values * 100
        Y_max = stats['max'].values * 100

        ax.fill_between(X, Y_min, Y_max, color=color_secondary, alpha=0.15)
        ax.fill_between(X, ci_l, ci_u, color=color_secondary, alpha=0.35)
        ax.plot(X, Y, color=color, linewidth=2, marker='o', markersize=4,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=1)

        # Systematic risk line
        sys_risk_pct = sys_risk * 100
        ax.axhline(y=sys_risk_pct, color=PALETTE["red_strong"], linewidth=1.2)

        ax.set_xlabel('Portfolio size')
        ax.set_ylabel('Annualized Volatility (%)' if idx % ncols == 0 else '')
        ax.set_title(f'{etf_name}', loc='left', fontweight='bold')
        ax.text(0.95, 0.05, f'Sys={sys_risk_pct:.1f}%', transform=ax.transAxes, ha='right', va='bottom', fontsize=9)

        ax.set_xlim(0, X.max() + 2)
        y_range = Y_max.max() - Y_min.min()
        ax.set_ylim(max(0, Y_min.min() - y_range * 0.1), Y_max.max() + y_range * 0.1)

    for idx in range(len(etf_results), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Annualized Volatility - {period_name}', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout(pad=2)
    fig.savefig(output_path, dpi=600, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Main Execution (EXACTLY follows notebook cell-18)
# ============================================================================

def main():
    """Main execution function."""

    print("="*70)
    print("ARK ETF DIVERSIFICATION ANALYSIS")
    print("Evans & Archer (1968) Methodology")
    print("="*70)

    base_dir = Path(__file__).parent
    data_dir = base_dir / "ark_etf"

    output_base = base_dir / "output" / "ark_etf"
    output_comparison = output_base / "comparison"
    output_comparison.mkdir(parents=True, exist_ok=True)
    for etf in ETF_LIST:
        (output_base / etf).mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_base}")

    today = datetime.now()
    print(f"Analysis Date: {today.strftime('%Y-%m-%d')}")

    # Load backup prices (EXACTLY from notebook cell-15)
    print("\n" + "-"*70)
    print("Loading prices.xlsx as backup price source...")
    print("-"*70)
    backup_prices = load_backup_prices(data_dir)
    if backup_prices is not None:
        print(f"  Loaded {len(backup_prices.columns)} tickers, {len(backup_prices)} dates")
        print(f"  Date range: {backup_prices.index.min().date()} to {backup_prices.index.max().date()}")
        print(f"  Tickers: {list(backup_prices.columns)}")

    # Load ETF data (EXACTLY from notebook cell-16)
    print("\n" + "-"*70)
    print("Loading ARK ETF Data")
    print("-"*70)

    etf_data = {}
    for etf in ETF_LIST:
        holdings, latest_date, raw_df = load_ark_etf_data(etf, data_dir)
        if holdings is not None:
            etf_data[etf] = {
                'holdings': holdings,
                'latest_date': latest_date,
                'raw_df': raw_df
            }
            print(f"{etf}: {len(holdings)} holdings as of {latest_date.date()}")
            print(f"  Top 5: {holdings['Ticker'].head(5).tolist()}")

    # Store all results
    all_results = {}
    missing_data_summary = {}
    backup_fill_summary = {}

    # Run analysis for each period (EXACTLY from notebook cell-18)
    for period in TIME_PERIODS:
        period_name = period['name']
        freq = period['freq']
        all_results[period_name] = {}

        start_date = (today - timedelta(days=period['days'])).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')

        print(f"\n{'='*70}")
        print(f"PERIOD: {period_name} ({start_date} to {end_date})")
        print(f"{'='*70}")

        # Get trading days from prices.xlsx (the source of truth for market open days)
        if backup_prices is not None:
            backup_start = pd.to_datetime(start_date)
            backup_end = pd.to_datetime(end_date)
            backup_filtered = backup_prices[(backup_prices.index >= backup_start) & (backup_prices.index <= backup_end)]
            if freq == 'weekly':
                backup_data = backup_filtered.resample('W-FRI').last()
            else:
                backup_data = backup_filtered
            backup_data.index = pd.to_datetime(backup_data.index).normalize()
            trading_days_from_backup = set(backup_data.index)
        else:
            trading_days_from_backup = None
            backup_data = None

        for etf_name, data in etf_data.items():
            print(f"\n{'-'*40}")
            print(f"ETF: {etf_name}")
            print(f"{'-'*40}")

            holdings = data['holdings']
            raw_df = data['raw_df']
            tickers = holdings['Ticker'].tolist()

            # Extract prices from ETF data file (uses Stock_Price column)
            prices_daily = extract_prices_from_etf_data(raw_df, tickers, start_date, end_date)

            # Resample based on frequency
            if freq == 'weekly':
                prices = get_weekly_prices(prices_daily)
            else:
                prices = prices_daily
            prices.index = pd.to_datetime(prices.index).normalize()

            # Filter prices to only include trading days from prices.xlsx
            if trading_days_from_backup:
                valid_trading_days = [d for d in prices.index if d in trading_days_from_backup]
                prices = prices.loc[valid_trading_days]

            # Track backup fills for this ETF/period
            backup_fills = {}

            # Fill ALL tickers from prices.xlsx (EXACTLY from notebook)
            if backup_prices is not None and backup_data is not None:
                # Align backup_data to prices index
                backup_data_aligned = backup_data.reindex(prices.index)

                # Record original valid counts before filling
                orig_valid_counts = {}
                for ticker in tickers:
                    if ticker in prices.columns:
                        orig_valid_counts[ticker] = prices[ticker].notna().sum()

                # Fill all tickers
                for ticker in tickers:
                    if ticker in backup_data_aligned.columns:
                        if ticker not in prices.columns:
                            # Add ticker from backup
                            prices[ticker] = backup_data_aligned[ticker]
                            backup_fills[ticker] = 100.0
                        else:
                            # Fill NaN values
                            prices[ticker] = prices[ticker].fillna(backup_data_aligned[ticker])

                # Calculate how much was filled from backup
                for ticker in tickers:
                    if ticker in prices.columns and ticker not in backup_fills:
                        orig_valid = orig_valid_counts.get(ticker, 0)
                        new_valid = prices[ticker].notna().sum()
                        filled_from_backup = new_valid - orig_valid
                        if filled_from_backup > 0:
                            pct_from_backup = (filled_from_backup / new_valid) * 100 if new_valid > 0 else 0
                            backup_fills[ticker] = pct_from_backup

                if backup_fills:
                    print(f"  [INFO] Filled from prices.xlsx:")
                    for ticker, pct in sorted(backup_fills.items(), key=lambda x: -x[1])[:5]:
                        print(f"    - {ticker}: {pct:.1f}% from prices.xlsx")
                    if len(backup_fills) > 5:
                        print(f"    ... and {len(backup_fills) - 5} more")

            # Store backup fill info
            if backup_fills:
                key = f"{etf_name}_{period_name}"
                backup_fill_summary[key] = {
                    'etf': etf_name,
                    'period': period_name,
                    'fills': backup_fills
                }

            # Check for STILL missing data AFTER backup fill
            n_trading_days = len(prices)
            still_missing = {}

            for ticker in tickers:
                if ticker not in prices.columns:
                    still_missing[ticker] = 100.0
                else:
                    nan_count = prices[ticker].isna().sum()
                    if nan_count > 0:
                        missing_pct = (nan_count / n_trading_days) * 100
                        still_missing[ticker] = missing_pct

            if still_missing:
                key = f"{etf_name}_{period_name}"
                missing_data_summary[key] = {
                    'etf': etf_name,
                    'period': period_name,
                    'missing_tickers': still_missing
                }
                print(f"  [WARNING] Still missing after backup fill ({len(still_missing)} tickers):")
                for ticker, pct in sorted(still_missing.items(), key=lambda x: -x[1])[:5]:
                    print(f"    - {ticker}: {pct:.1f}% missing")
                if len(still_missing) > 5:
                    print(f"    ... and {len(still_missing) - 5} more")
                print("  [INFO] Please add these tickers to prices.xlsx.")

            # Filter to tickers that have at least some data
            valid_tickers = [t for t in tickers if t in prices.columns and prices[t].notna().sum() > 0]

            if len(valid_tickers) < 5:
                print(f"  [SKIP] Not enough valid tickers ({len(valid_tickers)})")
                continue

            prices_clean = prices[valid_tickers].dropna(how='all', axis=0).dropna(how='all', axis=1)

            # Drop any remaining columns with >50% NaN
            valid_cols = prices_clean.columns[prices_clean.notna().sum() >= len(prices_clean) * 0.5]
            prices_clean = prices_clean[valid_cols].dropna()

            if len(prices_clean) < 3 or len(prices_clean.columns) < 5:
                print(f"  [SKIP] Insufficient data after cleaning")
                continue

            freq_label = "days" if freq == "daily" else "weeks"
            print(f"  Analyzing {len(prices_clean.columns)} tickers over {len(prices_clean)} {freq_label}")

            max_portfolio_size = min(40, len(prices_clean.columns))
            sys_risk = compute_systematic_risk(prices_clean, freq=freq)
            freq_label_display = "Daily" if freq == "daily" else "Weekly"

            # Run EQUAL WEIGHTED simulation
            results_ew = run_simulation(prices_clean, max_size=max_portfolio_size, freq=freq)
            A_ew, B_ew, R2_ew = fit_hyperbola(results_ew)

            single_stock_returns_ew = results_ew[results_ew['portfolio_size'] == 1]['mean_return']
            mean_return_ew = single_stock_returns_ew.mean()
            period_ret_ew = (mean_return_ew - 1) * 100
            annual_ret_ew = compute_annualized_return(mean_return_ew, freq)

            print(f"  [Equal Weighted] Y = {B_ew:.4f}/X + {A_ew:.4f}, R²={R2_ew:.4f}")
            print(f"    Systematic={sys_risk:.4f}, Mean Return: {period_ret_ew:.2f}% ({freq})")

            # Run ARK WEIGHTED simulation
            results_wt = run_simulation_weighted(prices_clean, holdings, max_size=max_portfolio_size, freq=freq)
            A_wt, B_wt, R2_wt = fit_hyperbola(results_wt)

            single_stock_returns_wt = results_wt[results_wt['portfolio_size'] == 1]['mean_return']
            mean_return_wt = single_stock_returns_wt.mean()
            period_ret_wt = (mean_return_wt - 1) * 100
            annual_ret_wt = compute_annualized_return(mean_return_wt, freq)

            print(f"  [MarketCap Weighted] Y = {B_wt:.4f}/X + {A_wt:.4f}, R²={R2_wt:.4f}")
            print(f"    Systematic={sys_risk:.4f}, Mean Return: {period_ret_wt:.2f}% ({freq})")

            all_results[period_name][etf_name] = {
                'equal_weighted': {
                    'results': results_ew,
                    'A': A_ew, 'B': B_ew, 'R_squared': R2_ew,
                    'systematic_risk': sys_risk,
                    'n_periods': len(prices_clean),
                    'n_tickers': len(prices_clean.columns),
                    'freq': freq,
                    'freq_label': freq_label_display,
                    'period_ret': period_ret_ew,
                    'annual_ret': annual_ret_ew
                },
                'marketcap_weighted': {
                    'results': results_wt,
                    'A': A_wt, 'B': B_wt, 'R_squared': R2_wt,
                    'systematic_risk': sys_risk,
                    'n_periods': len(prices_clean),
                    'n_tickers': len(prices_clean.columns),
                    'freq': freq,
                    'freq_label': freq_label_display,
                    'period_ret': period_ret_wt,
                    'annual_ret': annual_ret_wt
                }
            }

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)

    # Define weight types for iteration
    weight_types = [
        ('equal_weighted', 'Equal Weighted'),
        ('marketcap_weighted', 'MarketCap Weighted')
    ]

    # Comparison plots (std dev) - for each weight type
    print("\n>>> Comparison Plots (Std Dev):")
    for weight_key, weight_label in weight_types:
        for period_name, etf_results in all_results.items():
            if etf_results:
                safe_period = period_name.lower().replace(' ', '_')
                # Extract data for this weight type
                etf_results_for_weight = {}
                for etf_name, data in etf_results.items():
                    if weight_key in data:
                        etf_results_for_weight[etf_name] = data[weight_key]
                if etf_results_for_weight:
                    plot_etf_comparison(
                        etf_results_for_weight,
                        f"{period_name} ({weight_label})",
                        output_comparison / f"comparison_{safe_period}_{weight_key}.png"
                    )

    # Volatility comparison plots - for each weight type
    print("\n>>> Volatility Comparison Plots:")
    for weight_key, weight_label in weight_types:
        for period_name, etf_results in all_results.items():
            if etf_results:
                safe_period = period_name.lower().replace(' ', '_')
                etf_results_for_weight = {}
                for etf_name, data in etf_results.items():
                    if weight_key in data:
                        etf_results_for_weight[etf_name] = data[weight_key]
                if etf_results_for_weight:
                    plot_etf_volatility_comparison(
                        etf_results_for_weight,
                        f"{period_name} ({weight_label})",
                        output_comparison / f"volatility_{safe_period}_{weight_key}.png"
                    )

    # Individual ETF plots - for each weight type
    print("\n>>> Individual ETF Plots:")
    for period_name, etf_results in all_results.items():
        for etf_name, etf_data in etf_results.items():
            safe_period = period_name.lower().replace(' ', '_')

            for weight_key, weight_label in weight_types:
                if weight_key not in etf_data:
                    continue
                data = etf_data[weight_key]
                params = {
                    'A': data['A'],
                    'B': data['B'],
                    'R2': data['R_squared'],
                    'sys_risk': data['systematic_risk'],
                    'period_ret': data['period_ret'],
                    'annual_ret': data['annual_ret'],
                    'freq_label': data['freq_label']
                }
                title_suffix = f" ({weight_label})"

                # Diversification curve
                plot_diversification_curve(
                    data['results'],
                    params,
                    etf_name, period_name + title_suffix,
                    ETF_COLORS.get(etf_name, PALETTE["blue_main"]),
                    ETF_COLORS_SECONDARY.get(etf_name, PALETTE["blue_secondary"]),
                    output_base / etf_name / f"{etf_name}_{safe_period}_{weight_key}.png"
                )
                # Volatility distribution plot
                plot_volatility_distribution(
                    data['results'],
                    params,
                    etf_name, period_name + title_suffix,
                    ETF_COLORS.get(etf_name, PALETTE["blue_main"]),
                    ETF_COLORS_SECONDARY.get(etf_name, PALETTE["blue_secondary"]),
                    output_base / etf_name / f"{etf_name}_{safe_period}_{weight_key}_volatility.png"
                )

    # Summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    summary_rows = []
    for period_name, etf_results in all_results.items():
        for etf_name, etf_data in etf_results.items():
            for weight_key, weight_label in weight_types:
                if weight_key not in etf_data:
                    continue
                data = etf_data[weight_key]
                summary_rows.append({
                    'Period': period_name,
                    'ETF': etf_name,
                    'Weight_Type': weight_label,
                    'Tickers': data['n_tickers'],
                    'Periods': data['n_periods'],
                    'A': data['A'],
                    'B': data['B'],
                    'R²': data['R_squared'],
                    'Systematic': data['systematic_risk']
                })

    summary_df = pd.DataFrame(summary_rows)

    print(f"\n{'Period':<18} {'ETF':<6} {'Weight':<14} {'Tickers':>7} {'A':>8} {'B':>8} {'R²':>7} {'Sys':>7}")
    print("-"*90)
    for _, row in summary_df.iterrows():
        print(f"{row['Period']:<18} {row['ETF']:<6} {row['Weight_Type']:<14} {row['Tickers']:>7} "
              f"{row['A']:>8.4f} {row['B']:>8.4f} {row['R²']:>7.4f} {row['Systematic']:>7.4f}")

    summary_df.to_csv(output_base / "summary_results.csv", index=False)
    print(f"\nSaved summary to: {output_base / 'summary_results.csv'}")

    # Statistical tests (Evans & Archer 1968 method)
    print("\n" + "="*70)
    print("STATISTICAL TESTS (Evans & Archer 1968 Method)")
    print("="*70)
    print("\nSecurities needed for significant reduction in mean SD (α=0.05)")

    for period_name, etf_results in all_results.items():
        print(f"\n>>> {period_name}")
        for etf_name, etf_data in etf_results.items():
            print(f"\n  {etf_name}:")
            for weight_key, weight_label in weight_types:
                if weight_key not in etf_data:
                    continue
                data = etf_data[weight_key]
                max_size = data['results']['portfolio_size'].max()
                test_results = compute_statistical_tests(data['results'])
                print_statistical_tests(test_results, etf_name, period_name, weight_label, max_size)

    # Missing data summary
    if missing_data_summary:
        print("\n" + "="*70)
        print("MISSING DATA SUMMARY - REQUIRES MANUAL ATTENTION")
        print("="*70)
        print("\nThe following tickers had insufficient price data and were NOT found in prices.xlsx:")
        print("Please provide price data for these tickers manually.\n")

        for key, info in missing_data_summary.items():
            print(f"\n{info['etf']} - {info['period']}:")
            for ticker, pct in sorted(info['missing_tickers'].items(), key=lambda x: -x[1])[:10]:
                print(f"  - {ticker}: {pct:.1f}% missing")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput files saved to: {output_base}")

    return all_results


if __name__ == "__main__":
    all_results = main()
