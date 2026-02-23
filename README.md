# Diversification Analysis

Replication of Evans & Archer (1968) methodology using S&P 500 and ARK ETF holdings data.

## Overview

This project analyzes the diversification effects of portfolio construction. It follows the classic Evans & Archer (1968) approach to demonstrate how portfolio risk (standard deviation) decreases as the number of securities increases.

## Data Sources

- **S&P 500**: CRSP/WRDS historical data (1958-1967) for original replication
- **ARK ETF Holdings**: Daily holdings and weights from ARK Invest ETFs (ARKK, ARKF, ARKG, ARKQ, ARKW, ARKX)

## Main Scripts

| File | Description |
|------|-------------|
| `evans_archer_replication.py` | Original paper replication using S&P 500 data |
| `ark_etf_diversification_analysis.py` | Extended analysis using ARK ETF holdings |

## Usage

```bash
# S&P 500 replication (requires WRDS data)
python evans_archer_replication.py

# ARK ETF analysis
python ark_etf_diversification_analysis.py
```

## Output Structure

```
output/
├── sp500/
│   ├── with_dividends/   # S&P 500 with dividends
│   ├── price_only/       # S&P 500 price only
│   └── comparison/       # S&P 500 comparison charts
└── ark_etf/
    ├── comparison/       # Cross-ETF comparison
    ├── ARKK/
    ├── ARKF/
    ├── ARKG/
    ├── ARKQ/
    ├── ARKW/
    ├── ARKX/
    └── summary_results.csv
```

## Methodology

Monte Carlo simulations with two weighting schemes:
- **Equal Weighted**: Each security receives equal weight (1/n)
- **MarketCap Weighted**: Securities weighted by holdings weight

For each portfolio size (1 to N securities):
1. Randomly select securities
2. Compute portfolio log returns
3. Calculate annualized standard deviation
4. Fit hyperbolic model: Y = B/X + A

Statistical tests following Evans & Archer (1968):
- t-test: Significant reduction in mean SD
- F-test: Convergence of variance

## Reference

Evans, J. L., & Archer, S. H. (1968). Diversification and the reduction of dispersion: An empirical analysis. *The Journal of Finance*, 23(5), 761-767.
