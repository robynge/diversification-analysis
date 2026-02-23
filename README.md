# ARK ETF Diversification Analysis

Replication of Evans & Archer (1968) methodology using ARK ETF holdings data.

## Overview

This project analyzes the diversification effects of portfolio construction using holdings from ARK Invest ETFs. It follows the classic Evans & Archer (1968) approach to demonstrate how portfolio risk (standard deviation) decreases as the number of securities increases.

## Data Sources

- **ARK ETF Holdings**: Daily holdings and weights from ARK Invest ETFs (ARKK, ARKF, ARKG, ARKQ, ARKW, ARKX)
- **Price Data**: Historical stock prices from `prices.xlsx` (backup source)

## Main Scripts

| File | Description |
|------|-------------|
| `ark_etf_diversification_analysis.py` | Main analysis script - runs Monte Carlo simulations |
| `ark_etf/` | Directory containing ETF holdings data |
| `figures/ark_etf/` | Output directory for generated figures |

## Usage

```bash
python ark_etf_diversification_analysis.py
```

## Output Structure

```
figures/ark_etf/
├── comparison/           # Cross-ETF comparison charts
│   └── comparison_*.png
├── ARKK/                 # Individual ETF results
├── ARKF/
├── ARKG/
├── ARKQ/
├── ARKW/
├── ARKX/
└── summary_results.csv   # Summary statistics
```

## Methodology

The analysis performs Monte Carlo simulations with two weighting schemes:
- **Equal Weighted**: Each security receives equal weight (1/n)
- **MarketCap Weighted**: Securities weighted by their ARK ETF holdings weight

For each portfolio size (1 to N securities), the script:
1. Randomly selects securities
2. Computes portfolio log returns
3. Calculates annualized standard deviation
4. Fits a hyperbolic model: Y = B/X + A

## Time Periods

- 60 Days (Daily/Weekly)
- 120 Days (Daily/Weekly)
- 250 Days (Daily/Weekly)

## Reference

Evans, J. L., & Archer, S. H. (1968). Diversification and the reduction of dispersion: An empirical analysis. *The Journal of Finance*, 23(5), 761-767.
