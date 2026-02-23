# How Many Stocks to Significantly Reduce Portfolio Variance?

## Model Summary

Using Evans & Archer (1968) methodology:

```
Y = B/X + A
```

Where:
- Y = Portfolio standard deviation (annualized)
- X = Number of stocks
- A = Systematic risk (undiversifiable)
- B = Unsystematic risk decay coefficient

## ARKK Example (250 Days Daily)

| Parameter | Value |
|-----------|-------|
| A (asymptote) | 0.3019 |
| B (coefficient) | 0.3927 |
| Systematic variation | 0.3004 |
| R² | 0.9680 |

## Calculated Volatility by Portfolio Size

Using Y = 0.3927/X + 0.3019:

| Size | Volatility | Reduction from Size=1 | Marginal Reduction |
|------|------------|----------------------|-------------------|
| 1 | 69.5% | - | - |
| 2 | 49.8% | 28% | 28% |
| 3 | 43.2% | 38% | 13% |
| 5 | 38.1% | 45% | 6% |
| 8 | 35.1% | 50% | 3% |
| 10 | 34.1% | 51% | 2% |
| 15 | 32.8% | 53% | 1% |
| 20 | 32.2% | 54% | 0.5% |
| 40 | 31.2% | 55% | <0.5% |
| ∞ | 30.2% | 57% | 0% |

## Answer: Optimal Portfolio Size

### Definition 1: Marginal reduction < 5%
**Answer: 5 stocks**

After 5 stocks, adding one more stock reduces volatility by less than 5%.

### Definition 2: Capture 90% of diversification benefit
Total diversifiable risk = 69.5% - 30.2% = 39.3%
90% elimination = 35.4% reduction needed
**Answer: 8-10 stocks**

### Definition 3: Within 10% of systematic risk
Systematic = 30.2%, target = 33.2%
**Answer: ~15 stocks**

## Conclusion

| Criterion | Stocks Needed |
|-----------|---------------|
| Marginal benefit < 5% | 5 |
| 90% diversification benefit | 8-10 |
| Within 10% of floor | 15 |
| Diminishing returns threshold | 8-10 |

**Key Finding:** 8-10 stocks capture most diversification benefits. This aligns with Evans & Archer (1968) original conclusion.

Beyond 10-15 stocks, additional diversification provides minimal risk reduction relative to the added complexity and transaction costs.
