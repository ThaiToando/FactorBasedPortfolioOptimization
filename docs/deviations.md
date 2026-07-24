
# Deviations from Auh & Cho (2023)

Every departure from the paper's stated method or data, with its reason.

## D1 :  WBA excluded from the universe (data availability)

**Paper:** 30 DJIA constituents.
**This implementation:** 29. Walgreens Boots Alliance (WBA) is excluded.

Sycamore Partners completed its take-private acquisition of WBA on 2025-08-28;
Nasdaq suspended the listing on 2025-08-29. Yahoo Finance no longer serves the
symbol under either the bulk or per-ticker endpoint, and no free alternative
(stooq, pandas-datareader) returns dividend-adjusted history for it.

A price-only substitute was rejected: WBA yielded roughly 3-5% annually over
2010-2021, so an unadjusted series understates its total return by that amount
every year. Since the strategies here optimise on estimated mean returns, a
systematically depressed mean on one of thirty assets is not noise -- it
changes portfolio weights.

**Effect:** investable universe is 28 assets from 2009-12-31 to 2020-02-28 and
29 from 2020-03-31 onward. Results are not directly comparable to a 30-asset
run at the third decimal, though the direction and rough magnitude of the
Sharpe comparison should be unaffected by removing one of thirty names.

**Reproducibility note:** this is a *vintage* problem, not a method problem. A
replication attempted in 2021 would have had all 30. Anyone reproducing this
work in future should expect further attrition as constituents are acquired or
delisted.

## D2 : Survivorship bias in the static universe

The headline universe is the DJIA as of 2021-12-31 held fixed backwards, which
is almost certainly what the paper used but carries survivorship bias by
construction. A point-in-time variant is available for sensitivity analysis.
