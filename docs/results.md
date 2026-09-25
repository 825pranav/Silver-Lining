# Results

Regenerate with `python experiments/search.py` then
`python experiments/validate.py`. Every number below is written by those
scripts; nothing here is typed by hand.

Data: 6363 daily bars, 2001-05-03 to 2026-09-11.
Out-of-sample: 23 folds, 2004-06-08 to 2026-09-11.

## Method

Expanding-window walk-forward: at least 750 training bars, 20 bars purged
between train and test, 250-bar (about one year) test folds. Each fold's
regime model (HMM, scaler and GARCH parameters) is fitted on its training
rows only, and regime probabilities come from the forward filter, so the
call on day t uses bars up to t. A position decided at the close of day t
is traded at the close of t+1. Costs are 3bp (2bp spread + 1bp slippage)
per unit of turnover, including resizing. The strategy runs as one
continuous path across folds.

`strategy.SERVED` was chosen by `experiments/search.py` from 120
configurations using folds 1-11 only (highest pooled Sharpe); see
`docs/search.md` for every configuration. Round 2 of that search was
exploratory (added after seeing round-1 full-sample results), so folds
12-23 are a clean holdout only for the round-1 configuration shown below,
not for the served one.

Served: `no-regime gold_only trend=252 nodd`. Average exposure 64% of capital; flat 28% of days.

## Out-of-sample summary

| strategy                                                            | profitable_folds   |   sharpe |   sharpe_folds_1_11 |   sharpe_folds_12_23 | total_return   | cagr   | ann_vol   | max_drawdown   |
|:--------------------------------------------------------------------|:-------------------|---------:|--------------------:|---------------------:|:---------------|:-------|:----------|:---------------|
| served: no-regime gold_only trend=252 nodd                          | 13/23              |    0.721 |               0.841 |                0.599 | +543%          | 8.7%   | 12.8%     | -29.7%         |
| best regime config, round 1: k=4 sticky=0.5 raw hard trend=252 nodd | 14/23              |    0.553 |               0.772 |                0.328 | +315%          | 6.6%   | 13.2%     | -33.2%         |
| original rules (regimes + ratio rule), corrected                    | 15/23              |    0.462 |               0.61  |                0.303 | +270%          | 6.1%   | 15.3%     | -54.1%         |
| ratio rule alone                                                    | 14/23              |    0.492 |               0.462 |                0.523 | +347%          | 7.0%   | 16.5%     | -55.1%         |
| buy & hold gold                                                     | 18/23              |    0.691 |               0.629 |                0.766 | +1036%         | 11.6%  | 18.3%     | -44.4%         |
| buy & hold silver                                                   | 15/23              |    0.494 |               0.466 |                0.523 | +1047%         | 11.5%  | 34.7%     | -75.8%         |
| buy & hold gold at 0.70x (served strategy's vol)                    | 18/23              |    0.691 |               0.629 |                0.766 | +490%          | 8.3%   | 12.8%     | -32.5%         |

The volatility-matched row scales buy-and-hold gold by a constant chosen
after the fact to match the served strategy's realised volatility. It is
not a tradable benchmark; it shows how much of a drawdown gap comes from
holding less rather than from timing.

## Is the Sharpe ratio real?

| strategy                    |   PSR(SR>0) |   DSR |   trials |   E[max SR] of trials (ann.) | Sharpe - gold [95% CI]   |   P(not > gold) | Sharpe - silver [95% CI]   |
|:----------------------------|------------:|------:|---------:|-----------------------------:|:-------------------------|----------------:|:---------------------------|
| served                      |       1     | 0.983 |      120 |                        0.263 | +0.031 [-0.188, +0.255]  |           0.381 | +0.228 [-0.092, +0.539]    |
| best regime config, round 1 |       0.995 | 0.91  |      120 |                        0.263 | -0.138 [-0.493, +0.207]  |           0.784 | +0.059 [-0.255, +0.360]    |

PSR: probability the true Sharpe is above zero given the sample's length,
skew and kurtosis. DSR: the same probability against the best Sharpe
expected from 120 unskilled trials with the dispersion observed
across the search (102 pre-registered + 18 exploratory).
Intervals: paired stationary block bootstrap (mean block 20 days,
2,000 resamples) of the annualised Sharpe difference on the same days.

## Lookahead audit

The previously published regime-gated Sharpe was 0.708. The first row
below reproduces it; each later row adds one correction to the original
rule set. The last row is the corrected original strategy used above.

| step                                                                                                 |   sharpe | total_return   | max_drawdown   |
|:-----------------------------------------------------------------------------------------------------|---------:|:---------------|:---------------|
| as published: Viterbi over each test year, GARCH fitted on all data, trade at the signal's own close |    0.708 | +750%          | -51.3%         |
| + forward-filtered regimes (no future bars)                                                          |    0.505 | +321%          | -57.5%         |
| + GARCH refitted on each training fold                                                               |    0.559 | +402%          | -57.4%         |
| + trade one day after the signal, cost on all turnover, one continuous path                          |    0.462 | +270%          | -54.1%         |

## Per fold, served strategy (with buy & hold gold)

|   fold | test_start   | test_end   |   sharpe |    cagr |   max_drawdown |   hit_rate |   days |   gold_sharpe |   gold_max_dd |
|-------:|:-------------|:-----------|---------:|--------:|---------------:|-----------:|-------:|--------------:|--------------:|
|      1 | 2004-06-08   | 2005-06-09 |    0.774 |  0.0908 |        -0.1008 |     0.5125 |    248 |         0.834 |       -0.0952 |
|      2 | 2005-06-10   | 2006-06-09 |    2.496 |  0.4427 |        -0.0875 |     0.5628 |    250 |         1.986 |       -0.155  |
|      3 | 2006-06-12   | 2007-06-11 |    0.409 |  0.0529 |        -0.1167 |     0.5063 |    250 |         0.462 |       -0.1569 |
|      4 | 2007-06-12   | 2008-06-06 |    1.855 |  0.3498 |        -0.1255 |     0.552  |    250 |         1.706 |       -0.1538 |
|      5 | 2008-06-09   | 2009-06-04 |   -0.314 | -0.0513 |        -0.1901 |     0.4477 |    250 |         0.445 |       -0.279  |
|      6 | 2009-06-05   | 2010-06-02 |    1.313 |  0.2326 |        -0.1234 |     0.5774 |    250 |         1.386 |       -0.1357 |
|      7 | 2010-06-03   | 2011-05-27 |    1.716 |  0.2673 |        -0.0759 |     0.624  |    250 |         1.674 |       -0.0789 |
|      8 | 2011-05-31   | 2012-05-24 |    0.354 |  0.046  |        -0.1491 |     0.512  |    250 |         0.173 |       -0.1866 |
|      9 | 2012-05-25   | 2013-05-24 |   -0.595 | -0.0605 |        -0.0963 |     0.4314 |    250 |        -0.544 |       -0.2416 |
|     10 | 2013-05-28   | 2014-05-22 |    0     |  0      |         0      |     0      |    250 |        -0.259 |       -0.1588 |
|     11 | 2014-05-23   | 2015-05-20 |   -0.674 | -0.0378 |        -0.0565 |     0.3953 |    250 |        -0.381 |       -0.1467 |
|     12 | 2015-05-21   | 2016-05-17 |    0.287 |  0.0188 |        -0.0334 |     0.5077 |    250 |         0.423 |       -0.1308 |
|     13 | 2016-05-18   | 2017-05-17 |   -0.28  | -0.0405 |        -0.1708 |     0.4781 |    250 |        -0.044 |       -0.1737 |
|     14 | 2017-05-18   | 2018-05-16 |   -0.301 | -0.0303 |        -0.0899 |     0.4912 |    250 |         0.296 |       -0.0799 |
|     15 | 2018-05-17   | 2019-05-15 |   -0.96  | -0.0347 |        -0.0649 |     0.3962 |    250 |         0.097 |       -0.098  |
|     16 | 2019-05-16   | 2020-05-12 |    1.544 |  0.228  |        -0.0866 |     0.5579 |    250 |         1.549 |       -0.1178 |
|     17 | 2020-05-13   | 2021-05-10 |    0.593 |  0.08   |        -0.1699 |     0.564  |    250 |         0.521 |       -0.1822 |
|     18 | 2021-05-11   | 2022-05-05 |   -0.192 | -0.024  |        -0.0904 |     0.5175 |    250 |         0.208 |       -0.0975 |
|     19 | 2022-05-06   | 2023-05-04 |   -0.998 | -0.0888 |        -0.1325 |     0.4393 |    250 |         0.668 |       -0.1371 |
|     20 | 2023-05-05   | 2024-05-02 |    1     |  0.1387 |        -0.113  |     0.524  |    250 |         1.014 |       -0.113  |
|     21 | 2024-05-03   | 2025-05-01 |    2.113 |  0.3738 |        -0.0783 |     0.576  |    250 |         2.018 |       -0.0799 |
|     22 | 2025-05-02   | 2026-04-29 |    1.653 |  0.3151 |        -0.1346 |     0.576  |    250 |         1.424 |       -0.1773 |
|     23 | 2026-04-30   | 2026-09-11 |   -0.43  | -0.1076 |        -0.1142 |     0.5161 |     93 |        -0.333 |       -0.1557 |
