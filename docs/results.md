# Results

Regenerate with `python experiments/validate.py`. Every number below is
written by that script; nothing here is typed by hand.

Data: 6363 daily bars, 2001-05-03 to 2026-09-11.

## Method

Expanding-window walk-forward. The first 750 bars (~3 years) train
before any out-of-sample call, each test fold is 250 bars (~1 year),
and 20 bars are purged between train and test so rolling features
cannot leak across the boundary. The regime model is fitted on training rows
only and never sees the fold it scores. Costs are 2bp spread plus 1bp
slippage on every position change.

The positions come from `strategy.decide`, which is also what the dashboard
calls for its live recommendation, so the figures below describe the
strategy that is actually served.

## Out-of-sample summary

| strategy          | folds   | profitable_folds   | mean_fold_sharpe   |   pooled_sharpe |   total_return |   max_drawdown |
|:------------------|:--------|:-------------------|:-------------------|----------------:|---------------:|---------------:|
| ratio rule        | 23      | 14                 | 0.414              |           0.483 |         3.2267 |        -0.5876 |
| regime-gated      | 23      | 17                 | 0.669              |           0.708 |         7.5029 |        -0.513  |
| buy & hold gold   | -       | -                  | -                  |           0.69  |        10.3143 |        -0.4436 |
| buy & hold silver | -       | -                  | -                  |           0.491 |        10.2366 |        -0.7585 |

## Per fold, regime-gated

|   fold | test_start   | test_end   |   sharpe |    cagr |   max_drawdown |   hit_rate |   days |
|-------:|:-------------|:-----------|---------:|--------:|---------------:|-----------:|-------:|
|      1 | 2004-06-08   | 2005-06-09 |    0.946 |  0.1426 |        -0.087  |     0.5692 |    249 |
|      2 | 2005-06-10   | 2006-06-09 |    1.813 |  0.3302 |        -0.0885 |     0.6188 |    249 |
|      3 | 2006-06-12   | 2007-06-11 |    0.695 |  0.0992 |        -0.1004 |     0.5766 |    249 |
|      4 | 2007-06-12   | 2008-06-06 |    2.003 |  0.3515 |        -0.1242 |     0.5692 |    249 |
|      5 | 2008-06-09   | 2009-06-04 |    0.409 |  0.0579 |        -0.1891 |     0.4758 |    249 |
|      6 | 2009-06-05   | 2010-06-02 |    1.048 |  0.1699 |        -0.12   |     0.5775 |    249 |
|      7 | 2010-06-03   | 2011-05-27 |    2.898 |  0.5312 |        -0.0817 |     0.6116 |    249 |
|      8 | 2011-05-31   | 2012-05-24 |    0.239 |  0.0265 |        -0.1417 |     0.5081 |    249 |
|      9 | 2012-05-25   | 2013-05-24 |   -0.515 | -0.0939 |        -0.2379 |     0.4933 |    249 |
|     10 | 2013-05-28   | 2014-05-22 |   -0.569 | -0.1007 |        -0.1322 |     0.4632 |    249 |
|     11 | 2014-05-23   | 2015-05-20 |   -0.928 | -0.1582 |        -0.2819 |     0.482  |    249 |
|     12 | 2015-05-21   | 2016-05-17 |   -0.024 | -0.0141 |        -0.1727 |     0.4766 |    249 |
|     13 | 2016-05-18   | 2017-05-17 |    0.833 |  0.1385 |        -0.1506 |     0.5463 |    249 |
|     14 | 2017-05-18   | 2018-05-16 |    0.349 |  0.0377 |        -0.0793 |     0.5    |    249 |
|     15 | 2018-05-17   | 2019-05-15 |    0.289 |  0.0213 |        -0.0923 |     0.5    |    249 |
|     16 | 2019-05-16   | 2020-05-12 |    0.863 |  0.1126 |        -0.1137 |     0.5252 |    249 |
|     17 | 2020-05-13   | 2021-05-10 |    1.12  |  0.1903 |        -0.15   |     0.5833 |    249 |
|     18 | 2021-05-11   | 2022-05-05 |   -0.14  | -0.0257 |        -0.0924 |     0.5214 |    249 |
|     19 | 2022-05-06   | 2023-05-04 |    1.481 |  0.2095 |        -0.0901 |     0.5351 |    249 |
|     20 | 2023-05-05   | 2024-05-02 |    0.118 |  0.0068 |        -0.066  |     0.4951 |    249 |
|     21 | 2024-05-03   | 2025-05-01 |    0.742 |  0.1044 |        -0.0958 |     0.5241 |    249 |
|     22 | 2025-05-02   | 2026-04-29 |    1.999 |  0.4386 |        -0.1341 |     0.5565 |    249 |
|     23 | 2026-04-30   | 2026-09-11 |   -0.292 | -0.0659 |        -0.1356 |     0.5165 |     92 |
