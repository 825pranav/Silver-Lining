# Configuration search

Regenerate with `python experiments/search.py`. Every number below is
written by that script; nothing here is typed by hand.

**120 configurations** were run (102 in round 1, 18 in
round 2), all listed below. Every choice used only folds 1-11
(2004-06-08 to 2015-05-20): highest pooled Sharpe wins. Folds 12-23 (2015-05-21 to 2026-09-11)
are shown for every configuration but did not influence any choice.

Round 1 was fixed before any result was seen. Round 2 is exploratory:
it was added after round 1's full-sample results showed that plain
vol-targeted gold beat every round-1 configuration, and it adds a
gold-core universe (`gold_core`: trending holds gold, silver only in
risk_on; `gold_only`: silver never held). Its other dimensions were
pruned mechanically to values within 0.02 of the best mean
folds 1-11 Sharpe in round 1, plus a no-regime control. Because round 2
was motivated by results that include folds 12-23, those folds are not
a clean holdout for a round-2 winner; the Deflated Sharpe Ratio in
`docs/results.md` counts every configuration from both rounds.

Round 1 grid: regime model none, or a Gaussian HMM with 4 states or
BIC-chosen from 3/4/5; sticky transition prior 0 or 0.5; raw or
scale-free price features; hard (argmax) or soft (probability-weighted)
regime actions; time-series momentum gate off, 126 or 252 days;
drawdown de-risking off or on. Fixed, not searched: 1% daily vol target
per leg, no leverage, 3bp per unit of turnover, one-day execution lag,
the ratio-rule thresholds and the regime-to-metal mapping.

Rank correlation between folds 1-11 and folds 12-23 Sharpe across all
configurations: 0.25.

## Average effect of each round-1 choice

| dimension      | value   |   mean_sel_sharpe |   mean_conf_sharpe | kept_for_round_2   |
|:---------------|:--------|------------------:|-------------------:|:-------------------|
| n_states       | 4       |             0.528 |              0.323 | False              |
| n_states       | bic     |             0.606 |              0.469 | True               |
| n_states       | none    |             0.552 |              0.431 | False              |
| sticky         | 0.0     |             0.567 |              0.401 | True               |
| sticky         | 0.5     |             0.568 |              0.39  | True               |
| scale_free     | False   |             0.65  |              0.355 | True               |
| scale_free     | True    |             0.484 |              0.437 | False              |
| soft           | False   |             0.572 |              0.392 | True               |
| soft           | True    |             0.563 |              0.4   | True               |
| trend_lookback | 126     |             0.517 |              0.327 | False              |
| trend_lookback | 252     |             0.612 |              0.403 | True               |
| trend_lookback | none    |             0.57  |              0.464 | False              |
| dd_derisk      | False   |             0.572 |              0.389 | True               |
| dd_derisk      | True    |             0.56  |              0.407 | True               |

## All configurations, by folds 1-11 Sharpe

|   round | config                                             |   sel_sharpe |   sel_max_dd |   conf_sharpe |   conf_max_dd |   full_sharpe |   full_max_dd |   profitable_folds | selected   |
|--------:|:---------------------------------------------------|-------------:|-------------:|--------------:|--------------:|--------------:|--------------:|-------------------:|:-----------|
|       2 | no-regime gold_only trend=252 nodd                 |        0.841 |       -0.217 |         0.599 |        -0.282 |         0.721 |        -0.297 |                 13 | True       |
|       2 | no-regime gold_only trend=252 dd                   |        0.82  |       -0.195 |         0.667 |        -0.257 |         0.743 |        -0.282 |                 12 | False      |
|       1 | k=4 sticky=0.5 raw hard trend=252 nodd             |        0.772 |       -0.278 |         0.328 |        -0.332 |         0.553 |        -0.332 |                 14 | False      |
|       2 | k=bic sticky=0.5 raw hard gold_core trend=252 nodd |        0.756 |       -0.246 |         0.383 |        -0.255 |         0.574 |        -0.267 |                 12 | False      |
|       2 | k=bic sticky=0.5 raw soft gold_core trend=252 nodd |        0.752 |       -0.249 |         0.381 |        -0.255 |         0.571 |        -0.27  |                 12 | False      |
|       2 | k=bic sticky=0 raw hard gold_core trend=252 nodd   |        0.741 |       -0.294 |         0.497 |        -0.205 |         0.622 |        -0.294 |                 13 | False      |
|       1 | k=4 sticky=0.5 raw hard trend=252 dd               |        0.737 |       -0.229 |         0.251 |        -0.323 |         0.498 |        -0.323 |                 13 | False      |
|       1 | k=bic sticky=0 raw hard trend=252 nodd             |        0.736 |       -0.294 |         0.341 |        -0.33  |         0.541 |        -0.373 |                 14 | False      |
|       1 | k=4 sticky=0.5 raw soft trend=252 nodd             |        0.734 |       -0.278 |         0.359 |        -0.316 |         0.549 |        -0.316 |                 14 | False      |
|       2 | k=bic sticky=0 raw soft gold_core trend=252 nodd   |        0.728 |       -0.295 |         0.497 |        -0.208 |         0.616 |        -0.302 |                 13 | False      |
|       1 | k=bic sticky=0 raw hard trend=252 dd               |        0.721 |       -0.233 |         0.406 |        -0.276 |         0.564 |        -0.31  |                 13 | False      |
|       1 | k=bic sticky=0 raw soft trend=252 nodd             |        0.717 |       -0.296 |         0.348 |        -0.298 |         0.535 |        -0.356 |                 14 | False      |
|       1 | k=bic sticky=0.5 raw hard trend=252 nodd           |        0.716 |       -0.283 |         0.305 |        -0.273 |         0.514 |        -0.283 |                 13 | False      |
|       2 | k=bic sticky=0 raw soft gold_core trend=252 dd     |        0.714 |       -0.235 |         0.542 |        -0.226 |         0.628 |        -0.275 |                 12 | False      |
|       1 | k=4 sticky=0.5 raw soft trend=252 dd               |        0.714 |       -0.228 |         0.333 |        -0.313 |         0.526 |        -0.313 |                 13 | False      |
|       2 | k=bic sticky=0 raw hard gold_core trend=252 dd     |        0.712 |       -0.235 |         0.541 |        -0.206 |         0.626 |        -0.263 |                 12 | False      |
|       1 | k=4 sticky=0.5 raw soft trend=off dd               |        0.708 |       -0.316 |         0.425 |        -0.216 |         0.567 |        -0.371 |                 12 | False      |
|       1 | k=4 sticky=0.5 raw hard trend=off dd               |        0.708 |       -0.333 |         0.382 |        -0.204 |         0.546 |        -0.383 |                 13 | False      |
|       1 | k=4 sticky=0 raw soft trend=252 nodd               |        0.707 |       -0.282 |         0.339 |        -0.319 |         0.527 |        -0.319 |                 13 | False      |
|       1 | k=4 sticky=0 raw hard trend=252 nodd               |        0.707 |       -0.301 |         0.344 |        -0.299 |         0.528 |        -0.319 |                 14 | False      |
|       1 | k=bic sticky=0 raw soft trend=252 dd               |        0.706 |       -0.233 |         0.398 |        -0.24  |         0.553 |        -0.284 |                 12 | False      |
|       2 | k=bic sticky=0 raw soft gold_only trend=252 nodd   |        0.7   |       -0.215 |         0.567 |        -0.205 |         0.635 |        -0.288 |                 14 | False      |
|       1 | k=4 sticky=0 raw soft trend=252 dd                 |        0.697 |       -0.229 |         0.343 |        -0.317 |         0.523 |        -0.317 |                 11 | False      |
|       1 | k=bic sticky=0.5 raw soft trend=252 nodd           |        0.696 |       -0.289 |         0.342 |        -0.243 |         0.523 |        -0.289 |                 13 | False      |
|       1 | k=bic sticky=0.5 scalefree hard trend=252 nodd     |        0.696 |       -0.339 |         0.573 |        -0.279 |         0.635 |        -0.339 |                 14 | False      |
|       1 | k=4 sticky=0 raw hard trend=252 dd                 |        0.695 |       -0.237 |         0.296 |        -0.304 |         0.499 |        -0.304 |                 12 | False      |
|       1 | k=bic sticky=0.5 scalefree soft trend=252 nodd     |        0.695 |       -0.327 |         0.554 |        -0.279 |         0.625 |        -0.327 |                 14 | False      |
|       2 | k=bic sticky=0 raw hard gold_only trend=252 nodd   |        0.693 |       -0.215 |         0.549 |        -0.205 |         0.622 |        -0.29  |                 14 | False      |
|       1 | k=4 sticky=0 raw soft trend=off dd                 |        0.691 |       -0.326 |         0.336 |        -0.233 |         0.515 |        -0.406 |                 12 | False      |
|       2 | k=bic sticky=0.5 raw hard gold_core trend=252 dd   |        0.686 |       -0.226 |         0.401 |        -0.275 |         0.544 |        -0.275 |                 10 | False      |
|       1 | k=4 sticky=0 raw hard trend=off dd                 |        0.685 |       -0.329 |         0.262 |        -0.237 |         0.477 |        -0.398 |                 12 | False      |
|       2 | k=bic sticky=0 raw soft gold_only trend=252 dd     |        0.672 |       -0.195 |         0.619 |        -0.193 |         0.646 |        -0.272 |                 13 | False      |
|       1 | k=4 sticky=0.5 raw hard trend=126 nodd             |        0.662 |       -0.316 |         0.305 |        -0.263 |         0.483 |        -0.347 |                 14 | False      |
|       1 | k=4 sticky=0.5 raw hard trend=off nodd             |        0.661 |       -0.46  |         0.464 |        -0.214 |         0.565 |        -0.528 |                 16 | False      |
|       1 | k=bic sticky=0.5 scalefree hard trend=off dd       |        0.658 |       -0.289 |         0.672 |        -0.196 |         0.666 |        -0.318 |                 16 | False      |
|       1 | k=bic sticky=0.5 scalefree hard trend=252 dd       |        0.654 |       -0.259 |         0.577 |        -0.268 |         0.615 |        -0.268 |                 13 | False      |
|       2 | k=bic sticky=0.5 raw hard gold_only trend=252 nodd |        0.647 |       -0.215 |         0.469 |        -0.226 |         0.56  |        -0.236 |                 14 | False      |
|       2 | k=bic sticky=0 raw hard gold_only trend=252 dd     |        0.646 |       -0.196 |         0.596 |        -0.191 |         0.621 |        -0.27  |                 13 | False      |
|       1 | no-regime trend=252 dd                             |        0.644 |       -0.188 |         0.463 |        -0.301 |         0.552 |        -0.301 |                 12 | False      |
|       1 | k=bic sticky=0 raw hard trend=126 nodd             |        0.643 |       -0.226 |         0.294 |        -0.307 |         0.468 |        -0.322 |                 12 | False      |
|       1 | k=bic sticky=0.5 scalefree soft trend=252 dd       |        0.643 |       -0.25  |         0.544 |        -0.279 |         0.591 |        -0.279 |                 13 | False      |
|       1 | k=bic sticky=0 raw hard trend=126 dd               |        0.64  |       -0.207 |         0.341 |        -0.275 |         0.489 |        -0.289 |                 13 | False      |
|       1 | k=bic sticky=0.5 scalefree hard trend=off nodd     |        0.637 |       -0.384 |         0.651 |        -0.212 |         0.643 |        -0.411 |                 15 | False      |
|       1 | k=4 sticky=0.5 raw soft trend=off nodd             |        0.637 |       -0.451 |         0.46  |        -0.214 |         0.551 |        -0.528 |                 16 | False      |
|       1 | no-regime trend=252 nodd                           |        0.636 |       -0.251 |         0.446 |        -0.299 |         0.541 |        -0.299 |                 12 | False      |
|       1 | k=bic sticky=0 raw soft trend=off dd               |        0.635 |       -0.336 |         0.558 |        -0.214 |         0.596 |        -0.45  |                 15 | False      |
|       1 | k=4 sticky=0.5 raw soft trend=126 nodd             |        0.632 |       -0.302 |         0.298 |        -0.271 |         0.465 |        -0.339 |                 14 | False      |
|       1 | k=bic sticky=0 raw hard trend=off dd               |        0.631 |       -0.337 |         0.567 |        -0.225 |         0.599 |        -0.452 |                 14 | False      |
|       1 | k=bic sticky=0 raw soft trend=126 nodd             |        0.629 |       -0.219 |         0.306 |        -0.3   |         0.467 |        -0.334 |                 12 | False      |
|       1 | k=bic sticky=0.5 scalefree soft trend=off dd       |        0.629 |       -0.301 |         0.633 |        -0.197 |         0.631 |        -0.341 |                 13 | False      |
|       1 | k=4 sticky=0.5 raw hard trend=126 dd               |        0.629 |       -0.298 |         0.344 |        -0.248 |         0.484 |        -0.33  |                 11 | False      |
|       2 | k=bic sticky=0.5 raw soft gold_only trend=252 nodd |        0.628 |       -0.221 |         0.486 |        -0.226 |         0.558 |        -0.234 |                 14 | False      |
|       2 | k=bic sticky=0.5 raw soft gold_core trend=252 dd   |        0.625 |       -0.225 |         0.439 |        -0.269 |         0.532 |        -0.269 |                 11 | False      |
|       1 | k=bic sticky=0 raw soft trend=126 dd               |        0.623 |       -0.211 |         0.366 |        -0.261 |         0.494 |        -0.289 |                 13 | False      |
|       1 | k=bic sticky=0 raw hard trend=off nodd             |        0.623 |       -0.414 |         0.51  |        -0.281 |         0.567 |        -0.554 |                 16 | False      |
|       1 | k=4 sticky=0.5 raw soft trend=126 dd               |        0.622 |       -0.272 |         0.341 |        -0.248 |         0.48  |        -0.306 |                 12 | False      |
|       1 | k=bic sticky=0.5 raw hard trend=126 nodd           |        0.619 |       -0.199 |         0.275 |        -0.267 |         0.449 |        -0.267 |                 12 | False      |
|       1 | k=bic sticky=0.5 scalefree soft trend=off nodd     |        0.618 |       -0.389 |         0.645 |        -0.205 |         0.631 |        -0.422 |                 16 | False      |
|       1 | k=bic sticky=0 scalefree hard trend=off nodd       |        0.615 |       -0.323 |         0.69  |        -0.223 |         0.651 |        -0.34  |                 18 | False      |
|       1 | k=bic sticky=0 raw soft trend=off nodd             |        0.614 |       -0.423 |         0.515 |        -0.275 |         0.566 |        -0.56  |                 16 | False      |
|       1 | k=bic sticky=0 scalefree soft trend=252 nodd       |        0.613 |       -0.264 |         0.591 |        -0.187 |         0.602 |        -0.264 |                 15 | False      |
|       1 | k=bic sticky=0 scalefree soft trend=off nodd       |        0.613 |       -0.328 |         0.683 |        -0.198 |         0.646 |        -0.35  |                 17 | False      |
|       1 | k=4 sticky=0 raw hard trend=126 dd                 |        0.611 |       -0.277 |         0.265 |        -0.234 |         0.439 |        -0.308 |                 12 | False      |
|       1 | k=bic sticky=0.5 raw hard trend=252 dd             |        0.61  |       -0.237 |         0.37  |        -0.247 |         0.49  |        -0.247 |                 11 | False      |
|       1 | k=4 sticky=0 raw hard trend=off nodd               |        0.61  |       -0.476 |         0.303 |        -0.238 |         0.462 |        -0.541 |                 15 | False      |
|       1 | k=4 sticky=0 raw soft trend=off nodd               |        0.609 |       -0.463 |         0.311 |        -0.235 |         0.466 |        -0.54  |                 15 | False      |
|       1 | k=4 sticky=0 raw hard trend=126 nodd               |        0.606 |       -0.315 |         0.271 |        -0.261 |         0.439 |        -0.344 |                 13 | False      |
|       1 | k=bic sticky=0.5 raw soft trend=252 dd             |        0.603 |       -0.238 |         0.383 |        -0.238 |         0.493 |        -0.238 |                 11 | False      |
|       1 | k=4 sticky=0 raw soft trend=126 nodd               |        0.599 |       -0.31  |         0.274 |        -0.271 |         0.437 |        -0.344 |                 12 | False      |
|       1 | k=4 sticky=0 raw soft trend=126 dd                 |        0.598 |       -0.28  |         0.343 |        -0.239 |         0.471 |        -0.312 |                 11 | False      |
|       1 | k=bic sticky=0 scalefree hard trend=252 nodd       |        0.597 |       -0.264 |         0.599 |        -0.192 |         0.598 |        -0.264 |                 15 | False      |
|       1 | k=bic sticky=0.5 raw hard trend=126 dd             |        0.595 |       -0.208 |         0.249 |        -0.268 |         0.423 |        -0.268 |                 11 | False      |
|       1 | k=bic sticky=0.5 raw soft trend=126 nodd           |        0.594 |       -0.202 |         0.285 |        -0.27  |         0.442 |        -0.271 |                 11 | False      |
|       1 | k=bic sticky=0.5 raw hard trend=off dd             |        0.591 |       -0.339 |         0.416 |        -0.205 |         0.505 |        -0.437 |                 11 | False      |
|       1 | k=bic sticky=0.5 raw soft trend=126 dd             |        0.588 |       -0.208 |         0.277 |        -0.261 |         0.432 |        -0.261 |                 11 | False      |
|       1 | k=bic sticky=0.5 raw hard trend=off nodd           |        0.578 |       -0.433 |         0.387 |        -0.231 |         0.486 |        -0.512 |                 15 | False      |
|       1 | k=bic sticky=0 scalefree soft trend=off dd         |        0.575 |       -0.267 |         0.755 |        -0.197 |         0.667 |        -0.289 |                 16 | False      |
|       1 | k=bic sticky=0.5 raw soft trend=off nodd           |        0.555 |       -0.431 |         0.411 |        -0.231 |         0.485 |        -0.513 |                 16 | False      |
|       1 | k=bic sticky=0 scalefree soft trend=252 dd         |        0.551 |       -0.225 |         0.606 |        -0.215 |         0.58  |        -0.232 |                 13 | False      |
|       1 | k=bic sticky=0 scalefree hard trend=252 dd         |        0.548 |       -0.227 |         0.597 |        -0.206 |         0.573 |        -0.232 |                 13 | False      |
|       2 | k=bic sticky=0.5 raw hard gold_only trend=252 dd   |        0.543 |       -0.193 |         0.535 |        -0.205 |         0.539 |        -0.258 |                 13 | False      |
|       1 | no-regime trend=126 dd                             |        0.54  |       -0.225 |         0.329 |        -0.28  |         0.434 |        -0.326 |                 12 | False      |
|       1 | k=bic sticky=0.5 scalefree hard trend=126 nodd     |        0.534 |       -0.241 |         0.418 |        -0.229 |         0.476 |        -0.258 |                 11 | False      |
|       1 | k=4 sticky=0 scalefree hard trend=off dd           |        0.526 |       -0.248 |         0.305 |        -0.275 |         0.417 |        -0.42  |                 13 | False      |
|       1 | k=bic sticky=0.5 scalefree soft trend=126 dd       |        0.526 |       -0.262 |         0.347 |        -0.278 |         0.434 |        -0.291 |                  9 | False      |
|       1 | k=bic sticky=0 scalefree hard trend=off dd         |        0.525 |       -0.275 |         0.75  |        -0.22  |         0.64  |        -0.297 |                 16 | False      |
|       1 | no-regime trend=126 nodd                           |        0.525 |       -0.223 |         0.332 |        -0.333 |         0.427 |        -0.374 |                 13 | False      |
|       1 | k=4 sticky=0 scalefree hard trend=off nodd         |        0.524 |       -0.308 |         0.277 |        -0.298 |         0.406 |        -0.449 |                 14 | False      |
|       1 | k=bic sticky=0.5 scalefree hard trend=126 dd       |        0.521 |       -0.256 |         0.285 |        -0.32  |         0.401 |        -0.32  |                  9 | False      |
|       1 | k=bic sticky=0 scalefree soft trend=126 dd         |        0.519 |       -0.247 |         0.467 |        -0.202 |         0.493 |        -0.265 |                 13 | False      |
|       2 | k=bic sticky=0.5 raw soft gold_only trend=252 dd   |        0.515 |       -0.197 |         0.538 |        -0.211 |         0.526 |        -0.259 |                 13 | False      |
|       1 | k=bic sticky=0.5 raw soft trend=off dd             |        0.514 |       -0.336 |         0.411 |        -0.206 |         0.464 |        -0.431 |                 12 | False      |
|       1 | k=bic sticky=0.5 scalefree soft trend=126 nodd     |        0.51  |       -0.235 |         0.405 |        -0.23  |         0.457 |        -0.258 |                 11 | False      |
|       1 | no-regime trend=off dd                             |        0.504 |       -0.375 |         0.495 |        -0.246 |         0.499 |        -0.494 |                 13 | False      |
|       1 | k=4 sticky=0 scalefree soft trend=off nodd         |        0.499 |       -0.301 |         0.263 |        -0.304 |         0.387 |        -0.459 |                 14 | False      |
|       1 | k=bic sticky=0 scalefree soft trend=126 nodd       |        0.491 |       -0.219 |         0.493 |        -0.22  |         0.492 |        -0.238 |                 12 | False      |
|       1 | k=4 sticky=0 scalefree soft trend=off dd           |        0.49  |       -0.243 |         0.317 |        -0.245 |         0.405 |        -0.392 |                 12 | False      |
|       1 | k=bic sticky=0 scalefree hard trend=126 nodd       |        0.482 |       -0.214 |         0.505 |        -0.215 |         0.493 |        -0.241 |                 11 | False      |
|       1 | k=bic sticky=0 scalefree hard trend=126 dd         |        0.472 |       -0.246 |         0.402 |        -0.21  |         0.437 |        -0.267 |                 11 | False      |
|       1 | no-regime trend=off nodd                           |        0.462 |       -0.479 |         0.523 |        -0.24  |         0.492 |        -0.551 |                 14 | False      |
|       1 | k=4 sticky=0.5 scalefree soft trend=252 nodd       |        0.439 |       -0.267 |         0.311 |        -0.277 |         0.376 |        -0.283 |                 11 | False      |
|       1 | k=4 sticky=0.5 scalefree hard trend=252 nodd       |        0.427 |       -0.263 |         0.356 |        -0.28  |         0.392 |        -0.28  |                 10 | False      |
|       1 | k=4 sticky=0 scalefree hard trend=252 nodd         |        0.422 |       -0.27  |         0.304 |        -0.245 |         0.364 |        -0.299 |                 11 | False      |
|       1 | k=4 sticky=0 scalefree hard trend=252 dd           |        0.415 |       -0.238 |         0.353 |        -0.267 |         0.384 |        -0.284 |                 11 | False      |
|       1 | k=4 sticky=0.5 scalefree hard trend=252 dd         |        0.413 |       -0.223 |         0.372 |        -0.28  |         0.392 |        -0.286 |                 11 | False      |
|       1 | k=4 sticky=0 scalefree soft trend=252 nodd         |        0.405 |       -0.272 |         0.291 |        -0.251 |         0.349 |        -0.301 |                 11 | False      |
|       1 | k=4 sticky=0.5 scalefree soft trend=252 dd         |        0.388 |       -0.231 |         0.368 |        -0.293 |         0.378 |        -0.293 |                 10 | False      |
|       1 | k=4 sticky=0.5 scalefree soft trend=off nodd       |        0.377 |       -0.365 |         0.265 |        -0.31  |         0.323 |        -0.551 |                 13 | False      |
|       1 | k=4 sticky=0.5 scalefree hard trend=off nodd       |        0.371 |       -0.374 |         0.327 |        -0.28  |         0.349 |        -0.532 |                 11 | False      |
|       1 | k=4 sticky=0 scalefree soft trend=252 dd           |        0.37  |       -0.232 |         0.323 |        -0.284 |         0.347 |        -0.284 |                 10 | False      |
|       1 | k=4 sticky=0.5 scalefree soft trend=126 nodd       |        0.364 |       -0.226 |         0.254 |        -0.297 |         0.309 |        -0.342 |                 12 | False      |
|       1 | k=4 sticky=0.5 scalefree hard trend=126 nodd       |        0.364 |       -0.227 |         0.299 |        -0.284 |         0.332 |        -0.33  |                 12 | False      |
|       1 | k=4 sticky=0.5 scalefree hard trend=off dd         |        0.363 |       -0.351 |         0.401 |        -0.227 |         0.382 |        -0.465 |                 12 | False      |
|       1 | k=4 sticky=0.5 scalefree soft trend=off dd         |        0.335 |       -0.338 |         0.411 |        -0.226 |         0.371 |        -0.459 |                 11 | False      |
|       1 | k=4 sticky=0.5 scalefree soft trend=126 dd         |        0.325 |       -0.213 |         0.354 |        -0.214 |         0.339 |        -0.293 |                 12 | False      |
|       1 | k=4 sticky=0 scalefree hard trend=126 nodd         |        0.315 |       -0.255 |         0.21  |        -0.307 |         0.263 |        -0.4   |                 12 | False      |
|       1 | k=4 sticky=0 scalefree soft trend=126 nodd         |        0.309 |       -0.241 |         0.215 |        -0.295 |         0.263 |        -0.378 |                 13 | False      |
|       1 | k=4 sticky=0.5 scalefree hard trend=126 dd         |        0.304 |       -0.229 |         0.322 |        -0.258 |         0.313 |        -0.336 |                 12 | False      |
|       1 | k=4 sticky=0 scalefree soft trend=126 dd           |        0.3   |       -0.268 |         0.323 |        -0.213 |         0.312 |        -0.329 |                 13 | False      |
|       1 | k=4 sticky=0 scalefree hard trend=126 dd           |        0.284 |       -0.279 |         0.311 |        -0.223 |         0.298 |        -0.348 |                 12 | False      |
