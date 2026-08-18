# Validation Analysis Report
### AI/ML-Based Stock Market Screening and Analysis System

**Prepared by:** Khushi Khurana
**Date:** August 18, 2026
**Repository:** https://github.com/khushikhurana15/angel-screener

---

## 1. Purpose

This report addresses the specific analysis requirements raised in resubmission feedback:

1. What percentage of crossover signals were successfully avoided based on the ML analysis
2. What percentage of the remaining (accepted) crossover signals were profitable
3. What percentage of the remaining crossover signals resulted in losses
4. Whether incorporating LTQ and other real-time market parameters improves the accuracy and profitability of the crossover strategy

Two separate validation methods are reported here, each with a different data source and each disclosed with its own limitations.

---

## 2. Methodology Overview

| | **Held-Out Test Validation** | **Live Next-Day Pipeline** |
|---|---|---|
| Data source | 4,815 historical crossover trades, 75 stocks, 45-day lookback | Live signals logged during actual trading sessions via Angel One SmartAPI |
| Split method | Stock-grouped (`GroupShuffleSplit`, symbol-level, zero overlap) | Chronological — signals resolved against real subsequent price action |
| Sample size (this report) | 1,011 signals, 15 unseen stocks | 166 unique logged signals, 59 resolved |
| What it measures | Generalization to **unseen stocks** | Generalization to **unseen future time**, on genuine live data |
| Script | `train_model.py`, `analyze_validation.py` | `data_engine.py` (`log_live_prediction`), `resolve_predictions.py` |

The assignment's "next-day validation" language specifically describes temporal generalization — training on one day, testing on a subsequent day. True two-session validation could not be completed within the resubmission timeline, so the held-out test (Section 3) is used as the primary, statistically robust result, and the live pipeline (Section 4) is reported as a smaller, real, in-progress dataset toward genuine next-day validation.

---

## 3. Held-Out Test Set Validation (Primary Result)

**Data:** 1,011 historical crossover signals from 15 stocks **never seen during model training**. Train/test split is grouped by symbol (60 training stocks, 15 test stocks), verified to have zero symbol overlap — this eliminates the risk of the model memorizing per-stock price behavior rather than learning a generalizable pattern.

### 3.1 Signal filtering

| Metric | Value |
|---|---|
| Total crossover signals evaluated | 1,011 |
| Signals ML predicted **"Avoid"** | 506 (50.0%) |
| Signals ML predicted **"Accept"** | 505 (50.0%) |

### 3.2 Quality of avoided signals — answers Q1

Of the 506 signals the model recommended avoiding:

| Outcome | Count | % |
|---|---|---|
| Were genuine losses (correctly avoided) | 392 | **77.5%** |
| Would actually have been profitable (overly cautious) | 114 | 22.5% |

**77.5% of avoided signals were correctly identified as losses that were successfully avoided.**

### 3.3 Quality of accepted signals — answers Q2 and Q3

Of the 505 signals the model recommended accepting:

| Outcome | Count | % |
|---|---|---|
| Were actually profitable | 373 | **73.9%** |
| Were actually losses | 132 | **26.1%** |

### 3.4 Does the ML filter improve profitability? — answers Q4

| Scenario | Profitable rate |
|---|---|
| Taking **every** crossover signal blindly (no ML filter) | 48.2% |
| Taking **only** ML-approved ("Accept") signals | 73.9% |
| **Improvement from using the ML filter** | **+25.7 percentage points** |

This is the core finding of this validation: incorporating LTQ-ratio and SMMA-gap-based features into a filtering model **increases the proportion of profitable trades by roughly 1.5x** relative to accepting every crossover signal without filtering — directly supporting the assignment's stated hypothesis that LTQ-based analysis can distinguish profitable trades from losing ones.

Reproducible via `python analyze_validation.py`.

---

## 4. Live Next-Day Validation Pipeline (Supplementary, In Progress)

**Data:** Real live crossover signals logged during actual trading sessions on Angel One SmartAPI, each with its real entry price and timestamp captured directly from live candle data (not wall-clock time). A separate script, `resolve_predictions.py`, is run on a subsequent occasion to check what genuinely happened after each signal and compute real accuracy.

### 4.1 Current results

| Metric | Value |
|---|---|
| Total live signals logged (raw) | 402 |
| Unique signals after deduplication | 166 |
| Signals resolved so far (opposite crossover occurred) | 59 |
| Resolved correctly (prediction matched actual outcome) | 22 |
| **Resolved accuracy** | **37.3%** |

### 4.2 A bug found and fixed during this process

The raw log initially contained 402 rows for only 166 distinct signals. Investigation showed the logger was re-recording the same unchanged crossover on every dashboard refresh cycle, since `find_latest_crossover()` returns the same result until a genuinely new crossover occurs. This was inflating the sample with duplicate rows and skewing the raw (undeduplicated) accuracy figure. A deduplication guard, keyed on `(symbol, crossover_timestamp)`, was added to `data_engine.py` to prevent this going forward; all figures in this report use the deduplicated data.

### 4.3 Honest limitations of this result

- **Small sample.** At n=59, the 95% confidence interval on the true resolution accuracy is approximately ±12–13 percentage points — this number should not be treated as a precise or stable measurement.
- **Intraday, not full-day, resolution.** Most resolved signals reversed within the same trading session (SMMA(20)/SMMA(120) crossovers are relatively slow-moving; only faster-reversing signals had resolved by the time this report was generated). This is a different, likely noisier population than the full historical daily data used in Section 3.
- **Not directly comparable to Section 3's 75.67%/73.9% figures**, which are measured on 20x the sample size using full historical daily candle data.

This section demonstrates that the full logging-and-resolution pipeline functions correctly end-to-end on genuine live data — real entry prices, real timestamps, real subsequent outcomes — and is actively accumulating results. Given more elapsed trading days, this pipeline will produce a statistically meaningful next-day accuracy figure directly comparable to what the assignment describes.

---

## 5. Summary

| Question | Answer |
|---|---|
| % of crossovers avoided | 50.0% (held-out set) |
| % of avoided signals that were genuine losses | 77.5% |
| % of accepted signals that were profitable | 73.9% |
| % of accepted signals that were losses | 26.1% |
| Does LTQ/SMMA-based filtering improve profitability? | **Yes — +25.7 percentage points** (48.2% → 73.9%) |
| Live next-day pipeline status | Built, functioning, actively logging (37.3% on n=59, early/small sample) |

---

## 6. Reproducibility

All numbers in this report are generated by scripts in the repository, not hand-calculated:

```bash
python train_model.py          # trains model with stock-grouped split, prints accuracy/confusion matrix
python analyze_validation.py   # produces Section 3 numbers, saves validation_analysis_summary.csv
python resolve_predictions.py  # produces Section 4 numbers from live_predictions_log.csv
```