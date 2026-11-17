# Data Quality & Integrity Report

**Project:** Reinforcement Learning with RNN-Based Market State Representation for Algorithmic Trading  
**Dataset:** National Stock Exchange of India (NSE) - 5 Target Equities  
**Source:** Yahoo Finance (yfinance)  
**Lookback Window:** 5 Years Historical Daily Data  

---

## 1. Executive Summary

A rigorous data hygiene and structural integrity check was conducted across all downloaded stocks.
The validation protocol evaluated:
1. **Missing values (NaN / Null)** across Open, High, Low, Close, and Volume.
2. **Duplicate timestamps** and calendar continuity.
3. **Physical OHLC validity:** `High >= Low`, `High >= max(Open, Close)`, and `Low <= min(Open, Close)`.
4. **Price positivity:** `Open, High, Low, Close > 0`.
5. **Volume anomalies:** negative volume detection and zero-volume adjustments.
6. **Chronological monotonic ordering**.

---

## 2. Stock-by-Stock Quality Audit Table

| Symbol | Raw Rows | Cleaned Rows | Start Date | End Date | Missing Values | Duplicate Dates | Invalid OHLC | Non-Positive Prices | Zero / Neg Volumes |
|---|---|---|---|---|---|---|---|---|---|
| **RELIANCE** | 1236 | 1236 | 2021-11-04 | 2026-11-04 | 0 | 0 | 0 | 0 | 2 zero, 0 neg |
| **TCS** | 1236 | 1236 | 2021-11-04 | 2026-11-04 | 0 | 0 | 0 | 0 | 2 zero, 0 neg |
| **INFY** | 1236 | 1236 | 2021-11-04 | 2026-11-04 | 0 | 0 | 0 | 0 | 2 zero, 0 neg |
| **HDFCBANK** | 1236 | 1236 | 2021-11-04 | 2026-11-04 | 0 | 0 | 0 | 0 | 2 zero, 0 neg |
| **ITC** | 1236 | 1236 | 2021-11-04 | 2026-11-04 | 0 | 0 | 0 | 0 | 1 zero, 0 neg |

---

## 3. Data Cleaning Actions & Academic Methodology

### RELIANCE
- Adjusted 2 zero volume rows to baseline minimum volume (1705656) to prevent log/div zero errors.

### TCS
- Adjusted 2 zero volume rows to baseline minimum volume (154895) to prevent log/div zero errors.

### INFY
- Adjusted 2 zero volume rows to baseline minimum volume (371966) to prevent log/div zero errors.

### HDFCBANK
- Adjusted 2 zero volume rows to baseline minimum volume (1096808) to prevent log/div zero errors.

### ITC
- Adjusted 1 zero volume rows to baseline minimum volume (1142130) to prevent log/div zero errors.

---

## 4. Verification Statement

- **Zero Silently Discarded Records:** All adjustments were explicitly logged and audited.
- **Chronological Integrity:** Every series is strictly monotonic and sorted from past to present.
- **Academic Justification:** Price carry-forward (forward-fill) preserves the real market state where untraded assets retain their last settled price until subsequent market auctions.
