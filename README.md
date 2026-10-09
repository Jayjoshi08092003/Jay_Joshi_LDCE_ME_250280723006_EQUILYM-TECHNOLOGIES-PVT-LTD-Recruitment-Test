# Assignment 4 — Auto-Analytics Engine (Automated Insight Generation)

Ingests a district-level healthcare CSV and automatically produces trends, outliers,
threshold breaches and correlations as structured, human-readable insights.
General-purpose: any `month, district, <numeric indicators…>` CSV works; no district
or indicator names appear in the code, and every number in every sentence is computed.

## Files
| File | Purpose |
|---|---|
| `insights_engine.py` | Pure-Python engine (load/validate, detectors, insight generator, CLI) |
| `app.py` | Streamlit UI (filters, sliders, insight table, charts) |
| `healthcare_districts.csv` | Sample dataset (6 districts × 2 months) |
| `output/insights.json`, `insights.csv`, `correlation_matrix.csv` | Sample outputs |
| `smoke_test.py` | Drives the UI headlessly (filters + sliders) |
| `requirements.txt` | Dependencies |

## Run
```bash
Create Envoirement First in Python 3.14 then
pip install -r requirements.txt
streamlit run app.py                                   # interactive UI
python insights_engine.py healthcare_districts.csv     # headless; writes ./output
python smoke_test.py                                   # optional UI smoke test
```
Upload your own CSV from the sidebar. Required columns: `month` (e.g. `2026-07`
or `2026-07-01`) and `district`; every other numeric column is treated as an indicator.

## How each part works
- **A – Loading/validation:** `head()`, `info()`, per-column missing counts (shown in the UI
  expander and printed by the CLI). Also checks duplicates, bad months, non-numeric columns,
  and < 3 months / < 10 districts. Missing values are skipped, never imputed.
  Filters: district, month, indicator (live).
- **B – Trends:** per `(district, indicator)`, `pct_change = (current − previous) / previous × 100`
  against the previous recorded month; `is_significant` if `|pct_change| ≥ threshold` (default 10%).
- **C – Outliers:** IQR rule (default, multiplier 1.5) or Z-score (default 3), pooled across all
  districts and months per indicator. Output has district, indicator, month, value.
- **D – Correlation:** Pearson matrix (`DataFrame.corr()`), pairs with `|r| ≥ 0.70` flagged.
- **E – Insights:** fields `insight_id, type, indicator, entity, period, metric, prev_value, change,
  severity, explanation`. Types: `trend`, `outlier`, `correlation`, `threshold_breach`.
  Mapping to the §4.1 CSV header: `metric` = `value`, `prev_value` = `prev_value`
  (for outliers: the median/mean; for breaches: the benchmark), `change` = `change_pct`
  (% change, % deviation, or — for correlations — `metric` holds r).
- **F – UI:** insight list, severity bar chart, correlation heatmap, per-district line charts.

### Severity is derived, not hard-coded
`ratio = observed magnitude ÷ the threshold that triggered the insight`
(|%change| ÷ trend threshold, IQR-distance ÷ multiplier, |z| ÷ z-threshold, |deviation| ÷ tolerance).
`High` if ratio ≥ 1.75, `Medium` if ≥ 1.25, else `Low` — both cut-offs are sliders.
Correlations use p-value instead: `High` if p < α/5, `Medium` if p < α, else `Low`.

### threshold_breach
Compares each value with a benchmark (default: the indicator's median; editable per indicator,
with an alert direction of both / below / above) and fires when the deviation exceeds the
tolerance (default ±25%). Set direction to `above` for "lower is better" indicators such as
`high_risk_cases`, otherwise unusually *good* values are flagged too.

## Honest limitations
1. **Correlations are fragile.** The sample has 6 districts × 2 months = 12 rows, and the rows are
   not independent (the same district appears twice). `r` values such as −0.93 for
   ANC vs high-risk cases are driven largely by one district (Mehsana). The engine therefore
   states this in the UI and in each correlation insight, and caps severity at `Medium`
   whenever there are fewer than 10 districts. The full matrix is shown unfiltered.
2. **Z-score cannot reach 3 on 12 points.** With n = 12 (sample std), |z| can never exceed
   (n−1)/√n ≈ 3.18, and Mehsana's ANC of 42 has z ≈ −2.83, so Z-score ≥ 3 flags nothing here.
   IQR is the default for that reason; lower the Z slider to ~2.5 to flag it. The assignment's
   "3.1σ below state mean (76)" example does not reproduce from this dataset.
3. Mehsana ANC shows −50.0% month over month (84 → 42), not −44.7%: the engine compares with the
   district's own previous month, whereas −44.7% is a comparison to a different reference.
4. Trends compare consecutive recorded months only; with two months there is one change per pair.
5.Limitation: only 6 district(s) × 2 month(s) = 12 rows in view. With so few observations Pearson r is fragile (a single extreme row can drive it), so correlations are capped at Medium severity and should be read as indicative only.

Severity Explanation
1. Mehsana (High Severity Breaches)

Mehsana has recorded significant high-severity breaches across two critical healthcare metrics. First, the ANC Coverage Drop fell sharply from 84 to 42, representing a 50.0% decline. Because this drop heavily exceeds the threshold, its trend severity is flagged as High. Second, the district experienced a sharp High Risk Cases Spike, where figures jumped from 11 to 28 (a 154.5% increase). This rapid rise also surpasses safety thresholds, confirming a High severity status.

2. Ahmedabad (High & Medium Severity Breaches)

Ahmedabad shows concerning trends classified under high and medium severity. The district's ANC Coverage Drop went down from 85 to 69, marking an 18.8% decrease, which triggers a High severity alert. Alongside this, High Risk Cases Increased, rising from 10 to 13 (a 30.0% increase). This upward shift is also categorized under High severity.

3. Bhavnagar (Medium Severity Breach)

Bhavnagar noted a minor High Risk Cases Drop, with numbers decreasing from 17 to 15 (an 11.8% change). While this shift exceeds the standard baseline for tracking, it falls below the medium severity threshold. Therefore, this change technically registers as a Low severity trend.

4. Other Districts (Low Severity / No Breach)

Districts such as Surat, Vadodara, and Rajkot maintain steady healthcare metrics. Their month-over-month shifts remain well below the standard baseline, keeping their overall trend severity at Low.

