"""
Auto-Analytics Engine -- general-purpose insight generation for district-level
indicator data (long-form: one row per (district, month)).

Nothing here is specific to any district or indicator: every number in every
explanation is computed from the data, and every threshold lives in `Config`
(which the Streamlit UI exposes as sliders).

Run headless:  python insights_engine.py healthcare_districts.csv --out output
"""
from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

ID_COLS = ("month", "district")
SEVERITIES = ["Low", "Medium", "High"]
INSIGHT_COLUMNS = [
    "insight_id", "type", "indicator", "entity", "period",
    "metric", "prev_value", "change", "severity", "explanation",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # Part B - trends
    trend_threshold: float = 10.0        # % ; |pct_change| >= this -> significant
    # Part C - outliers
    outlier_method: str = "iqr"          # "iqr" | "zscore"
    iqr_mult: float = 1.5
    z_threshold: float = 3.0
    # Part D - correlation
    corr_threshold: float = 0.70
    min_districts_for_corr: int = 10     # below this, correlations are "fragile"
    alpha: float = 0.05                  # p-value cut-off used for severity
    # threshold_breach
    benchmark_tolerance: float = 25.0    # % deviation from benchmark allowed
    # Severity: ratio = (observed magnitude / the threshold that triggered it)
    sev_medium: float = 1.25             # ratio >= -> Medium
    sev_high: float = 1.75               # ratio >= -> High


def severity_from_ratio(ratio: float, cfg: Config) -> str:
    if ratio >= cfg.sev_high:
        return "High"
    if ratio >= cfg.sev_medium:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------- #
# Part A - loading & validation
# --------------------------------------------------------------------------- #
def load_csv(src) -> pd.DataFrame:
    df = pd.read_csv(src)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate(df: pd.DataFrame):
    """Return (clean_df, indicator_columns, list_of_issue_strings)."""
    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {missing}")

    issues: list[str] = []
    df = df.copy()
    df["district"] = df["district"].astype(str).str.strip()

    parsed = pd.to_datetime(df["month"].astype(str), errors="coerce")
    bad = int(parsed.isna().sum())
    if bad:
        issues.append(f"{bad} row(s) dropped: unparseable month value.")
    df["month"] = parsed.dt.strftime("%Y-%m")
    df = df.dropna(subset=["month"])

    candidates = [c for c in df.columns if c not in ID_COLS]
    indicators = []
    for c in candidates:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().any():
            indicators.append(c)
        else:
            issues.append(f"Column '{c}' ignored: no numeric values.")
    df = df[list(ID_COLS) + indicators]

    dups = int(df.duplicated(["district", "month"]).sum())
    if dups:
        issues.append(f"{dups} duplicate (district, month) row(s); kept the last.")
        df = df.drop_duplicates(["district", "month"], keep="last")

    n_missing = int(df[indicators].isna().sum().sum())
    if n_missing:
        issues.append(f"{n_missing} missing indicator value(s); they are skipped, not imputed.")

    df = df.sort_values(["district", "month"]).reset_index(drop=True)

    per_district = df.groupby("district")["month"].nunique()
    if (per_district < 3).any():
        issues.append("Some districts have < 3 months of data (recommended minimum for trend detection).")
    if df["district"].nunique() < 10:
        issues.append(
            f"Only {df['district'].nunique()} districts (recommended >= 10): "
            "correlation estimates are fragile."
        )
    return df, indicators, issues


def validation_report(df: pd.DataFrame) -> str:
    """head(), info() and per-column missing count, as one printable string."""
    buf = io.StringIO()
    df.info(buf=buf)
    return (
        "=== head() ===\n" + df.head().to_string()
        + "\n\n=== info() ===\n" + buf.getvalue()
        + "\n=== missing values per column ===\n" + df.isna().sum().to_string()
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def pretty(name: str) -> str:
    """anc_coverage -> 'ANC Coverage' (short tokens are treated as acronyms)."""
    return " ".join(w.upper() if len(w) <= 3 else w.capitalize() for w in name.split("_"))


def _fmt(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".") if abs(x) < 1000 else f"{x:,.0f}"


def _long(df: pd.DataFrame, indicators) -> pd.DataFrame:
    return (
        df.melt(id_vars=list(ID_COLS), value_vars=list(indicators),
                var_name="indicator", value_name="value")
        .dropna(subset=["value"])
    )


def _month_gap(a: str, b: str) -> int:
    return (pd.Period(b, "M") - pd.Period(a, "M")).n


# --------------------------------------------------------------------------- #
# Part B - trend detection
# --------------------------------------------------------------------------- #
def detect_trends(df, indicators, cfg: Config) -> pd.DataFrame:
    long = _long(df, indicators).sort_values(["district", "indicator", "month"])
    g = long.groupby(["district", "indicator"])
    long["prev_value"] = g["value"].shift(1)
    long["prev_period"] = g["month"].shift(1)
    t = long.dropna(subset=["prev_value"]).copy()
    t = t[t["prev_value"] != 0]
    t["pct_change"] = (t["value"] - t["prev_value"]) / t["prev_value"].abs() * 100
    t["ratio"] = t["pct_change"].abs() / cfg.trend_threshold
    t["is_significant"] = t["pct_change"].abs() >= cfg.trend_threshold
    cols = ["district", "indicator", "prev_period", "month", "prev_value",
            "value", "pct_change", "is_significant", "ratio"]
    return t[cols].rename(columns={"month": "period"}).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Part C - outlier detection (pooled across all districts and months)
# --------------------------------------------------------------------------- #
def detect_outliers(df, indicators, cfg: Config) -> pd.DataFrame:
    rows = []
    long = _long(df, indicators)
    for ind, grp in long.groupby("indicator"):
        v = grp["value"]
        if len(v) < 4:
            continue
        if cfg.outlier_method == "iqr":
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - cfg.iqr_mult * iqr, q3 + cfg.iqr_mult * iqr
            ref = v.median()
            for _, r in grp[(v < lo) | (v > hi)].iterrows():
                dist = ((q1 - r.value) if r.value < lo else (r.value - q3)) / iqr
                rows.append(dict(
                    district=r.district, indicator=ind, period=r.month, value=r.value,
                    reference=ref, score=dist, lower=lo, upper=hi,
                    ratio=dist / cfg.iqr_mult, method="iqr",
                    direction="below" if r.value < lo else "above"))
        else:
            mu, sd = v.mean(), v.std(ddof=1)
            if not sd or np.isnan(sd):
                continue
            z = (v - mu) / sd
            for idx in grp.index[z.abs() >= cfg.z_threshold]:
                r = grp.loc[idx]
                rows.append(dict(
                    district=r.district, indicator=ind, period=r.month, value=r.value,
                    reference=mu, score=float(z[idx]), lower=mu - cfg.z_threshold * sd,
                    upper=mu + cfg.z_threshold * sd,
                    ratio=abs(z[idx]) / cfg.z_threshold, method="zscore",
                    direction="below" if z[idx] < 0 else "above"))
    cols = ["district", "indicator", "period", "value", "reference", "score",
            "lower", "upper", "ratio", "method", "direction"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# threshold_breach - deviation from a benchmark (default: indicator median)
# --------------------------------------------------------------------------- #
def default_benchmarks(df, indicators) -> dict:
    return {i: {"value": float(df[i].median()), "direction": "both"} for i in indicators}


def detect_breaches(df, indicators, cfg: Config, benchmarks=None) -> pd.DataFrame:
    benchmarks = benchmarks or default_benchmarks(df, indicators)
    rows = []
    long = _long(df, indicators)
    for ind, grp in long.groupby("indicator"):
        b = benchmarks.get(ind)
        if not b or b["value"] == 0:
            continue
        for _, r in grp.iterrows():
            dev = (r.value - b["value"]) / abs(b["value"]) * 100
            if abs(dev) < cfg.benchmark_tolerance:
                continue
            if (b["direction"] == "below" and dev > 0) or (b["direction"] == "above" and dev < 0):
                continue
            rows.append(dict(district=r.district, indicator=ind, period=r.month,
                             value=r.value, benchmark=b["value"], deviation_pct=dev,
                             ratio=abs(dev) / cfg.benchmark_tolerance))
    cols = ["district", "indicator", "period", "value", "benchmark", "deviation_pct", "ratio"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Part D - correlation detection
# --------------------------------------------------------------------------- #
def correlation_matrix(df, indicators) -> pd.DataFrame:
    return df[list(indicators)].corr(method="pearson")


def detect_correlations(df, indicators, cfg: Config):
    """Returns (corr_matrix, flagged_pairs_df)."""
    inds = list(indicators)
    corr = correlation_matrix(df, inds)
    n_districts = df["district"].nunique()
    fragile = n_districts < cfg.min_districts_for_corr
    rows = []
    for i, a in enumerate(inds):
        for b in inds[i + 1:]:
            pair = df[[a, b]].dropna()
            if len(pair) < 3 or pair[a].std() == 0 or pair[b].std() == 0:
                continue
            r, p = stats.pearsonr(pair[a], pair[b])
            if abs(r) < cfg.corr_threshold:
                continue
            if p < cfg.alpha / 5:
                sev = "High"
            elif p < cfg.alpha:
                sev = "Medium"
            else:
                sev = "Low"
            if fragile and sev == "High":
                sev = "Medium"          # never "High" on a fragile sample
            rows.append(dict(a=a, b=b, r=r, p_value=p, n=len(pair),
                             n_districts=n_districts, fragile=fragile, severity=sev,
                             period=f"{df['month'].min()}..{df['month'].max()}"))
    cols = ["a", "b", "r", "p_value", "n", "n_districts", "fragile", "severity", "period"]
    return corr, pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Part E - insight generation
# --------------------------------------------------------------------------- #
def generate_insights(df, indicators, cfg: Config, benchmarks=None, corr_df=None):
    """Build the unified insight table. `corr_df` lets the UI base correlations
    on a filtered subset while trends/outliers use the full history."""
    trends = detect_trends(df, indicators, cfg)
    outliers = detect_outliers(df, indicators, cfg)
    breaches = detect_breaches(df, indicators, cfg, benchmarks)
    corr, pairs = detect_correlations(corr_df if corr_df is not None else df, indicators, cfg)

    rows = []

    # --- trends --------------------------------------------------------------
    for r in trends[trends["is_significant"]].itertuples():
        gap = _month_gap(r.prev_period, r.period)
        when = "the previous month" if gap == 1 else f"the previous recorded month ({r.prev_period})"
        verb = "dropped" if r.pct_change < 0 else "rose"
        rows.append(dict(
            type="trend", indicator=r.indicator, entity=r.district, period=r.period,
            metric=r.value, prev_value=r.prev_value, change=round(r.pct_change, 2),
            severity=severity_from_ratio(r.ratio, cfg), _rank=r.ratio,
            explanation=(f"{pretty(r.indicator)} in {r.district} {verb} by "
                         f"{abs(r.pct_change):.1f}% compared to {when} "
                         f"({_fmt(r.prev_value)} -> {_fmt(r.value)}), exceeding the "
                         f"{_fmt(cfg.trend_threshold)}% significant-change threshold.")))

    # --- outliers ------------------------------------------------------------
    for r in outliers.itertuples():
        dev = (r.value - r.reference) / abs(r.reference) * 100 if r.reference else np.nan
        if r.method == "iqr":
            how = (f"{abs(r.score):.1f} IQRs {r.direction} the {'lower' if r.direction=='below' else 'upper'} "
                   f"quartile, outside the IQR fence ({_fmt(r.lower)}-{_fmt(r.upper)}); "
                   f"median across all districts/months is {_fmt(r.reference)}")
        else:
            how = (f"{abs(r.score):.1f} standard deviations {r.direction} the mean "
                   f"({_fmt(r.reference)}) across all districts/months")
        rows.append(dict(
            type="outlier", indicator=r.indicator, entity=r.district, period=r.period,
            metric=r.value, prev_value=round(r.reference, 2),
            change=None if np.isnan(dev) else round(dev, 2),
            severity=severity_from_ratio(r.ratio, cfg), _rank=r.ratio,
            explanation=f"{r.district}'s {pretty(r.indicator)} of {_fmt(r.value)} in {r.period} is {how}; flagged for review."))

    # --- threshold breaches ---------------------------------------------------
    for r in breaches.itertuples():
        side = "below" if r.deviation_pct < 0 else "above"
        rows.append(dict(
            type="threshold_breach", indicator=r.indicator, entity=r.district, period=r.period,
            metric=r.value, prev_value=round(r.benchmark, 2), change=round(r.deviation_pct, 2),
            severity=severity_from_ratio(r.ratio, cfg), _rank=r.ratio,
            explanation=(f"{pretty(r.indicator)} in {r.district} ({r.period}) is {_fmt(r.value)}, "
                         f"{abs(r.deviation_pct):.1f}% {side} the benchmark of {_fmt(r.benchmark)} "
                         f"(allowed tolerance +/-{_fmt(cfg.benchmark_tolerance)}%).")))

    # --- correlations ---------------------------------------------------------
    for r in pairs.itertuples():
        direction = "negatively" if r.r < 0 else "positively"
        caveat = (f" Fragile: only {r.n_districts} district(s) and {r.n} observations "
                  f"(< {cfg.min_districts_for_corr} districts), so treat as indicative, not conclusive."
                  if r.fragile else "")
        rows.append(dict(
            type="correlation", indicator=f"{r.a}:{r.b}", entity="All districts", period=r.period,
            metric=round(r.r, 3), prev_value=None, change=None, severity=r.severity,
            _rank=abs(r.r),
            explanation=(f"{pretty(r.a)} and {pretty(r.b)} are strongly {direction} correlated "
                         f"(r = {r.r:.2f}, {'p < 0.001' if r.p_value < 0.001 else f'p = {r.p_value:.3f}'}, n = {r.n}), above the "
                         f"|r| >= {cfg.corr_threshold:.2f} threshold.{caveat}")))

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=INSIGHT_COLUMNS)
    else:
        out["_sev"] = out["severity"].map({s: i for i, s in enumerate(SEVERITIES)})
        out = out.sort_values(["_sev", "_rank"], ascending=[False, False]).reset_index(drop=True)
        out.insert(0, "insight_id", [f"INS-{i+1:04d}" for i in range(len(out))])
        out = out[INSIGHT_COLUMNS]

    artifacts = dict(trends=trends, outliers=outliers, breaches=breaches,
                     corr_matrix=corr, corr_pairs=pairs)
    return out, artifacts


def filter_insights(ins, districts=None, months=None, indicators=None) -> pd.DataFrame:
    """Apply the UI filters. Cross-district (correlation) insights are kept when
    both of their indicators are selected."""
    m = pd.Series(True, index=ins.index)
    if districts is not None:
        m &= ins["entity"].isin(list(districts) + ["All districts"])
    if months is not None:
        is_corr = ins["type"] == "correlation"
        m &= is_corr | ins["period"].isin(months)
    if indicators is not None:
        sel = set(indicators)
        m &= ins["indicator"].apply(lambda s: all(p in sel for p in s.split(":")))
    return ins[m].reset_index(drop=True)


def severity_counts(ins) -> pd.Series:
    return ins["severity"].value_counts().reindex(SEVERITIES, fill_value=0)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Auto-Analytics Engine (headless)")
    ap.add_argument("csv")
    ap.add_argument("--out", default="output")
    ap.add_argument("--trend", type=float, default=10.0)
    ap.add_argument("--method", choices=["iqr", "zscore"], default="iqr")
    ap.add_argument("--corr", type=float, default=0.70)
    a = ap.parse_args()

    cfg = Config(trend_threshold=a.trend, outlier_method=a.method, corr_threshold=a.corr)
    raw = load_csv(a.csv)
    df, inds, issues = validate(raw)
    print(validation_report(raw))
    for i in issues:
        print("[validation]", i)

    ins, art = generate_insights(df, inds, cfg)
    os.makedirs(a.out, exist_ok=True)
    ins.to_csv(f"{a.out}/insights.csv", index=False)
    art["corr_matrix"].to_csv(f"{a.out}/correlation_matrix.csv")
    with open(f"{a.out}/insights.json", "w") as f:
        json.dump(json.loads(ins.to_json(orient="records")), f, indent=2)
    print(f"\n{len(ins)} insights written to {a.out}/  |  severity counts:")
    print(severity_counts(ins).to_string())
    print()
    for r in ins.itertuples():
        print(f"{r.insight_id} [{r.severity:6}] {r.type:16} {r.explanation}")


if __name__ == "__main__":
    main()
