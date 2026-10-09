"""Streamlit UI for the Auto-Analytics Engine.   Run:  streamlit run app.py"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

import insights_engine as eng

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "healthcare_districts.csv")
SEV_COLORS = {"Low": "#2e9e5b", "Medium": "#f0a30a", "High": "#d33c3c"}

st.set_page_config(page_title="Auto-Analytics Engine", layout="wide")
st.title("Auto-Analytics Engine")
st.caption("Upload a district-level indicator CSV (columns: month, district, then numeric indicators) "
           "and get trends, outliers, threshold breaches and correlations as plain-English insights.")

# ------------------------------------------------------------------ data ----
up = st.sidebar.file_uploader("CSV file", type="csv")
try:
    raw = eng.load_csv(up if up else DEFAULT_CSV)
    df, indicators, issues = eng.validate(raw)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load data: {e}")
    st.stop()

# --------------------------------------------------------------- filters ----
st.sidebar.header("Filters")
districts = st.sidebar.multiselect("District", sorted(df["district"].unique()),
                                   default=sorted(df["district"].unique()))
months = st.sidebar.multiselect("Month", sorted(df["month"].unique()),
                                default=sorted(df["month"].unique()))
sel_ind = st.sidebar.multiselect("Indicator", indicators, default=indicators)

# ------------------------------------------------------------- thresholds ----
st.sidebar.header("Thresholds")
trend_thr = st.sidebar.slider("Trend: significant |% change| ≥", 1.0, 50.0, 10.0, 0.5)
method = st.sidebar.radio("Outlier method", ["iqr", "zscore"], horizontal=True)
iqr_mult = st.sidebar.slider("IQR multiplier", 0.5, 3.0, 1.5, 0.1, disabled=method != "iqr")
z_thr = st.sidebar.slider("Z-score threshold", 1.0, 4.0, 3.0, 0.1, disabled=method != "zscore")
corr_thr = st.sidebar.slider("Correlation: |r| ≥", 0.30, 0.99, 0.70, 0.01)
min_d = st.sidebar.slider("Min districts for a 'reliable' correlation", 3, 20, 10)
alpha = st.sidebar.slider("Correlation p-value α (severity)", 0.01, 0.20, 0.05, 0.01)
tol = st.sidebar.slider("Benchmark tolerance (±%)", 5.0, 100.0, 25.0, 1.0)
st.sidebar.subheader("Severity cut-offs")
st.sidebar.caption("Severity = how many times the triggering threshold was exceeded.")
sev_med = st.sidebar.slider("Medium at ≥ × threshold", 1.0, 3.0, 1.25, 0.05)
sev_high = st.sidebar.slider("High at ≥ × threshold", 1.0, 5.0, 1.75, 0.05)
if sev_high < sev_med:
    st.sidebar.warning("High cut-off is below Medium; using Medium's value for both.")
    sev_high = sev_med

with st.sidebar.expander("Threshold-breach benchmarks"):
    st.caption("Default benchmark = indicator median across all data.")
    bench = {}
    for ind in indicators:
        c1, c2 = st.columns(2)
        v = c1.number_input(f"{ind}", value=float(df[ind].median()), key=f"b_{ind}")
        d = c2.selectbox("alert", ["both", "below", "above"], key=f"d_{ind}")
        bench[ind] = {"value": v, "direction": d}

cfg = eng.Config(trend_threshold=trend_thr, outlier_method=method, iqr_mult=iqr_mult,
                 z_threshold=z_thr, corr_threshold=corr_thr, min_districts_for_corr=min_d,
                 alpha=alpha, benchmark_tolerance=tol, sev_medium=sev_med, sev_high=sev_high)

# ------------------------------------------------------- Part A: validation ----
with st.expander("Data loading & validation (head / info / missing values)", expanded=False):
    st.code(eng.validation_report(raw), language="text")
    for i in issues:
        st.warning(i)

# ----------------------------------------------------------------- analysis ----
if not (districts and months and sel_ind):
    st.info("Select at least one district, month and indicator.")
    st.stop()

# Trends/outliers use full history (a trend needs the previous month; outliers need
# the full distribution). Filters apply to what is *shown*. Correlations are
# computed on the filtered rows so the heatmap and insights agree.
fdf = df[df["district"].isin(districts) & df["month"].isin(months)]
insights_all, art = eng.generate_insights(df, indicators, cfg, bench, corr_df=fdf)
ins = eng.filter_insights(insights_all, districts, months, sel_ind)
counts = eng.severity_counts(ins)

# --------------------------------------------------------------- summary ----
k = st.columns(4)
k[0].metric("Insights", len(ins))
k[1].metric("High", int(counts["High"]))
k[2].metric("Medium", int(counts["Medium"]))
k[3].metric("Low", int(counts["Low"]))

tab_ins, tab_charts, tab_corr, tab_detail = st.tabs(
    ["Insights", "Charts", "Correlation", "Detector detail"])

with tab_ins:
    types = st.multiselect("Insight type", ["trend", "outlier", "correlation", "threshold_breach"],
                           default=["trend", "outlier", "correlation", "threshold_breach"])
    sevs = st.multiselect("Severity", eng.SEVERITIES, default=eng.SEVERITIES)
    view = ins[ins["type"].isin(types) & ins["severity"].isin(sevs)]
    st.dataframe(view, width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button("Download insights CSV", view.to_csv(index=False), "insights.csv", "text/csv")
    c2.download_button("Download insights JSON", view.to_json(orient="records", indent=2),
                       "insights.json", "application/json")

with tab_charts:
    left, right = st.columns([1, 2])
    bar = counts.reset_index()
    bar.columns = ["severity", "count"]
    left.plotly_chart(
        px.bar(bar, x="severity", y="count", color="severity", color_discrete_map=SEV_COLORS,
               title="Insights by severity", text="count").update_layout(showlegend=False),
        width="stretch")
    long = fdf.melt(id_vars=["month", "district"], value_vars=sel_ind,
                    var_name="indicator", value_name="value").dropna()
    fig = px.line(long, x="month", y="value", color="district", markers=True,
                  facet_col="indicator", facet_col_wrap=2, title="Per-district trend")
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    right.plotly_chart(fig, width="stretch")

with tab_corr:
    use = [i for i in sel_ind if i in fdf.columns]
    if len(use) < 2 or len(fdf) < 3:
        st.info("Need ≥ 2 indicators and ≥ 3 rows to compute correlations.")
    else:
        corr = eng.correlation_matrix(fdf, use)
        st.plotly_chart(
            px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                      title="Pearson correlation matrix"), width="stretch")
        if fdf["district"].nunique() < min_d:
            st.warning(
                f"Limitation: only {fdf['district'].nunique()} district(s) × "
                f"{fdf['month'].nunique()} month(s) = {len(fdf)} rows in view. With so few "
                "observations Pearson r is fragile (a single extreme row can drive it), "
                "so correlations are capped at Medium severity and should be read as "
                "indicative only.")
        _, pairs = eng.detect_correlations(fdf, use, cfg)
        st.write(f"Pairs with |r| ≥ {corr_thr:.2f}:")
        st.dataframe(pairs, width="stretch", hide_index=True)
        st.download_button("Download correlation matrix CSV", corr.to_csv(),
                           "correlation_matrix.csv", "text/csv")

with tab_detail:
    st.subheader("All month-over-month changes")
    t = art["trends"]
    t = t[t["district"].isin(districts) & t["period"].isin(months) & t["indicator"].isin(sel_ind)]
    st.dataframe(t.drop(columns="ratio"), width="stretch", hide_index=True)
    st.subheader("Flagged trends")
    st.dataframe(t[t["is_significant"]].drop(columns="ratio"), width="stretch", hide_index=True)
    st.subheader("Outliers")
    o = art["outliers"]
    o = o[o["district"].isin(districts) & o["period"].isin(months) & o["indicator"].isin(sel_ind)]
    st.dataframe(o[["district", "indicator", "period", "value", "method", "score"]],
                 width="stretch", hide_index=True)
