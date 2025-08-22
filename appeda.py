# streamlit_app.py
# Auto-tailored Streamlit dashboard that adapts to your uploaded dataset.
# - Reads Excel/CSV
# - Profiles dtypes, cardinality, missingness
# - Detects likely target columns and datetime fields
# - Builds the exact dashboard blocks from your checklist — only if data supports them
# - Uses your real column names dynamically; no hardcoded assumptions

import io
import re
import math
import json
import string
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go

# ----------------------------- Page config ----------------------------- #
st.set_page_config(page_title="Auto Dashboard — Upload & Explore", layout="wide")
st.title("📊 Auto-Tailored Dashboard (Upload → Components)")

# ----------------------------- Helpers ----------------------------- #
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(file_bytes))
    elif filename.lower().endswith('.csv'):
        # Try utf-8, then fallback to cp1252
        try:
            return pd.read_csv(io.BytesIO(file_bytes))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(file_bytes), encoding='cp1252')
    else:
        # Attempt pandas auto-detect for delimited files
        return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def profile_data(df: pd.DataFrame):
    prof = {}
    # Datetime inference: try safe parse on columns with 'date' in the name or datetime dtype
    datetime_cols, parsed_cols = [], []
    for c in df.columns:
        s = df[c]
        if np.issubdtype(s.dtype, np.datetime64):
            datetime_cols.append(c)
            continue
        name_l = c.lower()
        if any(tok in name_l for tok in ['date', 'time', 'timestamp']):
            parsed = pd.to_datetime(s, errors='coerce', utc=False)
            if parsed.notna().sum() >= max(10, 0.2*len(df)):
                df[c] = parsed
                parsed_cols.append(c)
                datetime_cols.append(c)
    # Numeric & categorical
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    object_cols  = [c for c in df.columns if df[c].dtype == 'object']
    boollike_cols = []
    for c in df.columns:
        s = df[c].dropna().unique()
        if len(s) == 0: 
            continue
        # binary if subset of {0,1,True,False} OR exactly two distinct numerical values 0/1
        lowered = {str(x).strip().lower() for x in s}
        if lowered.issubset({'0','1','true','false','yes','no'}) or (pd.api.types.is_numeric_dtype(df[c]) and len(s) <= 2):
            boollike_cols.append(c)
    # Text columns: long-ish strings
    def likely_text(series: pd.Series) -> bool:
        if series.dtype != 'object':
            return False
        sample = series.dropna().astype(str).head(500)
        if sample.empty:
            return False
        avg_len = sample.str.len().mean()
        return avg_len >= 30  # heuristic

    text_cols = [c for c in object_cols if likely_text(df[c])]
    # Categorical = objects with manageable cardinality and not text
    categorical_cols = []
    for c in object_cols:
        if c in text_cols:
            continue
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 50:
            categorical_cols.append(c)
    # Geo guess
    lat_candidates = [c for c in df.columns if c.lower() in {'lat','latitude'}]
    lon_candidates = [c for c in df.columns if c.lower() in {'lon','lng','longitude'}]
    geo = {
        'lat': lat_candidates[0] if lat_candidates else None,
        'lon': lon_candidates[0] if lon_candidates else None,
        'region_like': [c for c in df.columns if any(k in c.lower() for k in ['region','governorate','district','city','country'])]
    }
    # Missingness & cardinality
    missingness = df.isna().mean().sort_values(ascending=False)
    cardinality = df.nunique(dropna=True).sort_values(ascending=False)
    # Likely targets by name or binary
    name_patterns = re.compile(r"^(target|label|churn|outcome|replaced_.*|is_.*|will_.*)$", re.IGNORECASE)
    likely_targets = []
    for c in df.columns:
        if name_patterns.match(c):
            likely_targets.append(c)
    # Add any other binary columns
    for c in boollike_cols:
        if c not in likely_targets:
            likely_targets.append(c)
    # Preferred date fields
    date_priority = []
    for prefer in ['tag_date','untag_date']:
        if prefer in df.columns and pd.api.types.is_datetime64_any_dtype(df[prefer]):
            date_priority.append(prefer)
    if not date_priority:
        date_priority = datetime_cols
    prof.update(dict(
        df=df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        text_cols=text_cols,
        boollike_cols=boollike_cols,
        geo=geo,
        missingness=missingness,
        cardinality=cardinality,
        likely_targets=likely_targets,
        preferred_date=date_priority[0] if date_priority else None,
        parsed_cols=parsed_cols,
    ))
    return prof

# Utility: safe Top-N categories
@st.cache_data(show_spinner=False)
def top_categories(series: pd.Series, n:int=10):
    vc = series.astype('object').fillna('∅ Missing').value_counts(dropna=False)
    top = vc.head(n)
    other = vc.iloc[n:].sum()
    if other > 0:
        top['Other'] = other
    return top

# Cramér's V
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    from scipy.stats import chi2_contingency
    table = pd.crosstab(x, y)
    if table.size == 0:
        return np.nan
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    return math.sqrt((chi2 / n) / (min(k-1, r-1) or 1))

# Lift by bin
def binned_lift(df: pd.DataFrame, numeric_col: str, target_col: str, bins: int = 10):
    dd = df[[numeric_col, target_col]].dropna()
    if dd.empty: return pd.DataFrame()
    dd['bin'] = pd.qcut(dd[numeric_col], q=min(bins, dd[numeric_col].nunique()), duplicates='drop')
    grp = dd.groupby('bin')[target_col].agg(['mean','count'])
    grp = grp.rename(columns={'mean':'target_rate','count':'n'}).reset_index()
    baseline = dd[target_col].mean() if dd[target_col].nunique() > 1 else np.nan
    return grp, baseline

# Winsorization helper
def winsorize(s: pd.Series, p: float = 0.01):
    if not pd.api.types.is_numeric_dtype(s):
        return s
    lo, hi = s.quantile([p, 1-p])
    return s.clip(lo, hi)

# ----------------------------- Upload ----------------------------- #
st.sidebar.header("📥 Upload dataset")
uploaded = st.sidebar.file_uploader("Upload an Excel (.xlsx) or CSV file", type=["xlsx","xls","csv"]) 

if not uploaded:
    st.info("Upload a dataset to generate the dashboard automatically.")
    st.stop()

raw_bytes = uploaded.read()
df = load_data(raw_bytes, uploaded.name)
prof = profile_data(df.copy())

# ----------------------------- Global Controls (Sidebar) ----------------------------- #
st.sidebar.header("🎛️ Global Controls")
# Date filter (if any)
if prof['preferred_date']:
    date_col = prof['preferred_date']
    dmin = pd.to_datetime(df[date_col], errors='coerce').min()
    dmax = pd.to_datetime(df[date_col], errors='coerce').max()
    dr = st.sidebar.date_input(
        f"Date range • {date_col}",
        value=(dmin.date() if pd.notna(dmin) else None, dmax.date() if pd.notna(dmax) else None)
    )
else:
    date_col = None
    dr = None

# Pick top categorical filters (prioritize domain-relevant names if present)
priority_cats = [
    'cc_type','decision','maid_grouped_nationality','maid_education_label',
    'clientmts_dayoff_policy','clientmts_living_arrangement','clientmts_household_type',
    'maid_speaks_language','maid_nationality'
]
cat_filters = []
for c in priority_cats:
    if c in prof['categorical_cols']:
        cat_filters.append(c)
# Fill remaining slots up to 8
for c in prof['categorical_cols']:
    if c not in cat_filters and len(cat_filters) < 8:
        cat_filters.append(c)

selected_filters = {}
for c in cat_filters:
    opts = sorted([x for x in df[c].dropna().unique()])
    chosen = st.sidebar.multiselect(c, options=opts, default=[])
    if chosen:
        selected_filters[c] = set(chosen)

# Granularity (if datetime exists)
if prof['preferred_date']:
    gran = st.sidebar.radio("Granularity", options=["Day","Week","Month"], index=1, horizontal=True)
    freq_map = {"Day":"D","Week":"W","Month":"MS"}
else:
    gran, freq_map = None, {}

# Apply filters
fdf = df.copy()
if date_col and dr and isinstance(dr, tuple) and len(dr) == 2 and all(dr):
    start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
    fdf = fdf[(fdf[date_col] >= start) & (fdf[date_col] <= end + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1))]
for c, vals in selected_filters.items():
    fdf = fdf[fdf[c].isin(list(vals))]

st.caption(f"Filtered rows: {len(fdf):,} / {len(df):,}")

# ----------------------------- 2) KPI Strip ----------------------------- #
st.subheader("2) KPI Strip")

# Unique entity counts (if present)
col_names = list(df.columns)
entity_cols = [c for c in ['client_name','contract_id','maid_id'] if c in col_names]
cols = st.columns(max(3, len(entity_cols)+2))
# Total rows
cols[0].metric("Total rows", f"{len(fdf):,}")
# Unique entities
for i, c in enumerate(entity_cols, start=1):
    cols[i].metric(f"Unique {c}", f"{fdf[c].nunique():,}")

# Detect likely target columns
targets = [c for c in prof['likely_targets'] if c in fdf.columns and fdf[c].dropna().nunique() == 2]
# Prefer specific known targets if available
preferred_targets = [c for c in ['replaced_within_3_days','replaced_within_1week'] if c in targets]
if preferred_targets:
    targets = preferred_targets + [t for t in targets if t not in preferred_targets]

if targets:
    # Show up to two key target rates
    for j, t in enumerate(targets[:2], start=len(entity_cols)+1):
        rate = fdf[t].mean() if pd.api.types.is_numeric_dtype(fdf[t]) else fdf[t].map({True:1, False:0}).mean()
        cols[j % len(cols)].metric(f"Target rate • {t}", f"{rate*100:.1f}%", help="Mean of binary indicator")

# Business KPIs (if present)
num_kpis = [c for c in ['matching_score_percent','match_score','total_complaints','years_of_experience','duration_days'] if c in prof['numeric_cols']]
for k in num_kpis[:5]:
    val = fdf[k].mean()
    cols[(cols.index(k) if hasattr(cols, 'index') else 0) % len(cols)] if False else None
# Show as a neat row
if num_kpis:
    kcols = st.columns(len(num_kpis))
    for i,k in enumerate(num_kpis):
        kval = fdf[k].mean()
        kcols[i].metric(f"Avg {k}", f"{kval:.2f}")

# ----------------------------- 3) Overview ----------------------------- #
st.subheader("3) Overview")
# Datatype breakdown
dtype_groups = {
    'Numeric': len(prof['numeric_cols']),
    'Categorical': len(prof['categorical_cols']),
    'Datetime': len(prof['datetime_cols']),
    'Text': len(prof['text_cols'])
}
fig_dtype = px.treemap(
    names=list(dtype_groups.keys()),
    parents=["Fields"]*len(dtype_groups),
    values=list(dtype_groups.values()),
)
fig_dtype.update_layout(margin=dict(t=30,l=0,r=0,b=0))
st.plotly_chart(fig_dtype, use_container_width=True)

# Activity heatmap (weekday × hour) if a datetime exists
if prof['preferred_date'] and fdf[date_col].notna().any():
    dd = fdf[[date_col]].dropna().copy()
    dd['weekday'] = dd[date_col].dt.day_name()
    dd['hour'] = dd[date_col].dt.hour
    pivot = dd.pivot_table(index='weekday', columns='hour', values=date_col, aggfunc='count').fillna(0)
    # Order weekdays Mon→Sun
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pivot = pivot.reindex(order)
    fig_heat = px.imshow(pivot, aspect='auto', title=f"Activity heatmap • {date_col}")
    st.plotly_chart(fig_heat, use_container_width=True)

# Target by top segment
if targets:
    seg_candidates = [c for c in ['cc_type','decision','maid_grouped_nationality'] if c in prof['categorical_cols']]
    if not seg_candidates:
        seg_candidates = prof['categorical_cols'][:1]
    if seg_candidates:
        seg = st.selectbox("Segment for stacked target", options=seg_candidates, index=0)
        dd = fdf[[seg] + targets[:1]].dropna()
        if not dd.empty:
            rates = dd.groupby(seg)[targets[0]].mean().sort_values(ascending=False)
            base = dd[targets[0]].mean()
            fig_seg = px.bar(rates, title=f"{targets[0]} rate by {seg}")
            fig_seg.add_hline(y=base, line_dash='dash', annotation_text='Baseline', annotation_position='top left')
            st.plotly_chart(fig_seg, use_container_width=True)

# ----------------------------- 4) Distributions ----------------------------- #
st.subheader("4) Distributions")
# Numeric histogram/box
if prof['numeric_cols']:
    num = st.selectbox("Numeric column (hist/box)", options=prof['numeric_cols'], index=0)
    wins = st.checkbox("Winsorize 1% tails", value=False)
    s = fdf[num].dropna()
    if wins:
        s = winsorize(s)
    c1, c2 = st.columns(2)
    fig_hist = px.histogram(s, nbins=30, title=f"Histogram • {num}")
    c1.plotly_chart(fig_hist, use_container_width=True)
    fig_box = px.box(s, points='outliers', title=f"Box • {num}")
    c2.plotly_chart(fig_box, use_container_width=True)

# Categorical frequency (Top-N + Other)
if prof['categorical_cols']:
    cat = st.selectbox("Categorical column (Top-N)", options=prof['categorical_cols'], index=0)
    N = st.slider("Top-N", min_value=3, max_value=30, value=10)
    vc = top_categories(fdf[cat], n=N)
    fig_cat = px.bar(vc, title=f"Frequency • {cat}")
    st.plotly_chart(fig_cat, use_container_width=True)

# Donut share by chosen category
if prof['categorical_cols']:
    cat2 = st.selectbox("Donut by category", options=[c for c in prof['categorical_cols'] if c != cat], index=0)
    vc2 = fdf[cat2].astype('object').fillna('∅ Missing').value_counts()
    fig_donut = px.pie(vc2, hole=0.5, title=f"Share • {cat2}")
    st.plotly_chart(fig_donut, use_container_width=True)

# ----------------------------- 5) Relationships ----------------------------- #
st.subheader("5) Relationships")
# Correlation heatmap
num_for_corr = [c for c in prof['numeric_cols'] if fdf[c].nunique() > 1]
if len(num_for_corr) >= 2:
    corr = fdf[num_for_corr].corr()
    fig_corr = px.imshow(corr, text_auto=False, title="Correlation heatmap (numeric)")
    st.plotly_chart(fig_corr, use_container_width=True)

# Cramér's V (categorical)
cats_for_v = [c for c in prof['categorical_cols'] if fdf[c].nunique() > 1][:10]
if len(cats_for_v) >= 2:
    mat = pd.DataFrame(index=cats_for_v, columns=cats_for_v, dtype=float)
    for i,a in enumerate(cats_for_v):
        for j,b in enumerate(cats_for_v):
            if i <= j:
                v = cramers_v(fdf[a], fdf[b])
                mat.loc[a,b] = v
                mat.loc[b,a] = v
    fig_v = px.imshow(mat, zmin=0, zmax=1, title="Cramér’s V heatmap (categorical)")
    st.plotly_chart(fig_v, use_container_width=True)

# Scatter + trend
if len(num_for_corr) >= 2:
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X", options=num_for_corr, index=0)
    y = c2.selectbox("Y", options=[c for c in num_for_corr if c != x], index=0)
    color_by = c3.selectbox("Color by", options=(prof['categorical_cols'] + targets) if targets else prof['categorical_cols'], index=0 if prof['categorical_cols'] else None)
    fig_sc = px.scatter(fdf, x=x, y=y, color=color_by, opacity=0.7, trendline=None)
    st.plotly_chart(fig_sc, use_container_width=True)

# Lift chart: binned numeric vs target
if targets and prof['numeric_cols']:
    c1, c2 = st.columns(2)
    metric = c1.selectbox("Numeric metric", options=prof['numeric_cols'], index=prof['numeric_cols'].index('matching_score_percent') if 'matching_score_percent' in prof['numeric_cols'] else 0)
    tgt = c2.selectbox("Target", options=targets, index=0)
    grp, base = binned_lift(fdf, metric, tgt)
    if not grp.empty:
        fig_lift = px.bar(grp, x='bin', y='target_rate', hover_data=['n'], title=f"Lift • {metric} vs {tgt}")
        if not np.isnan(base):
            fig_lift.add_hline(y=base, line_dash='dash', annotation_text='Baseline')
        st.plotly_chart(fig_lift, use_container_width=True)

# ----------------------------- 6) Segments & Ranking ----------------------------- #
st.subheader("6) Segments & Ranking")
if targets and prof['categorical_cols']:
    seg = st.selectbox("Segment column", options=prof['categorical_cols'], index=prof['categorical_cols'].index('maid_education_label') if 'maid_education_label' in prof['categorical_cols'] else 0)
    tgt = st.selectbox("Target", options=targets, index=0)
    grp = fdf.groupby(seg)[tgt].agg(['mean','count']).rename(columns={'mean':'rate','count':'n'}).sort_values('rate', ascending=False)
    st.dataframe(grp, use_container_width=True)
    fig_rank = px.bar(grp.reset_index(), x=seg, y='rate', hover_data=['n'], title=f"Top segments by {tgt} rate")
    st.plotly_chart(fig_rank, use_container_width=True)

# Two-way crosstab heatmap
if targets and len(prof['categorical_cols']) >= 2:
    c1, c2 = st.columns(2)
    a = c1.selectbox("Cat A", options=prof['categorical_cols'], index=prof['categorical_cols'].index('cc_type') if 'cc_type' in prof['categorical_cols'] else 0)
    b = c2.selectbox("Cat B", options=[c for c in prof['categorical_cols'] if c != a], index=0)
    tgt = st.selectbox("Target", options=targets, index=0, key='tgt2')
    pivot = fdf.pivot_table(index=a, columns=b, values=tgt, aggfunc='mean')
    fig_ct = px.imshow(pivot, title=f"{tgt} rate • {a} × {b}")
    st.plotly_chart(fig_ct, use_container_width=True)

# Simple rule explorer (best single split on target)
if targets and prof['categorical_cols']:
    tgt = st.selectbox("Target for rule explorer", options=targets, index=0, key='tgt3')
    best_rule, best_lift = None, 0
    base = fdf[tgt].mean()
    for c in prof['categorical_cols']:
        vc = fdf[c].value_counts()
        for val, n in vc.items():
            if n < max(20, 0.02*len(fdf)):
                continue
            rate = fdf.loc[fdf[c]==val, tgt].mean()
            lift = rate - base
            if abs(lift) > abs(best_lift):
                best_lift = lift
                best_rule = (c, val, n, rate)
    if best_rule:
        c, val, n, rate = best_rule
        st.info(f"Best single split: {c} = {val} → rate={rate:.3f} (Δ vs base {best_lift:+.3f}) [n={n}]")

# ----------------------------- 7) Time & Cohorts ----------------------------- #
st.subheader("7) Time & Cohorts")
if prof['preferred_date']:
    dd = fdf.copy()
    freq = freq_map.get(gran, 'W')
    dd = dd.set_index(date_col).sort_index()
    # Build time series metrics as toggles
    metrics = {}
    metrics['rows'] = dd.resample(freq).size()
    if targets:
        tgt = targets[0]
        metrics[f'{tgt}_rate'] = dd[tgt].resample(freq).mean()
    for k in ['matching_score_percent','match_score','duration_days']:
        if k in dd.columns:
            metrics[f'avg_{k}'] = dd[k].resample(freq).mean()
    # Toggle which to show
    choices = st.multiselect("Time series metrics", options=list(metrics.keys()), default=list(metrics.keys())[:2])
    if choices:
        ts = pd.concat({k:v for k,v in metrics.items() if k in choices}, axis=1)
        fig_ts = px.line(ts, title=f"Time series • grouped by {gran}")
        st.plotly_chart(fig_ts, use_container_width=True)

    # Seasonality (weekday/hour)
    dd2 = fdf[[date_col]].dropna().copy()
    dd2['weekday'] = dd2[date_col].dt.day_name()
    dd2['hour'] = dd2[date_col].dt.hour
    wkp = dd2['weekday'].value_counts().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    fig_wd = px.bar(wkp, title="By weekday")
    st.plotly_chart(fig_wd, use_container_width=True)
    hp = dd2['hour'].value_counts().sort_index()
    fig_hr = px.bar(hp, title="By hour")
    st.plotly_chart(fig_hr, use_container_width=True)

    # Cohort by first-seen date (month)
    dd3 = fdf.copy()
    dd3['cohort_month'] = dd3[date_col].dt.to_period('M').astype(str)
    if 'duration_days' in dd3.columns:
        agg = dd3.groupby('cohort_month')['duration_days'].mean()
        fig_cohort = px.bar(agg, title="Average duration_days by cohort month")
        st.plotly_chart(fig_cohort, use_container_width=True)
else:
    st.caption("No datetime fields detected → Time & Cohorts: N/A")

# ----------------------------- 8) Funnel / Lifecycle ----------------------------- #
st.subheader("8) Funnel / Lifecycle")
stages = [c for c in ['replaced_within_3_days','replaced_within_1week','replaced_within_14_days','replaced_within_30_days','replaced_after_30_days'] if c in fdf.columns]
if stages:
    mode = st.radio("Window definition", options=["Mutually exclusive","Cumulative"], index=0, horizontal=True)
    counts = {}
    placed = len(fdf)
    if mode == "Cumulative":
        for s in stages:
            counts[s] = int(fdf[s].fillna(0).astype(int).sum())
    else:
        # Mutually exclusive windows based on first true stage
        # Build order using the stage list order
        remaining = fdf.copy()
        for s in stages:
            mask = remaining[s] == 1
            counts[s] = int(mask.sum())
            remaining = remaining.loc[~mask]
        counts['Active'] = len(remaining)
    # Plot funnel
    lab, val = list(counts.keys()), list(counts.values())
    fig_fun = go.Figure(go.Funnel(y=lab, x=val, textinfo="value+percent initial"))
    st.plotly_chart(fig_fun, use_container_width=True)

    # Time-to-event
    if 'duration_days' in fdf.columns:
        fig_dur = px.histogram(fdf['duration_days'].dropna(), nbins=30, title="Time-to-event • duration_days")
        st.plotly_chart(fig_dur, use_container_width=True)
else:
    st.caption("No stage flags detected → Funnel/Lifecycle: N/A")

# ----------------------------- 9) Geospatial ----------------------------- #
st.subheader("9) Geospatial")
if prof['geo']['lat'] and prof['geo']['lon']:
    fig_map = px.scatter_mapbox(
        fdf.dropna(subset=[prof['geo']['lat'], prof['geo']['lon']]),
        lat=prof['geo']['lat'], lon=prof['geo']['lon'],
        hover_data=[c for c in fdf.columns if c not in prof['text_cols']][:5],
        zoom=4, height=400
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.caption("No lat/lon fields → Geospatial: N/A")

# ----------------------------- 10) Text ----------------------------- #
st.subheader("10) Text")
if prof['text_cols']:
    tcol = st.selectbox("Text column", options=prof['text_cols'], index=prof['text_cols'].index('complaint_summary') if 'complaint_summary' in prof['text_cols'] else 0)
    s = fdf[tcol].dropna().astype(str)
    # Length distribution & missingness
    lengths = s.str.len()
    miss_pct = 100*(1 - len(s)/max(1,len(fdf)))
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(lengths, nbins=30, title=f"Length distribution • {tcol}"), use_container_width=True)
    c2.metric("% Missing", f"{miss_pct:.1f}%")

    # Top tokens / bigrams (simple tokenization)
    ngram = st.radio("N-gram", options=["unigram","bigram"], index=0, horizontal=True)
    def tokenize(text: str):
        tokens = re.findall(r"[A-Za-z]+", text.lower())
        return tokens
    tokens = []
    if ngram == 'unigram':
        for txt in s.head(5000):
            tokens.extend(tokenize(txt))
        counts = Counter(tokens).most_common(30)
        fig_tok = px.bar(pd.Series(dict(counts)), title=f"Top {ngram}s")
    else:
        for txt in s.head(5000):
            toks = tokenize(txt)
            tokens.extend([' '.join(toks[i:i+2]) for i in range(len(toks)-1)])
        counts = Counter(tokens).most_common(30)
        fig_tok = px.bar(pd.Series(dict(counts)), title=f"Top {ngram}s")
    st.plotly_chart(fig_tok, use_container_width=True)

    # Keyword impact on target
    if targets:
        keyword = st.text_input("Keyword to test impact", value="infant")
        if keyword:
            present = fdf[tcol].astype(str).str.contains(fr"\b{re.escape(keyword)}\b", case=False, na=False)
            base = fdf[targets[0]].mean()
            rate = fdf.loc[present, targets[0]].mean() if present.any() else np.nan
            n = int(present.sum())
            st.info(f"{keyword!r} present: {n} rows • {targets[0]} rate = {rate:.3f} (Δ vs base {rate-base:+.3f})")
else:
    st.caption("No long text fields → Text: N/A")

# ----------------------------- 11) Data Quality & Ops ----------------------------- #
st.subheader("11) Data Quality & Ops")
# Duplicate viewer
keys_default = [c for c in ['client_name','contract_id','cc_type','maid_id','tag_date','untag_date'] if c in fdf.columns]
key_cols = st.multiselect("Duplicate keys", options=list(fdf.columns), default=keys_default)
if key_cols:
    grp = fdf.groupby(key_cols).size().reset_index(name='n')
    dups = grp[grp['n']>1].sort_values('n', ascending=False)
    st.dataframe(dups.head(500), use_container_width=True)

# Rare-category detector
if prof['categorical_cols']:
    rc = st.selectbox("Rare-category scan on", options=prof['categorical_cols'], index=0)
    thr = st.slider("Rarity threshold (count <)", min_value=2, max_value=50, value=5)
    vc = fdf[rc].value_counts(dropna=False)
    rare = vc[vc < thr]
    st.dataframe(rare.to_frame(name='count'))

# Outlier table
if prof['numeric_cols']:
    numo = st.selectbox("Outlier scan on", options=prof['numeric_cols'], index=prof['numeric_cols'].index('duration_days') if 'duration_days' in prof['numeric_cols'] else 0)
    s = fdf[numo].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    out = fdf[(fdf[numo] < lo) | (fdf[numo] > hi)]
    st.dataframe(out.head(500), use_container_width=True)

# ----------------------------- 12) Tables & Export ----------------------------- #
st.subheader("12) Tables & Export")
choose_cols = st.multiselect("Columns to show", options=list(fdf.columns), default=list(fdf.columns)[:15])
st.dataframe(fdf[choose_cols].head(3000), use_container_width=True)

# Download filtered view
csv = fdf.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered CSV", data=csv, file_name="filtered_view.csv", mime="text/csv")

st.caption("Built from your dataset profile: columns, dtypes, cardinality, missingness → blocks rendered only when applicable.")
