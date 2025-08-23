# streamlit_app.py
# One-Page EDA Dashboard — auto-tailored to your uploaded dataset
# - Upload Excel/CSV
# - Profile dtypes, dates, targets
# - Render a single-page grid of KPI cards + charts (like the example image)
# - Uses actual column names; shows blocks only if fields exist

import io
import re
import math
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="One-Page EDA Dashboard", layout="wide")
st.markdown("""
<style>
/* compact KPI cards */
div[data-testid="stMetric"] {background: #f7f9fc; border:1px solid #e7edf3; padding: 8px 12px; border-radius: 12px; box-shadow: 0 1px 2px rgba(0,0,0,.04);} 
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("📊 One‑Page Auto EDA")

# ----------------------------- Helpers ----------------------------- #
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(file_bytes))
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(file_bytes), encoding='cp1252')

@st.cache_data(show_spinner=False)
def profile_data(df: pd.DataFrame):
    # Parse dates by dtype or name hints
    datetime_cols, parsed_cols = [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(c); continue
        if any(t in c.lower() for t in ['date','time','timestamp']):
            p = pd.to_datetime(s, errors='coerce')
            if p.notna().sum() >= max(10, 0.2*len(df)):
                df[c] = p; parsed_cols.append(c); datetime_cols.append(c)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    object_cols  = [c for c in df.columns if df[c].dtype == 'object']
    # long text heuristic
    def is_text(s: pd.Series):
        if s.dtype != 'object': return False
        sample = s.dropna().astype(str).head(500)
        return False if sample.empty else sample.str.len().mean() >= 30
    text_cols = [c for c in object_cols if is_text(df[c])]
    categorical_cols = [c for c in object_cols if c not in text_cols and 2 <= df[c].nunique(dropna=True) <= 50]
    # likely binary/target
    boollike = []
    for c in df.columns:
        vals = set(str(x).strip().lower() for x in df[c].dropna().unique())
        if vals and vals.issubset({'0','1','true','false','yes','no'}) or (pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) == 2):
            boollike.append(c)
    pat = re.compile(r"^(target|label|churn|outcome|replaced_.*|is_.*|will_.*)$", re.I)
    likely_targets = []
    for c in df.columns:
        if pat.match(c): likely_targets.append(c)
    for c in boollike:
        if c not in likely_targets: likely_targets.append(c)
    # preferred date
    preferred_date = None
    for p in ['tag_date','untag_date']:
        if p in df.columns and pd.api.types.is_datetime64_any_dtype(df[p]):
            preferred_date = p; break
    if preferred_date is None and datetime_cols:
        preferred_date = datetime_cols[0]
    # geo hints
    geo = {
        'lat': next((c for c in df.columns if c.lower() in {'lat','latitude'}), None),
        'lon': next((c for c in df.columns if c.lower() in {'lon','lng','longitude'}), None)
    }
    return dict(df=df, numeric=numeric_cols, cats=categorical_cols, text=text_cols, dates=datetime_cols,
                targets=[c for c in likely_targets if df[c].dropna().nunique()==2], date_col=preferred_date, geo=geo)

@st.cache_data(show_spinner=False)
def top_categories(series: pd.Series, n:int=10):
    vc = series.astype('object').fillna('∅ Missing').value_counts(dropna=False)
    top = vc.head(n)
    other = vc.iloc[n:].sum()
    if other>0: top['Other'] = other
    return top

# ----------------------------- Upload ----------------------------- #
up = st.file_uploader("Upload Excel/CSV", type=["xlsx","xls","csv"], label_visibility="collapsed")
if not up:
    st.info("Upload a dataset to generate the dashboard.")
    st.stop()

raw = up.read()
df = load_data(raw, up.name)
prof = profile_data(df.copy())

# ----------------------------- Top Filter Bar ----------------------------- #
bar = st.container()
with bar:
    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
    # Date range
    date_col = prof['date_col']
    dr = None
    if date_col:
        dmin = pd.to_datetime(df[date_col], errors='coerce').min()
        dmax = pd.to_datetime(df[date_col], errors='coerce').max()
        dr = c1.date_input(f"Date range • {date_col}", value=(dmin.date() if pd.notna(dmin) else None, dmax.date() if pd.notna(dmax) else None))
    else:
        c1.markdown("**Date range:** N/A")
    # Top categorical filters (up to 3)
    priority = ['cc_type','decision','maid_grouped_nationality','maid_education_label','maid_speaks_language','maid_nationality','clientmts_dayoff_policy','clientmts_living_arrangement']
    cats = [c for c in priority if c in prof['cats']][:3]
    # backfill if <3
    for c in prof['cats']:
        if len(cats) >= 3: break
        if c not in cats: cats.append(c)
    selections = {}
    if cats:
        opts = df[cats[0]].dropna().unique().tolist()
        sel0 = c2.multiselect(cats[0], opts)
        if sel0: selections[cats[0]] = set(sel0)
    if len(cats) > 1:
        opts = df[cats[1]].dropna().unique().tolist()
        sel1 = c3.multiselect(cats[1], opts)
        if sel1: selections[cats[1]] = set(sel1)
    if len(cats) > 2:
        opts = df[cats[2]].dropna().unique().tolist()
        sel2 = c4.multiselect(cats[2], opts)
        if sel2: selections[cats[2]] = set(sel2)

# Granularity
gran = st.radio("Granularity", options=["Week","Month","Day"], index=0, horizontal=True) if prof['date_col'] else None
freq = {"Day":"D","Week":"W","Month":"MS"}.get(gran, None)

# Apply filters
fdf = df.copy()
if prof['date_col'] and isinstance(dr, tuple) and len(dr)==2 and all(dr):
    start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
    fdf = fdf[(fdf[date_col] >= start) & (fdf[date_col] <= end + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1))]
for c, vals in selections.items():
    fdf = fdf[fdf[c].isin(list(vals))]

st.caption(f"Filtered rows: {len(fdf):,} / {len(df):,}")

# Determine targets
targets = [t for t in prof['targets'] if t in fdf.columns]
preferred = [t for t in ['replaced_within_3_days','replaced_within_1week'] if t in targets]
if preferred: targets = preferred + [t for t in targets if t not in preferred]

# ----------------------------- KPI Strip (cards) ----------------------------- #
row1 = st.columns(8)
row1[0].metric("Total rows", f"{len(fdf):,}")
if 'client_name' in fdf.columns:
    row1[1].metric("Unique client_name", f"{fdf['client_name'].nunique():,}")
if 'contract_id' in fdf.columns:
    row1[2].metric("Unique contract_id", f"{fdf['contract_id'].nunique():,}")
if 'maid_id' in fdf.columns:
    row1[3].metric("Unique maid_id", f"{fdf['maid_id'].nunique():,}")
# targets
if targets:
    if len(targets) >= 1:
        rate = fdf[targets[0]].mean()
        row1[4].metric(f"Rate • {targets[0]}", f"{rate*100:.1f}%")
    if len(targets) >= 2:
        rate2 = fdf[targets[1]].mean()
        row1[5].metric(f"Rate • {targets[1]}", f"{rate2*100:.1f}%")
# business numerics
bnums = [c for c in ['matching_score_percent','match_score','total_complaints','duration_days'] if c in prof['numeric']]
if bnums:
    if len(bnums) > 0:
        row1[6].metric(f"Avg {bnums[0]}", f"{fdf[bnums[0]].mean():.2f}")
    if len(bnums) > 1:
        row1[7].metric(f"Avg {bnums[1]}", f"{fdf[bnums[1]].mean():.2f}")

# ----------------------------- Charts Grid ----------------------------- #
# Row A: Top category, Usage over time, Composition donut
A1, A2, A3 = st.columns(3)
# A1 Top 10 of chosen category
if prof['cats']:
    catA = A1.selectbox("Top‑N by category", options=prof['cats'], index=prof['cats'].index('maid_grouped_nationality') if 'maid_grouped_nationality' in prof['cats'] else 0, key='catA')
    vc = fdf[catA].astype('object').fillna('∅ Missing').value_counts().head(10)
    A1.plotly_chart(px.bar(vc, title=f"Top 10 • {catA}"), use_container_width=True, height=320)
else:
    A1.info("No categorical fields")
# A2 Usage over time (counts)
if prof['date_col']:
    dd = fdf.set_index(prof['date_col']).sort_index()
    ts = dd.resample(freq or 'W').size()
    A2.plotly_chart(px.line(ts, title="Usage Over Time"), use_container_width=True, height=320)
else:
    A2.info("No date field")
# A3 Donut by category
if prof['cats']:
    catB = A3.selectbox("Donut • category", options=[c for c in prof['cats'] if c != (locals().get('catA'))] or prof['cats'], index=0, key='catB')
    vc2 = fdf[catB].astype('object').fillna('∅ Missing').value_counts()
    A3.plotly_chart(px.pie(vc2, hole=0.55, title=f"Composition • {catB}"), use_container_width=True, height=320)
else:
    A3.info("No categorical fields")

# Row B: Target by segment, Weekday×Hour heatmap, Scatter
B1, B2, B3 = st.columns(3)
# B1 Target by segment
if targets and prof['cats']:
    seg = B1.selectbox("Target by segment", options=prof['cats'], index=prof['cats'].index('decision') if 'decision' in prof['cats'] else 0, key='seg')
    dd = fdf[[seg, targets[0]]].dropna()
    if not dd.empty:
        rates = dd.groupby(seg)[targets[0]].mean().sort_values(ascending=False)
        base = dd[targets[0]].mean()
        fig = px.bar(rates, title=f"{targets[0]} by {seg}")
        fig.add_hline(y=base, line_dash='dash', annotation_text='Baseline', annotation_position='top left')
        B1.plotly_chart(fig, use_container_width=True, height=320)
else:
    B1.info("No binary target or category")
# B2 Weekday×Hour
if prof['date_col'] and fdf[prof['date_col']].notna().any():
    dd = fdf[[prof['date_col']]].dropna().copy()
    dd['weekday'] = dd[prof['date_col']].dt.day_name()
    dd['hour'] = dd[prof['date_col']].dt.hour
    pivot = dd.pivot_table(index='weekday', columns='hour', values=prof['date_col'], aggfunc='count').fillna(0)
    pivot = pivot.reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    B2.plotly_chart(px.imshow(pivot, aspect='auto', title="Activity Heatmap (weekday×hour)"), use_container_width=True, height=320)
else:
    B2.info("No date field")
# B3 Scatter
nums = [c for c in prof['numeric'] if fdf[c].nunique()>1]
if len(nums) >= 2:
    x = B3.selectbox("X", options=nums, index=0, key='sx')
    y = B3.selectbox("Y", options=[c for c in nums if c != x], index=0, key='sy')
    color_by = targets[0] if targets else (prof['cats'][0] if prof['cats'] else None)
    fig_sc = px.scatter(fdf, x=x, y=y, color=color_by, opacity=0.75, title="Scatter")
    B3.plotly_chart(fig_sc, use_container_width=True, height=320)
else:
    B3.info("Not enough numeric fields")

# Row C (optional): Funnel & Text quick‑look
C1, C2, C3 = st.columns(3)
# Funnel if stage flags
stages = [c for c in ['replaced_within_3_days','replaced_within_1week','replaced_within_14_days','replaced_within_30_days','replaced_after_30_days'] if c in fdf.columns]
if stages:
    counts = {}
    rem = fdf.copy()
    for s in stages:
        m = rem[s] == 1
        counts[s] = int(m.sum())
        rem = rem.loc[~m]
    counts['Active'] = len(rem)
    lab, val = list(counts.keys()), list(counts.values())
    C1.plotly_chart(go.Figure(go.Funnel(y=lab, x=val, textinfo="value+percent initial")), use_container_width=True, height=320)
else:
    C1.info("No stage flags")
# Text quick‑look
if prof['text']:
    tcol = prof['text'][0]
    lengths = fdf[tcol].dropna().astype(str).str.len()
    C2.plotly_chart(px.histogram(lengths, nbins=30, title=f"Text length • {tcol}"), use_container_width=True, height=320)
else:
    C2.info("No long text fields")
# Geomap
if prof['geo']['lat'] and prof['geo']['lon']:
    fig_map = px.scatter_mapbox(
        fdf.dropna(subset=[prof['geo']['lat'], prof['geo']['lon']]),
        lat=prof['geo']['lat'], lon=prof['geo']['lon'], zoom=2, height=320,
        hover_data=[c for c in fdf.columns if c not in prof['text']][:4]
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    C3.plotly_chart(fig_map, use_container_width=True)
else:
    C3.info("No lat/lon fields")

# Data grid at bottom
st.markdown("---")
st.dataframe(fdf.head(1500), use_container_width=True)
