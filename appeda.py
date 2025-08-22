import io
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Data Explorer Dashboard", layout="wide", page_icon="📊")

# ---------- Helpers
@st.cache_data(show_spinner=False)
def read_any(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = file.name.lower()
    content = file.read()
    bio = io.BytesIO(content)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(bio)
    # CSV (robust)
    try:
        bio.seek(0)
        return pd.read_csv(bio, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        bio.seek(0)
        return pd.read_csv(bio, sep=None, engine="python", encoding_errors="ignore", on_bad_lines="skip")

def detect_columns(df: pd.DataFrame):
    # attempt to parse plausible date cols
    date_like = []
    for c in df.columns:
        if df[c].dtype.kind in "Mm":
            date_like.append(c)
        elif any(k in c.lower() for k in ["date", "time", "dt"]):
            try:
                pd.to_datetime(df[c], errors="raise")
                date_like.append(c)
            except Exception:
                pass
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Treat booleans separately (often encode flags)
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # add booleans to categorical for faceting
    cat_cols = [c for c in cat_cols if c not in date_like]
    return sorted(list(dict.fromkeys(date_like))), sorted(num_cols), sorted(bool_cols), sorted(cat_cols)

def apply_filters(df: pd.DataFrame, filter_cfg: dict):
    out = df.copy()
    for c, spec in (filter_cfg or {}).items():
        if c not in out.columns: 
            continue
        kind = spec["kind"]
        if kind == "cat":
            vals = spec.get("values", [])
            if vals:
                out = out[out[c].astype(str).isin(vals)]
        elif kind == "num":
            lo, hi = spec.get("range", (None, None))
            if lo is not None: out = out[out[c] >= lo]
            if hi is not None: out = out[out[c] <= hi]
        elif kind == "date":
            lo, hi = spec.get("range", (None, None))
            col = pd.to_datetime(out[c], errors="coerce")
            if lo is not None: out = out[col >= lo]
            if hi is not None: out = out[col <= hi]
    return out

def agg_series(df, metric, agg):
    if metric == "__rows__":
        return df.assign(__rows__=1)["__rows__"].agg("sum")
    func = {"Sum":"sum","Mean":"mean","Median":"median","Min":"min","Max":"max","Std":"std"}[agg]
    return df[metric].agg(func)

PLOTLY_TEMPLATES = {"Light": "plotly_white", "Dark": "plotly_dark"}

# ---------- Sidebar: data + theme
with st.sidebar:
    st.header("Upload data")
    up = st.file_uploader("CSV or Excel", type=["csv","xlsx","xls"])
    theme = st.radio("Plot theme", ["Light","Dark"], index=0, horizontal=True)
    st.caption("All charts update with the filters you set below.")

df = read_any(up)
if df.empty:
    st.info("Upload a CSV/XLSX to explore.")
    st.stop()

date_cols, num_cols, bool_cols, cat_cols = detect_columns(df)
# convenience: show special fields if present
special_cols = [c for c in ["client_name","maid_id","match_score","decision","theme_penalty","tag_date"] if c in df.columns]

st.title("📊 Data Explorer")
st.caption(f"Rows: **{len(df):,}** • Columns: **{len(df.columns)}**")
if special_cols:
    st.caption("Detected key fields: " + ", ".join(special_cols))

# ---------- Global filters
with st.expander("Filters", expanded=True):
    cols1, cols2, cols3 = st.columns(3)
    filter_cfg = {}

    with cols1:
        cat_pick = st.multiselect("Categorical filters", cat_cols, default=[])
        for c in cat_pick:
            vals = st.multiselect(f"→ {c}", sorted(df[c].astype(str).dropna().unique().tolist()))
            filter_cfg[c] = {"kind":"cat","values":vals}

    with cols2:
        num_pick = st.multiselect("Numeric filters", num_cols, default=[])
        for c in num_pick:
            s = df[c].dropna()
            lo, hi = float(s.min()), float(s.max())
            r = st.slider(f"→ {c}", lo, hi, value=(lo, hi))
            filter_cfg[c] = {"kind":"num","range":r}

    with cols3:
        dt_pick = st.multiselect("Date filters", date_cols, default=[])
        for c in dt_pick:
            s = pd.to_datetime(df[c], errors="coerce").dropna()
            if s.empty: continue
            lo, hi = s.min().date(), s.max().date()
            r = st.date_input(f"→ {c}", (lo, hi))
            if isinstance(r, tuple) and len(r)==2:
                filter_cfg[c] = {"kind":"date","range":r}

fdf = apply_filters(df, filter_cfg)

# ---------- KPI Tiles
st.subheader("Key metrics")
k1,k2,k3,k4 = st.columns(4)
with k1:
    st.metric("Rows (filtered)", f"{len(fdf):,}")
with k2:
    if "client_name" in fdf.columns:
        st.metric("Unique clients", f"{fdf['client_name'].nunique():,}")
    else:
        pick = st.selectbox("Unique of…", options=["—"]+cat_cols, index=0)
        if pick != "—":
            st.metric(f"Unique {pick}", f"{fdf[pick].nunique():,}")
with k3:
    # mean of a numeric (default: match_score if present)
    defaults = ["match_score","theme_penalty"]
    default_metric = next((m for m in defaults if m in num_cols), (num_cols[0] if num_cols else None))
    metric_for_mean = st.selectbox("Mean of", options=["—"]+num_cols, index=(num_cols.index(default_metric)+1 if default_metric else 0))
    if metric_for_mean != "—":
        st.metric(f"Mean {metric_for_mean}", f"{fdf[metric_for_mean].mean():.2f}")
with k4:
    if "decision" in fdf.columns:
        ok = (fdf["decision"].astype(str).str.upper()=="OK").mean()*100
        st.metric("OK rate", f"{ok:.1f}%")

st.divider()

# ---------- Charts
tpl = PLOTLY_TEMPLATES[theme]

# Time series
st.subheader("Time series")
tcol1, tcol2, tcol3, tcol4 = st.columns([1.2,1,1,1])
with tcol1:
    dt_col = st.selectbox("Date column", options=(date_cols or ["(no date cols)"]))
with tcol2:
    period = st.selectbox("Resample", options=["D","W","M","Q"], index=1, help="Day/Week/Month/Quarter")
with tcol3:
    ts_metric = st.selectbox("Metric", options=["__rows__"]+num_cols, index=0)
with tcol4:
    color_by = st.selectbox("Color by (optional)", options=["(none)"]+cat_cols, index=0)

ts_fig = go.Figure()
if date_cols and dt_col in fdf.columns:
    work = fdf.copy()
    work[dt_col] = pd.to_datetime(work[dt_col], errors="coerce")
    work = work.dropna(subset=[dt_col]).sort_values(dt_col)
    by = [color_by] if color_by in work.columns else []
    if ts_metric == "__rows__":
        g = work.groupby([pd.Grouper(key=dt_col, freq=period)] + by).size().reset_index(name="value")
    else:
        g = work.groupby([pd.Grouper(key=dt_col, freq=period)] + by)[ts_metric].agg("mean").reset_index(name="value")
    ts_fig = px.line(g, x=dt_col, y="value", color=(color_by if color_by in work.columns else None), template=tpl)
    ts_fig.update_traces(mode="lines+markers", marker_size=4)
    ts_fig.update_layout(yaxis_title=("Count" if ts_metric=="__rows__" else ts_metric))
st.plotly_chart(ts_fig, use_container_width=True, config={"displaylogo": False})

# Categorical explorer
st.subheader("Categorical explorer")
cc1, cc2, cc3, cc4 = st.columns([1,1,1,1])
with cc1:
    cat_dim = st.selectbox("Dimension", options=(cat_cols or ["(no categorical)"]))
with cc2:
    cat_metric = st.selectbox("Metric", options=["__rows__"]+num_cols, index=0)
with cc3:
    topn = st.number_input("Top N", min_value=3, max_value=50, value=12, step=1)
with cc4:
    percent_stack = st.checkbox("Normalize to %", value=False)

cat_fig = go.Figure()
if cat_cols and cat_dim in fdf.columns:
    if cat_metric == "__rows__":
        g = fdf.groupby(cat_dim).size().reset_index(name="value")
    else:
        g = fdf.groupby(cat_dim)[cat_metric].mean().reset_index(name="value")
    g = g.sort_values("value", ascending=False).head(int(topn))
    if percent_stack and cat_metric=="__rows__":
        total = g["value"].sum()
        g["value"] = g["value"]/total*100
    cat_fig = px.bar(g, x="value", y=cat_dim, orientation="h", template=tpl,
                     text=([f"{v:.1f}%" for v in g["value"]] if percent_stack and cat_metric=="__rows__" else None))
    cat_fig.update_layout(xaxis_title=("%" if percent_stack and cat_metric=="__rows__" else cat_metric),
                          yaxis_title=cat_dim)
st.plotly_chart(cat_fig, use_container_width=True, config={"displaylogo": False})

# Numeric distributions
st.subheader("Numeric distributions")
nc1, nc2, nc3 = st.columns([1,1,1])
with nc1:
    num_col = st.selectbox("Numeric", options=(num_cols or ["(no numeric)"]))
with nc2:
    by_cat = st.selectbox("By category (optional)", options=["(none)"]+cat_cols)
with nc3:
    bins = st.slider("Bins", 5, 80, 30)

num_fig = go.Figure()
if num_cols and num_col in fdf.columns:
    num_fig = px.histogram(fdf, x=num_col, color=(by_cat if by_cat in fdf.columns else None),
                           nbins=bins, marginal="box", template=tpl, opacity=0.85)
st.plotly_chart(num_fig, use_container_width=True, config={"displaylogo": False})

# Correlation heatmap
if len(num_cols) >= 2:
    st.subheader("Correlation (numeric)")
    corr = fdf[num_cols].corr(numeric_only=True)
    heat = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmin=-1, zmax=1
    ))
    heat.update_layout(template=tpl, height=420)
    st.plotly_chart(heat, use_container_width=True, config={"displaylogo": False})

# Scatter
st.subheader("Scatter")
sc1, sc2, sc3, sc4 = st.columns([1,1,1,1])
with sc1:
    xcol = st.selectbox("X", options=(num_cols or ["(no numeric)"]), index=(num_cols.index("match_score") if "match_score" in num_cols else 0 if num_cols else 0))
with sc2:
    ycol = st.selectbox("Y", options=(num_cols or ["(no numeric)"]), index=(num_cols.index("theme_penalty") if "theme_penalty" in num_cols else 0 if num_cols else 0))
with sc3:
    color_sc = st.selectbox("Color", options=["(none)"]+cat_cols)
with sc4:
    size_sc = st.selectbox("Size", options=["(none)"]+num_cols)

sc_fig = go.Figure()
if num_cols and xcol in fdf.columns and ycol in fdf.columns:
    sc_fig = px.scatter(
        fdf, x=xcol, y=ycol,
        color=(color_sc if color_sc in fdf.columns else None),
        size=(size_sc if size_sc in fdf.columns else None),
        hover_data=[c for c in ["client_name","maid_id","decision"] if c in fdf.columns],
        template=tpl, opacity=0.85
    )
st.plotly_chart(sc_fig, use_container_width=True, config={"displaylogo": False})

# Pivot builder
st.subheader("Pivot builder")
pv1, pv2, pv3, pv4 = st.columns([1,1,1,1])
with pv1:
    rows_dim = st.selectbox("Rows", options=["(none)"]+cat_cols)
with pv2:
    cols_dim = st.selectbox("Cols", options=["(none)"]+cat_cols, index=(1 if len(cat_cols)>1 else 0))
with pv3:
    pv_metric = st.selectbox("Metric", options=["__rows__"]+num_cols)
with pv4:
    agg_fn = st.selectbox("Agg", options=["Count","Sum","Mean","Median"], index=0 if pv_metric=="__rows__" else 2)

pvt = pd.DataFrame()
if rows_dim!="(none)":
    data = fdf.copy()
    if pv_metric == "__rows__":
        data["__rows__"] = 1
        val = "__rows__"
    else:
        val = pv_metric
    pvt = pd.pivot_table(
        data,
        index=rows_dim,
        columns=(None if cols_dim=="(none)" else cols_dim),
        values=val,
        aggfunc=("count" if pv_metric=="__rows__" and agg_fn=="Count" else agg_fn.lower()),
        fill_value=0
    )
    st.dataframe(pvt, use_container_width=True)

# Segment compare (A vs B)
st.subheader("Segment compare — A vs B")
sgA, sgB = st.columns(2)
def pick_segment(label):
    out = {}
    st.markdown(f"**{label}**")
    choose = st.multiselect(f"Filters for {label}", cat_cols, key=f"{label}_cats")
    for c in choose:
        vals = st.multiselect(f"{label} → {c}", sorted(fdf[c].astype(str).dropna().unique().tolist()), key=f"{label}_{c}")
        out[c] = {"kind":"cat","values":vals}
    return out

with sgA:
    segA = pick_segment("A")
with sgB:
    segB = pick_segment("B")

cA = apply_filters(fdf, segA)
cB = apply_filters(fdf, segB)
cm1, cm2, cm3 = st.columns(3)
default_metric = "match_score" if "match_score" in num_cols else (num_cols[0] if num_cols else None)
with cm1:
    st.metric("Rows A", f"{len(cA):,}")
    if default_metric: st.metric(f"Mean {default_metric} (A)", f"{cA[default_metric].mean():.2f}")
with cm2:
    st.metric("Rows B", f"{len(cB):,}")
    if default_metric: st.metric(f"Mean {default_metric} (B)", f"{cB[default_metric].mean():.2f}")
with cm3:
    if "decision" in fdf.columns:
        okA = (cA["decision"].astype(str).str.upper()=="OK").mean()*100 if len(cA) else 0
        okB = (cB["decision"].astype(str).str.upper()=="OK").mean()*100 if len(cB) else 0
        st.metric("OK% A vs B", f"{okA:.1f}%  vs  {okB:.1f}%")

# Table + export
st.subheader("Rows (filtered)")
st.dataframe(fdf.head(5000), use_container_width=True)
st.download_button(
    "⬇️ Download filtered CSV",
    data=fdf.to_csv(index=False).encode("utf-8"),
    file_name="filtered_export.csv",
    mime="text/csv",
)
