"""
智慧旅宿「滯銷風險」預警平台
Inside Airbnb Taipei · Japanese Minimalist · Light Theme
─────────────────────────────────────────────────────────
每張圖表均標示使用的資料分析方法
Data: listings.csv / monthly_calendar.csv / monthly_reviews.csv
Deploy: Streamlit Community Cloud (share.streamlit.io)
"""

# ── Imports ──────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="旅宿滯銷風險預警 · Taipei",
    page_icon="🏯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  DESIGN TOKENS · 日系簡約淡色系
# ══════════════════════════════════════════════════════════════════
P = dict(
    bg       = "#F8F7F5",   # 和紙白
    surface  = "#FFFFFF",
    card     = "#FDFCFA",
    border   = "#E8E4DE",
    border2  = "#D4CFC8",
    ink      = "#2A2A2A",
    ink2     = "#505050",
    muted    = "#9A9490",
    primary  = "#4E7FB0",   # 藍墨
    accent   = "#8B7BA8",   # 紫苑
    high     = "#C4645A",   # 朱
    medium   = "#C49A4A",   # 金
    low      = "#5B9E73",   # 若草
    tag_bg   = "#F2F0EC",
    method   = "#EEF4FB",   # method badge bg
    method_t = "#3D6B96",   # method badge text
)
RC = {"高風險": P["high"], "中風險": P["medium"], "低風險": P["low"]}
RT_C = {"整棟出租": P["primary"], "私人套房": P["accent"],
        "共用套房": P["medium"],   "飯店客房": P["low"]}
ROOM_JP = {"Entire home/apt": "整棟出租", "Private room": "私人套房",
           "Shared room": "共用套房",    "Hotel room": "飯店客房"}

# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');
html,body,.stApp{{background:{P['bg']};color:{P['ink']};
  font-family:'Noto Sans TC',sans-serif;}}
section[data-testid="stSidebar"]{{background:{P['surface']};
  border-right:1px solid {P['border']};}}

/* Metrics */
[data-testid="stMetric"]{{background:{P['surface']};
  border:1px solid {P['border']};border-radius:10px;padding:16px 20px;
  box-shadow:0 1px 3px rgba(0,0,0,.04);}}
[data-testid="stMetricLabel"]{{color:{P['muted']} !important;
  font-size:.73rem !important;letter-spacing:.08em;text-transform:uppercase;}}
[data-testid="stMetricValue"]{{color:{P['ink']} !important;
  font-size:1.55rem !important;font-weight:700;}}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{{background:transparent;
  border-bottom:2px solid {P['border']};gap:0;padding:0;}}
.stTabs [data-baseweb="tab"]{{color:{P['muted']};border-radius:0;
  padding:10px 24px;border-bottom:2px solid transparent;margin-bottom:-2px;
  font-size:.87rem;font-weight:500;letter-spacing:.02em;}}
.stTabs [aria-selected="true"]{{color:{P['primary']} !important;
  border-bottom:2px solid {P['primary']} !important;background:transparent !important;}}

/* Sidebar */
section[data-testid="stSidebar"] label{{color:{P['ink2']} !important;font-size:.81rem;}}

/* Section header */
.sec{{font-size:.72rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:{P['muted']};margin:20px 0 4px;padding-bottom:7px;
  border-bottom:1px solid {P['border']};}}

/* Method badge */
.mbadge{{display:inline-flex;align-items:center;gap:5px;
  background:{P['method']};border:1px solid #C8DCF0;
  border-radius:5px;padding:3px 10px;font-size:.71rem;font-weight:600;
  color:{P['method_t']};letter-spacing:.04em;margin-bottom:8px;}}

/* Risk chips */
.chip-h{{background:#FDECEA;color:{P['high']};padding:2px 10px;
  border-radius:12px;font-size:.74rem;font-weight:700;}}
.chip-m{{background:#FDF5E4;color:#A07A20;padding:2px 10px;
  border-radius:12px;font-size:.74rem;font-weight:700;}}
.chip-l{{background:#EAF5EE;color:#3D7A55;padding:2px 10px;
  border-radius:12px;font-size:.74rem;font-weight:700;}}

/* Note panel */
.note{{background:{P['tag_bg']};border-left:3px solid {P['primary']};
  padding:9px 14px;border-radius:0 6px 6px 0;
  font-size:.8rem;color:{P['ink2']};margin:8px 0;}}

/* Divider */
hr{{border:none;border-top:1px solid {P['border']} !important;margin:16px 0;}}

/* Scrollbar */
::-webkit-scrollbar{{width:4px;}}
::-webkit-scrollbar-track{{background:transparent;}}
::-webkit-scrollbar-thumb{{background:{P['border2']};border-radius:2px;}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  HELPERS · UI COMPONENTS
# ══════════════════════════════════════════════════════════════════
def sec(label: str):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)

def method_badge(zh: str, en: str):
    """在圖表上方顯示分析方法標籤"""
    st.markdown(
        f'<span class="mbadge">📐 {zh} &ensp;·&ensp; {en}</span>',
        unsafe_allow_html=True,
    )

def note(msg: str):
    st.markdown(f'<div class="note">{msg}</div>', unsafe_allow_html=True)

def html_table(df_in: pd.DataFrame,
               fmt: dict = None,
               cell_fn: dict = None,
               height: int = 380) -> None:
    """
    PyArrow-free HTML table renderer.
    fmt     : {col: format_string}
    cell_fn : {col: callable(val) -> css-str}
    """
    fmt     = fmt or {}
    cell_fn = cell_fn or {}

    th = (f"background:{P['tag_bg']};color:{P['muted']};font-size:.72rem;"
          f"letter-spacing:.07em;text-transform:uppercase;padding:9px 14px;"
          f"border-bottom:2px solid {P['border2']};white-space:nowrap;"
          f"position:sticky;top:0;z-index:1;")
    td_base = (f"padding:8px 14px;font-size:.82rem;color:{P['ink']};"
               f"border-bottom:1px solid {P['border']};white-space:nowrap;")

    header = "".join(f'<th style="{th}">{c}</th>' for c in df_in.columns)
    rows   = []
    for i, (_, row) in enumerate(df_in.iterrows()):
        bg = P["surface"] if i % 2 == 0 else P["tag_bg"]
        cells = []
        for col in df_in.columns:
            val  = row[col]
            disp = ("–" if pd.isna(val) else
                    (fmt[col].format(val) if col in fmt and pd.notna(val) else str(val)))
            css  = f"background:{bg};"
            if col in cell_fn:
                try: css += cell_fn[col](val)
                except: pass
            cells.append(f'<td style="{td_base}{css}">{disp}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")

    st.markdown(
        f'<div style="overflow:auto;max-height:{height}px;border:1px solid {P["border"]};'
        f'border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.05);">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
#  CHART THEME
# ══════════════════════════════════════════════════════════════════
_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font = dict(color=P["ink2"], family="Noto Sans TC,sans-serif", size=11),
    margin = dict(l=46, r=16, t=30, b=34),
    legend = dict(bgcolor="rgba(255,255,255,.7)", bordercolor=P["border"],
                  borderwidth=1, font=dict(color=P["ink2"])),
    xaxis = dict(gridcolor=P["border"], linecolor=P["border"], zeroline=False,
                 tickfont=dict(color=P["muted"])),
    yaxis = dict(gridcolor=P["border"], linecolor=P["border"], zeroline=False,
                 tickfont=dict(color=P["muted"])),
)
def T(fig, h=None, legend=True):
    kw = dict(**_LAYOUT)
    if h:          kw["height"]     = h
    if not legend: kw["showlegend"] = False
    fig.update_layout(**kw)
    return fig

# ══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════
DATA_DIR = Path(__file__).parent / "data"

@st.cache_data(show_spinner=False)
def load_listings() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "listings.csv", low_memory=False)

    # ── Price: numeric, filter outliers ──
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].between(200, 80000)].copy()

    # ── Room type Chinese ──
    df["room_type_zh"] = df["room_type"].map(ROOM_JP).fillna(df["room_type"])

    # ── Fill NaN ──
    df["reviews_per_month"]    = df["reviews_per_month"].fillna(0)
    df["number_of_reviews_ltm"] = df["number_of_reviews_ltm"].fillna(0)

    # ── Derived: occupancy proxy ──
    df["occupancy_pct"] = ((365 - df["availability_365"]) / 365 * 100).clip(0, 100).round(1)

    # ── Unsold Risk Score (Weighted Scoring Model) ──
    # Component A: Availability rate          — high avail → high risk  (40%)
    a = (df["availability_365"] / 365).clip(0, 1)
    # Component B: Cumulative review absence  — few reviews → high risk (30%)
    b = 1 - np.clip(df["number_of_reviews"] / 100, 0, 1)
    # Component C: Recent 12-month inactivity — low LTM    → high risk  (20%)
    c = 1 - np.clip(df["number_of_reviews_ltm"] / 20, 0, 1)
    # Component D: Price premium              — high price → risk       (10%)
    p95 = df["price"].quantile(0.95)
    d   = np.clip(df["price"] / p95, 0, 1)

    df["risk_score"] = (0.40*a + 0.30*b + 0.20*c + 0.10*d).clip(0, 1).round(3)
    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-0.001, 0.35, 0.60, 1.001],
        labels=["低風險", "中風險", "高風險"],
    ).astype(str)

    # ── last_review year ──
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    df["last_review_year"] = df["last_review"].dt.year.fillna(0).astype(int)

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_monthly_calendar() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "monthly_calendar.csv")


@st.cache_data(show_spinner=False)
def load_monthly_reviews() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "monthly_reviews.csv")


@st.cache_data(show_spinner=False)
def nb_agg(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("neighbourhood").agg(
            房源數     = ("id",                     "count"),
            高風險數   = ("risk_level",              lambda x: (x=="高風險").sum()),
            中風險數   = ("risk_level",              lambda x: (x=="中風險").sum()),
            低風險數   = ("risk_level",              lambda x: (x=="低風險").sum()),
            平均風險   = ("risk_score",              "mean"),
            中位價格   = ("price",                  "median"),
            平均評論數 = ("number_of_reviews",        "mean"),
            平均入住率 = ("occupancy_pct",            "mean"),
            平均可訂率 = ("availability_365",         lambda x: (x/365*100).mean()),
        ).reset_index()
    )
    g["高風險佔比"] = (g["高風險數"] / g["房源數"] * 100).round(1)
    g["平均風險"]   = g["平均風險"].round(3)
    g["中位價格"]   = g["中位價格"].round(0).astype(int)
    g["平均評論數"] = g["平均評論數"].round(1)
    g["平均入住率"] = g["平均入住率"].round(1)
    g["平均可訂率"] = g["平均可訂率"].round(1)
    return g.sort_values("高風險佔比", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════════
with st.spinner("載入資料 …"):
    DF_ALL = load_listings()
    DF_CAL = load_monthly_calendar()
    DF_REV = load_monthly_reviews()

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR · FILTERS
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:8px 0 18px;">
      <div style="font-size:1.08rem;font-weight:700;color:{P['ink']};
           letter-spacing:-.3px;">🏯 旅宿滯銷風險預警</div>
      <div style="font-size:.74rem;color:{P['muted']};margin-top:3px;
           letter-spacing:.04em;">Inside Airbnb · 台北市</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sec">篩選條件</div>', unsafe_allow_html=True)

    all_nb  = sorted(DF_ALL["neighbourhood"].dropna().unique())
    sel_nb  = st.multiselect("🗺 行政區", all_nb, default=all_nb,
                              placeholder="全部行政區")

    all_rt  = sorted(DF_ALL["room_type_zh"].dropna().unique())
    sel_rt  = st.multiselect("🛏 房型", all_rt, default=all_rt,
                              placeholder="全部房型")

    sel_risk = st.multiselect(
        "⚠️ 風險等級",
        ["高風險", "中風險", "低風險"],
        default=["高風險", "中風險", "低風險"],
    )

    p_lo, p_hi = 200, int(DF_ALL["price"].quantile(.98))
    sel_p = st.slider("💰 每晚價格（TWD）", p_lo, p_hi, (p_lo, p_hi), step=200)

    sel_min_rev = st.slider("💬 最低評論數", 0, 100, 0)

    st.divider()
    st.markdown(f'<div class="sec">趨勢時段</div>', unsafe_allow_html=True)
    all_rev_months = sorted(DF_REV["month"].unique())
    idx_default    = max(0, len(all_rev_months) - 24)
    trend_from     = st.selectbox("起始月份", all_rev_months,
                                   index=idx_default, key="tf")
    st.divider()
    st.caption(f"台北市 {len(DF_ALL):,} 筆真實房源\n"
               f"資料：Inside Airbnb 2025-09\n"
               f"© 2026 智慧旅宿平台")

# ── Apply filters ────────────────────────────────────────────────
mask = (
    DF_ALL["neighbourhood"].isin(sel_nb) &
    DF_ALL["room_type_zh"].isin(sel_rt) &
    DF_ALL["risk_level"].isin(sel_risk) &
    DF_ALL["price"].between(*sel_p) &
    (DF_ALL["number_of_reviews"] >= sel_min_rev)
)
df  = DF_ALL[mask].copy()
NB  = nb_agg(df)
CAL = DF_CAL.copy()
REV = DF_REV[DF_REV["month"] >= trend_from].copy()

# ══════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ══════════════════════════════════════════════════════════════════
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(f"""
    <div style="padding:4px 0 2px;">
      <h1 style="font-size:1.5rem;font-weight:700;color:{P['ink']};
           margin:0;letter-spacing:-.4px;">
        智慧旅宿「滯銷風險」預警平台
      </h1>
      <p style="font-size:.82rem;color:{P['muted']};margin:4px 0 0;
           letter-spacing:.02em;">
        台北市 Inside Airbnb 真實資料 &ensp;｜&ensp;
        加權評分模型 × 地理空間 × 時間序列 &ensp;｜&ensp;
        2025–2026
      </p>
    </div>
    """, unsafe_allow_html=True)
with h2:
    st.markdown(f"""
    <div style="text-align:right;padding-top:6px;">
      <span style="font-size:.75rem;color:{P['muted']};">篩選中</span><br>
      <span style="font-size:1.75rem;font-weight:700;
           color:{P['primary']};">{len(df):,}</span>
      <span style="font-size:.75rem;color:{P['muted']};"> 筆</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr style="margin:10px 0 16px;">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════
T1,T2,T3,T4,T5 = st.tabs([
    "概覽儀表板", "風險地圖", "趨勢分析", "行政區比較", "房源明細"
])

# ╔═══════════════════════╗
# ║  TAB 1 · 概覽儀表板   ║
# ╚═══════════════════════╝
with T1:
    total = len(df)
    h_n = (df["risk_level"]=="高風險").sum()
    m_n = (df["risk_level"]=="中風險").sum()
    l_n = (df["risk_level"]=="低風險").sum()

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("篩選房源",    f"{total:,}")
    k2.metric("🔴 高風險",  f"{h_n:,}",
              f"{h_n/total*100:.1f}%" if total else "–", delta_color="inverse")
    k3.metric("🟡 中風險",  f"{m_n:,}")
    k4.metric("🟢 低風險",  f"{l_n:,}")
    k5.metric("平均風險分數", f"{df['risk_score'].mean():.3f}")
    k6.metric("平均入住率",  f"{df['occupancy_pct'].mean():.1f}%")

    st.divider()

    # Row 1
    r1a, r1b, r1c = st.columns([1.1, 1.5, 1.8])

    with r1a:
        sec("風險等級佔比")
        method_badge("佔比分析", "Proportional Analysis")
        rc = (df["risk_level"].value_counts()
                .reindex(["高風險","中風險","低風險"]).reset_index())
        rc.columns = ["風險等級","數量"]
        fig = px.pie(rc, values="數量", names="風險等級",
                     color="風險等級", color_discrete_map=RC, hole=0.58)
        fig.update_traces(
            textfont=dict(size=11),
            marker_line_width=2, marker_line_color=P["bg"],
            pull=[0.05 if r=="高風險" else 0 for r in rc["風險等級"]],
        )
        T(fig, h=255, legend=True).update_layout(
            margin=dict(l=5,r=5,t=5,b=5),
            legend=dict(orientation="v", x=1, y=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)

    with r1b:
        sec("房型 × 風險等級")
        method_badge("多變量類別分佈", "Multivariate Categorical Distribution")
        rt_risk = (df.groupby(["room_type_zh","risk_level"]).size()
                     .reset_index(name="數量"))
        fig = px.bar(rt_risk, x="room_type_zh", y="數量", color="risk_level",
                     color_discrete_map=RC, barmode="stack",
                     category_orders={"risk_level":["低風險","中風險","高風險"]},
                     labels={"room_type_zh":"","數量":"房源數","risk_level":""})
        fig.update_traces(marker_line_width=0)
        T(fig, h=255).update_layout(margin=dict(l=30,r=10,t=5,b=40))
        st.plotly_chart(fig, use_container_width=True)

    with r1c:
        sec("每晚價格 vs 滯銷風險分數")
        method_badge("雙變量相關分析 · LOWESS 趨勢線", "Bivariate Correlation · LOWESS")
        samp = (df[df["price"] < df["price"].quantile(.96)]
                  .sample(min(900,len(df)), random_state=1))
        fig = px.scatter(
            samp, x="price", y="risk_score",
            color="risk_level", color_discrete_map=RC, opacity=0.52,
            hover_name="neighbourhood",
            hover_data={"price":True,"risk_score":":.3f",
                        "number_of_reviews":True,"risk_level":False},
            trendline="lowess", trendline_color_override=P["primary"],
            labels={"price":"每晚價格(TWD)","risk_score":"滯銷風險分數","risk_level":""},
        )
        fig.update_traces(marker=dict(size=5, line=dict(width=0)),
                          selector=dict(mode="markers"))
        T(fig, h=255).update_layout(margin=dict(l=44,r=10,t=5,b=40))
        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    r2a, r2b = st.columns(2)
    with r2a:
        sec("每晚價格頻率分佈")
        method_badge("頻率分佈分析", "Frequency Distribution Analysis")
        pdat = df[df["price"] < df["price"].quantile(.97)]
        fig = px.histogram(
            pdat, x="price", nbins=48,
            color="risk_level", color_discrete_map=RC,
            barmode="overlay", opacity=0.68,
            labels={"price":"每晚價格(TWD)","count":"房源數","risk_level":""},
        )
        fig.update_traces(marker_line_width=0)
        T(fig, h=240).update_layout(margin=dict(l=44,r=10,t=5,b=36))
        st.plotly_chart(fig, use_container_width=True)

    with r2b:
        sec("入住率 × 風險等級分佈")
        method_badge("核密度估計 (KDE)", "Kernel Density Estimation")
        fig = px.violin(
            df, x="risk_level", y="occupancy_pct",
            color="risk_level", color_discrete_map=RC,
            box=True, points=False,
            category_orders={"risk_level":["低風險","中風險","高風險"]},
            labels={"risk_level":"","occupancy_pct":"入住率 (%)"},
        )
        T(fig, h=240, legend=False).update_layout(margin=dict(l=44,r=10,t=5,b=36))
        st.plotly_chart(fig, use_container_width=True)

    # Risk model note
    note("""
    <b>加權評分模型（Weighted Scoring Model）</b><br>
    滯銷風險分數 = <b>0.40</b> × 可訂率（availability_365/365）
    ＋ <b>0.30</b> × 評論稀疏度（1 − reviews/100）
    ＋ <b>0.20</b> × 近12月活躍度缺失（1 − reviews_ltm/20）
    ＋ <b>0.10</b> × 相對價格偏高（price/P95）
    &emsp;→&emsp; 0–0.35: 低風險 ｜ 0.35–0.60: 中風險 ｜ 0.60–1.0: 高風險
    """)

# ╔═══════════════════════╗
# ║  TAB 2 · 風險地圖     ║
# ╚═══════════════════════╝
with T2:
    sec("台北市 Airbnb 滯銷風險熱點地圖")
    method_badge("地理空間分析 · 散佈地圖", "Geospatial Analysis · Scatter Map")

    mc1, mc2, mc3 = st.columns([1,1,1])
    color_by = mc1.selectbox("著色依據", ["風險等級","房型","入住率(%)"], key="cb")
    size_by  = mc2.radio("點大小", ["一致","依風險分數"], horizontal=True, key="sb")
    n_show   = mc3.slider("顯示筆數", 300, min(2000,len(df)),
                           min(1200,len(df)), step=100, key="ns")

    geo = df.dropna(subset=["latitude","longitude"]).sample(
        min(n_show, len(df)), random_state=2)

    if color_by == "風險等級":
        c_kw = dict(color="risk_level", color_discrete_map=RC,
                    category_orders={"risk_level":["高風險","中風險","低風險"]})
    elif color_by == "房型":
        c_kw = dict(color="room_type_zh", color_discrete_map=RT_C)
    else:
        c_kw = dict(color="occupancy_pct",
                    color_continuous_scale=["#FDECEA","#FDF5E4","#EAF5EE"],
                    range_color=[0,100])

    s_kw = dict(size="risk_score", size_max=14) if size_by=="依風險分數" else {}

    fig = px.scatter_mapbox(
        geo, lat="latitude", lon="longitude",
        hover_name="name",
        hover_data={"neighbourhood":True,"room_type_zh":True,"price":True,
                    "risk_score":":.3f","number_of_reviews":True,
                    "occupancy_pct":":.1f","latitude":False,"longitude":False},
        mapbox_style="carto-positron",
        zoom=11, opacity=0.82, height=560,
        center={"lat":25.047,"lon":121.517},
        labels={"risk_level":"風險","room_type_zh":"房型",
                "occupancy_pct":"入住率%"},
        **c_kw, **s_kw,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0),
        legend=dict(bgcolor=P["surface"], bordercolor=P["border"], borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mini nb table
    nb_mini = NB[["neighbourhood","房源數","高風險佔比","平均風險","平均入住率"]].copy()
    nb_mini.columns = ["行政區","房源數","高風險佔比(%)","平均風險分數","平均入住率(%)"]

    def _risk_css(v):
        if isinstance(v,(int,float)):
            if v>=60: return f"color:{P['high']};font-weight:700;"
            if v>=40: return f"color:{P['medium']};font-weight:600;"
        return ""

    html_table(nb_mini,
               fmt={"高風險佔比(%)":"{:.1f}","平均風險分數":"{:.3f}","平均入住率(%)":"{:.1f}"},
               cell_fn={"高風險佔比(%)": _risk_css},
               height=260)

# ╔═══════════════════════╗
# ║  TAB 3 · 趨勢分析     ║
# ╚═══════════════════════╝
with T3:
    CAL_F = CAL[CAL["month"] >= trend_from].copy()

    r1, r2 = st.columns(2)

    with r1:
        sec("月度平均入住率 (%)")
        method_badge("時間序列分析 · 面積圖", "Time Series Analysis · Area Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=CAL_F["month"], y=CAL_F["avg_occupancy_pct"],
            mode="lines+markers",
            line=dict(color=P["low"], width=2.5),
            marker=dict(size=6, color=P["low"],
                        line=dict(width=1.5, color=P["surface"])),
            fill="tozeroy", fillcolor="rgba(91,158,115,0.10)",
            hovertemplate="%{x}<br>入住率：%{y:.1f}%<extra></extra>",
        ))
        fig.add_hrect(y0=0, y1=35, fillcolor=P["high"], opacity=0.04,
                      annotation_text="低入住危險區 (<35%)",
                      annotation_font=dict(size=9, color=P["high"]),
                      annotation_position="top left")
        T(fig, h=290, legend=False).update_layout(
            yaxis=dict(range=[0,100], title="入住率 (%)"), xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        sec("月度可訂率（滯銷閒置率）%")
        method_badge("時間序列分析 · 面積圖", "Time Series Analysis · Area Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=CAL_F["month"], y=CAL_F["avg_availability_pct"],
            mode="lines+markers",
            line=dict(color=P["high"], width=2.5),
            marker=dict(size=6, color=P["high"],
                        line=dict(width=1.5, color=P["surface"])),
            fill="tozeroy", fillcolor="rgba(196,100,90,0.08)",
            hovertemplate="%{x}<br>可訂率：%{y:.1f}%<extra></extra>",
        ))
        T(fig, h=290, legend=False).update_layout(
            yaxis=dict(range=[0,100], title="可訂率 (%)"), xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    r3, r4 = st.columns(2)

    with r3:
        sec("月度評論數（市場活躍度）")
        method_badge("時序頻率分析 · 長條圖", "Temporal Frequency Analysis · Bar Chart")
        rev_plot = REV[REV["month"] <= "2025-09"].copy()
        bar_cols  = [P["high"] if v < 3000 else P["primary"]
                     for v in rev_plot["review_count"]]
        fig = go.Figure(go.Bar(
            x=rev_plot["month"], y=rev_plot["review_count"],
            marker=dict(color=bar_cols, line=dict(width=0)),
            hovertemplate="%{x}<br>評論數：%{y:,}<extra></extra>",
        ))
        T(fig, h=270, legend=False).update_layout(
            yaxis_title="評論數", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with r4:
        sec("入住率 vs 可訂率 雙指標對比")
        method_badge("雙指標時序比較", "Dual-Metric Time Series Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=CAL_F["month"], y=CAL_F["avg_occupancy_pct"],
            name="入住率", mode="lines",
            line=dict(color=P["low"], width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=CAL_F["month"], y=CAL_F["avg_availability_pct"],
            name="可訂率（滯銷）", mode="lines",
            line=dict(color=P["high"], width=2.5, dash="dot"),
        ))
        T(fig, h=270).update_layout(
            yaxis=dict(range=[0,100], title="%"), xaxis_title="",
            legend=dict(orientation="h", y=1.08, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    note("月度入住率 / 可訂率來自 <b>calendar.csv</b>（真實訂房日曆資料）；"
         "評論數來自 <b>reviews.csv</b>，作為市場活躍度的代理指標（Proxy Metric）。")

# ╔═══════════════════════╗
# ║  TAB 4 · 行政區比較   ║
# ╚═══════════════════════╝
with T4:
    a1, a2 = st.columns(2)

    with a1:
        sec("各行政區高風險比例排名")
        method_badge("排名比較分析 · 水平長條圖", "Ranking Comparative Analysis · Horizontal Bar")
        nb_s = NB.sort_values("高風險佔比", ascending=True)
        fig  = go.Figure(go.Bar(
            x=nb_s["高風險佔比"], y=nb_s["neighbourhood"],
            orientation="h",
            marker=dict(
                color=nb_s["高風險佔比"],
                colorscale=[[0,P["low"]],[0.4,P["medium"]],[1,P["high"]]],
                cmin=0, cmax=85, line=dict(width=0),
            ),
            text=nb_s["高風險佔比"].map("{:.1f}%".format),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>高風險佔比：%{x:.1f}%<extra></extra>",
        ))
        T(fig, h=370, legend=False).update_layout(
            xaxis=dict(range=[0,95], title="高風險比例 (%)"),
            yaxis_title="", margin=dict(l=64,r=55,t=5,b=36),
        )
        st.plotly_chart(fig, use_container_width=True)

    with a2:
        sec("行政區多維散佈（中位價格 × 平均風險 × 房源數）")
        method_badge("多維散佈分析 · 氣泡圖", "Multi-dimensional Scatter · Bubble Chart")
        fig = px.scatter(
            NB, x="中位價格", y="平均風險",
            size="房源數", color="高風險佔比",
            color_continuous_scale=[[0,P["low"]],[0.5,P["medium"]],[1,P["high"]]],
            text="neighbourhood",
            hover_data={"房源數":True,"平均入住率":True,"平均評論數":True},
            size_max=60,
            labels={"中位價格":"中位每晚價格(TWD)","平均風險":"平均滯銷風險分數",
                    "高風險佔比":"高風險佔比(%)"},
        )
        fig.update_traces(
            textposition="top center",
            textfont=dict(size=10, color=P["ink"]),
            marker=dict(line=dict(width=1.5, color=P["surface"])),
        )
        T(fig, h=370).update_layout(
            coloraxis_colorbar=dict(title="高風險%", len=0.65),
            margin=dict(l=44,r=10,t=5,b=36),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Stacked 100% bar
    sec("各行政區房型成分比例")
    method_badge("成分比例分析 · 100% 堆疊長條圖", "Compositional Analysis · 100% Stacked Bar")
    nb_rt = df.groupby(["neighbourhood","room_type_zh"]).size().reset_index(name="n")
    tot   = nb_rt.groupby("neighbourhood")["n"].transform("sum")
    nb_rt["pct"] = (nb_rt["n"] / tot * 100).round(1)
    fig = px.bar(
        nb_rt, x="neighbourhood", y="pct", color="room_type_zh",
        color_discrete_map=RT_C, barmode="stack",
        text=nb_rt["pct"].map("{:.0f}%".format),
        labels={"neighbourhood":"","pct":"佔比 (%)","room_type_zh":"房型"},
        category_orders={"room_type_zh":["整棟出租","私人套房","共用套房","飯店客房"]},
    )
    fig.update_traces(marker_line_width=0, textposition="inside",
                      textfont=dict(color="white", size=9))
    T(fig, h=270).update_layout(
        yaxis=dict(range=[0,103], title=""),
        margin=dict(l=10,r=10,t=5,b=36),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    sec("行政區統計彙整")
    method_badge("描述性統計", "Descriptive Statistics")
    tbl = NB[["neighbourhood","房源數","高風險數","高風險佔比",
               "平均風險","中位價格","平均入住率","平均評論數"]].copy()
    tbl.columns = ["行政區","房源數","高風險數","高風險佔比(%)","平均風險分數",
                   "中位價格(TWD)","平均入住率(%)","平均評論數"]

    html_table(
        tbl,
        fmt={"高風險佔比(%)":"{:.1f}","平均風險分數":"{:.3f}",
             "平均入住率(%)":"{:.1f}","平均評論數":"{:.1f}"},
        cell_fn={"高風險佔比(%)": lambda v:
                 f"color:{P['high']};font-weight:700;" if isinstance(v,(int,float)) and v>=60
                 else (f"color:{P['medium']};font-weight:600;" if isinstance(v,(int,float)) and v>=40 else "")},
        height=320,
    )

# ╔═══════════════════════╗
# ║  TAB 5 · 房源明細     ║
# ╚═══════════════════════╝
with T5:
    d1, d2 = st.columns([2,1])
    with d1:
        sort_by = st.selectbox(
            "排序欄位",
            ["risk_score","price","number_of_reviews","occupancy_pct","availability_365"],
            format_func=lambda x: {
                "risk_score":"滯銷風險分數","price":"每晚價格",
                "number_of_reviews":"評論數","occupancy_pct":"入住率",
                "availability_365":"年度可訂天數",
            }[x],
        )
    with d2:
        asc = st.radio("排序", ["由高→低","由低→高"], horizontal=True) == "由低→高"

    SHOW_MAP = {
        "id":"ID","neighbourhood":"行政區","room_type_zh":"房型",
        "price":"每晚價格(TWD)","number_of_reviews":"評論數",
        "number_of_reviews_ltm":"近12月評論","occupancy_pct":"入住率(%)",
        "availability_365":"年度可訂天數","risk_score":"滯銷風險分數","risk_level":"風險等級",
    }
    avail = [c for c in SHOW_MAP if c in df.columns]
    df_s  = (df.sort_values(sort_by, ascending=asc)[avail]
               .head(300).reset_index(drop=True))
    df_s.index = df_s.index + 1
    df_s.columns = [SHOW_MAP[c] for c in avail]

    def _r_css(v):
        if v=="高風險": return f"background:#FDECEA;color:{P['high']};font-weight:700;"
        if v=="中風險": return f"background:#FDF5E4;color:#A07A20;font-weight:700;"
        if v=="低風險": return f"background:#EAF5EE;color:#3D7A55;font-weight:700;"
        return ""
    def _s_css(v):
        if isinstance(v,float) and 0<=v<=1:
            if v>=.60: return f"color:{P['high']};font-weight:700;"
            if v>=.35: return f"color:{P['medium']};font-weight:600;"
            return f"color:{P['low']};"
        return ""

    sec("篩選房源明細（最多顯示 300 筆）")
    method_badge("描述性統計 · 多條件篩選排序", "Descriptive Statistics · Multi-filter Sorting")
    html_table(
        df_s,
        fmt={"滯銷風險分數":"{:.3f}","入住率(%)":"{:.1f}"},
        cell_fn={"風險等級":_r_css, "滯銷風險分數":_s_css},
        height=480,
    )

    csv_out = df_s.to_csv(index=False, encoding="utf-8-sig").encode()
    st.download_button("⬇ 下載篩選結果 CSV", data=csv_out,
                       file_name=f"risk_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")

    st.divider()
    e1, e2 = st.columns(2)

    with e1:
        sec("高風險 TOP 10 房源")
        method_badge("極值排序分析", "Extremum Ranking Analysis")
        top10h = (df[df["risk_level"]=="高風險"]
                  .sort_values("risk_score", ascending=False).head(10))
        if not top10h.empty:
            fig = go.Figure(go.Bar(
                x=top10h["risk_score"],
                y=[f"{r.neighbourhood} · #{r.id}" for r in top10h.itertuples()],
                orientation="h",
                marker=dict(color=P["high"], line=dict(width=0)),
                text=top10h["risk_score"].map("{:.3f}".format),
                textposition="outside",
            ))
            T(fig, h=310, legend=False).update_layout(
                xaxis=dict(range=[0,1.08],title="滯銷風險分數"),
                yaxis_title="", margin=dict(l=130,r=55,t=5,b=36),
            )
            st.plotly_chart(fig, use_container_width=True)

    with e2:
        sec("低風險 TOP 10 優良房源")
        method_badge("極值排序分析", "Extremum Ranking Analysis")
        top10l = (df[df["risk_level"]=="低風險"]
                  .sort_values("risk_score", ascending=True).head(10))
        if not top10l.empty:
            fig = go.Figure(go.Bar(
                x=top10l["risk_score"],
                y=[f"{r.neighbourhood} · #{r.id}" for r in top10l.itertuples()],
                orientation="h",
                marker=dict(color=P["low"], line=dict(width=0)),
                text=top10l["risk_score"].map("{:.3f}".format),
                textposition="outside",
            ))
            T(fig, h=310, legend=False).update_layout(
                xaxis=dict(range=[0,.45],title="滯銷風險分數"),
                yaxis_title="", margin=dict(l=130,r=55,t=5,b=36),
            )
            st.plotly_chart(fig, use_container_width=True)
