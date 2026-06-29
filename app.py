"""
智慧旅宿「滯銷風險」預警平台  ·  Inside Airbnb Taipei
日系簡約 · 淡色系 · 每圖標示分析方法 + 真實 ML 指標
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── sklearn ──
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score,
)

# ═══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="旅宿滯銷風險預警", page_icon="🏯",
                   layout="wide", initial_sidebar_state="expanded")

# ─── Design tokens ──────────────────────────────────────────────
P = dict(
    bg="F8F7F5", surface="#FFFFFF", card="#FDFCFA",
    border="#E8E4DE", border2="#D4CFC8",
    ink="#2A2A2A", ink2="#505050", muted="#9A9490",
    primary="#4E7FB0", accent="#8B7BA8",
    high="#C4645A", medium="#C49A4A", low="#5B9E73",
    tag_bg="#F2F0EC", mbg="#EEF4FB", mtxt="#3D6B96",
)
P["bg"] = "#" + P["bg"]
RC  = {"高風險": P["high"],   "中風險": P["medium"],  "低風險": P["low"]}
RTC = {"整棟出租": P["primary"],"私人套房": P["accent"],
       "共用套房": P["medium"], "飯店客房": P["low"]}
ROOM_JP = {"Entire home/apt":"整棟出租","Private room":"私人套房",
           "Shared room":"共用套房",   "Hotel room":"飯店客房"}
FEAT_ZH = {
    "availability_365":             "年度可訂天數",
    "number_of_reviews":            "評論總數",
    "number_of_reviews_ltm":        "近12月評論數",
    "reviews_per_month":            "月均評論數",
    "price":                        "每晚價格",
    "calculated_host_listings_count":"房東房源數",
    "minimum_nights":               "最少入住晚數",
    "rt_Entire home/apt":           "房型：整棟",
    "rt_Shared room":               "房型：共用",
    "rt_Private room":              "房型：私人",
    "rt_Hotel room":                "房型：飯店",
}

# ─── CSS ────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');
html,body,.stApp{{background:{P['bg']};color:{P['ink']};
  font-family:'Noto Sans TC',sans-serif;}}
section[data-testid="stSidebar"]{{background:{P['surface']};
  border-right:1px solid {P['border']};}}
[data-testid="stMetric"]{{background:{P['surface']};border:1px solid {P['border']};
  border-radius:10px;padding:15px 18px;box-shadow:0 1px 3px rgba(0,0,0,.04);}}
[data-testid="stMetricLabel"]{{color:{P['muted']} !important;
  font-size:.72rem !important;letter-spacing:.08em;text-transform:uppercase;}}
[data-testid="stMetricValue"]{{color:{P['ink']} !important;
  font-size:1.45rem !important;font-weight:700;}}
.stTabs [data-baseweb="tab-list"]{{background:transparent;
  border-bottom:2px solid {P['border']};gap:0;padding:0;}}
.stTabs [data-baseweb="tab"]{{color:{P['muted']};border-radius:0;
  padding:9px 20px;border-bottom:2px solid transparent;margin-bottom:-2px;
  font-size:.85rem;font-weight:500;}}
.stTabs [aria-selected="true"]{{color:{P['primary']} !important;
  border-bottom:2px solid {P['primary']} !important;background:transparent !important;}}
section[data-testid="stSidebar"] label{{color:{P['ink2']} !important;font-size:.80rem;}}
.sec{{font-size:.71rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:{P['muted']};margin:18px 0 3px;padding-bottom:7px;
  border-bottom:1px solid {P['border']};}}
.mb{{display:inline-flex;align-items:center;gap:4px;
  background:{P['mbg']};border:1px solid #C8DCF0;border-radius:5px;
  padding:3px 10px;font-size:.70rem;font-weight:600;color:{P['mtxt']};
  letter-spacing:.03em;margin-bottom:7px;}}
.mhigh{{background:#FEF2F0;border:1px solid #F0B8B4;color:#A03028;}}
.note{{background:{P['tag_bg']};border-left:3px solid {P['primary']};
  padding:9px 14px;border-radius:0 6px 6px 0;
  font-size:.79rem;color:{P['ink2']};margin:8px 0;}}
hr{{border:none;border-top:1px solid {P['border']} !important;margin:14px 0;}}
::-webkit-scrollbar{{width:4px;}}
::-webkit-scrollbar-thumb{{background:{P['border2']};border-radius:2px;}}
</style>
""", unsafe_allow_html=True)

# ─── UI helpers ─────────────────────────────────────────────────
def sec(t): st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)

def mb(text, warning=False):
    cls = "mb mhigh" if warning else "mb"
    st.markdown(f'<span class="{cls}">📐 {text}</span>', unsafe_allow_html=True)

def note(t): st.markdown(f'<div class="note">{t}</div>', unsafe_allow_html=True)

def html_table(df_in, fmt=None, cell_fn=None, height=360):
    fmt = fmt or {};  cell_fn = cell_fn or {}
    th = (f"background:{P['tag_bg']};color:{P['muted']};font-size:.70rem;"
          f"letter-spacing:.07em;text-transform:uppercase;padding:8px 13px;"
          f"border-bottom:2px solid {P['border2']};white-space:nowrap;"
          f"position:sticky;top:0;z-index:1;")
    td0 = f"padding:7px 13px;font-size:.80rem;color:{P['ink']};border-bottom:1px solid {P['border']};white-space:nowrap;"
    hdr = "".join(f'<th style="{th}">{c}</th>' for c in df_in.columns)
    rows = []
    for i, (_, row) in enumerate(df_in.iterrows()):
        bg = P["surface"] if i%2==0 else P["tag_bg"]
        cells = []
        for col in df_in.columns:
            v    = row[col]
            disp = "–" if pd.isna(v) else (fmt[col].format(v) if col in fmt and pd.notna(v) else str(v))
            css  = f"background:{bg};"
            if col in cell_fn:
                try: css += cell_fn[col](v)
                except: pass
            cells.append(f'<td style="{td0}{css}">{disp}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        f'<div style="overflow:auto;max-height:{height}px;border:1px solid {P["border"]};'
        f'border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.04);">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>{hdr}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>',
        unsafe_allow_html=True)

# ─── Chart base ─────────────────────────────────────────────────
_L = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
          font=dict(color=P["ink2"], family="Noto Sans TC,sans-serif", size=11),
          margin=dict(l=46,r=16,t=28,b=34),
          legend=dict(bgcolor="rgba(255,255,255,.8)", bordercolor=P["border"],
                      borderwidth=1, font=dict(color=P["ink2"])),
          xaxis=dict(gridcolor=P["border"], linecolor=P["border"], zeroline=False,
                     tickfont=dict(color=P["muted"])),
          yaxis=dict(gridcolor=P["border"], linecolor=P["border"], zeroline=False,
                     tickfont=dict(color=P["muted"])))
def T(fig, h=None, legend=True):
    kw = dict(**_L)
    if h: kw["height"]=h
    if not legend: kw["showlegend"]=False
    return fig.update_layout(**kw)

# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════
DATA = Path(__file__).parent / "data"

@st.cache_data(show_spinner=False)
def load_listings():
    df = pd.read_csv(DATA/"listings.csv", low_memory=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].between(200, 80000)].copy()
    df["room_type_zh"] = df["room_type"].map(ROOM_JP).fillna(df["room_type"])
    df["reviews_per_month"]     = df["reviews_per_month"].fillna(0)
    df["number_of_reviews_ltm"] = df["number_of_reviews_ltm"].fillna(0)
    df["occupancy_pct"] = ((365 - df["availability_365"]) / 365 * 100).clip(0,100).round(1)
    # Weighted scoring model
    a = (df["availability_365"] / 365).clip(0,1)
    b = 1 - np.clip(df["number_of_reviews"] / 100, 0, 1)
    c = 1 - np.clip(df["number_of_reviews_ltm"] / 20, 0, 1)
    d = np.clip(df["price"] / df["price"].quantile(0.95), 0, 1)
    df["risk_score"] = (0.40*a + 0.30*b + 0.20*c + 0.10*d).clip(0,1).round(3)
    df["risk_level"] = pd.cut(df["risk_score"], bins=[-0.001,0.35,0.60,1.001],
                               labels=["低風險","中風險","高風險"]).astype(str)
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_calendar():
    return pd.read_csv(DATA/"monthly_calendar.csv")

@st.cache_data(show_spinner=False)
def load_reviews():
    return pd.read_csv(DATA/"monthly_reviews.csv")

# ─── ML model training ─────────────────────────────────────────
FEAT_COLS = ["price","minimum_nights","number_of_reviews","reviews_per_month",
             "calculated_host_listings_count","availability_365","number_of_reviews_ltm"]

@st.cache_data(show_spinner=False)
def train_models(_df):
    rt_dum = pd.get_dummies(_df["room_type"], prefix="rt")
    X = pd.concat([_df[FEAT_COLS].fillna(0), rt_dum], axis=1)
    y = (_df["risk_level"] == "高風險").astype(int)
    feat_names = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    # ── Logistic Regression ──
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_tr_s, y_tr)
    lr_prob = lr.predict_proba(X_te_s)[:,1]
    lr_pred = lr.predict(X_te_s)

    # ── Random Forest ──
    rf = RandomForestClassifier(n_estimators=100, random_state=42,
                                 class_weight="balanced", n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_prob = rf.predict_proba(X_te)[:,1]
    rf_pred = rf.predict(X_te)

    def m(yt, yp, yprob):
        fpr, tpr, _ = roc_curve(yt, yprob)
        prec, rec, _ = precision_recall_curve(yt, yprob)
        return dict(
            accuracy  = accuracy_score(yt, yp),
            recall    = recall_score(yt, yp),
            precision = precision_score(yt, yp),
            f1        = f1_score(yt, yp),
            auc       = roc_auc_score(yt, yprob),
            ap        = average_precision_score(yt, yprob),
            cm        = confusion_matrix(yt, yp),
            fpr=fpr, tpr=tpr,
            prec=prec, rec=rec,
        )

    corr_df = X_tr.copy()
    corr_df.columns = [FEAT_ZH.get(c,c) for c in corr_df.columns]

    return dict(
        lr = m(y_te, lr_pred, lr_prob),
        rf = m(y_te, rf_pred, rf_prob),
        lr_coef   = dict(zip(feat_names, lr.coef_[0])),
        rf_import = dict(zip(feat_names, rf.feature_importances_)),
        feat_names = feat_names,
        corr = corr_df.corr(),
        n_train = len(X_tr), n_test = len(X_te),
        n_pos = int(y.sum()), n_neg = int((1-y).sum()),
    )

@st.cache_data(show_spinner=False)
def nb_agg(_df):
    g = (_df.groupby("neighbourhood").agg(
        房源數     =("id","count"),
        高風險數   =("risk_level", lambda x:(x=="高風險").sum()),
        平均風險   =("risk_score","mean"),
        中位價格   =("price","median"),
        平均評論數 =("number_of_reviews","mean"),
        平均入住率 =("occupancy_pct","mean"),
    ).reset_index())
    g["高風險佔比"] = (g["高風險數"]/g["房源數"]*100).round(1)
    g["平均風險"]   = g["平均風險"].round(3)
    g["中位價格"]   = g["中位價格"].round(0).astype(int)
    g["平均評論數"] = g["平均評論數"].round(1)
    g["平均入住率"] = g["平均入住率"].round(1)
    return g.sort_values("高風險佔比", ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════
#  LOAD
# ═══════════════════════════════════════════════════════════════════
with st.spinner("載入資料與訓練模型 …"):
    DF_ALL = load_listings()
    DF_CAL = load_calendar()
    DF_REV = load_reviews()
    MDL    = train_models(DF_ALL)

LR = MDL["lr"];  RF = MDL["rf"]

# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:6px 0 16px;">
      <div style="font-size:1.05rem;font-weight:700;color:{P['ink']};">
        🏯 旅宿滯銷風險預警</div>
      <div style="font-size:.73rem;color:{P['muted']};margin-top:2px;
           letter-spacing:.04em;">Inside Airbnb · 台北市 · {len(DF_ALL):,} 筆</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="sec">篩選條件</div>', unsafe_allow_html=True)
    all_nb = sorted(DF_ALL["neighbourhood"].dropna().unique())
    sel_nb = st.multiselect("🗺 行政區", all_nb, default=all_nb, placeholder="全部")
    all_rt = sorted(DF_ALL["room_type_zh"].dropna().unique())
    sel_rt = st.multiselect("🛏 房型", all_rt, default=all_rt, placeholder="全部")
    sel_risk = st.multiselect("⚠️ 風險等級",
        ["高風險","中風險","低風險"], default=["高風險","中風險","低風險"])
    p_lo, p_hi = 200, int(DF_ALL["price"].quantile(.98))
    sel_p = st.slider("💰 每晚價格（TWD）", p_lo, p_hi, (p_lo, p_hi), step=200)
    sel_min_rev = st.slider("💬 最低評論數", 0, 100, 0)

    st.divider()
    st.markdown(f'<div class="sec">趨勢時段</div>', unsafe_allow_html=True)
    all_rm = sorted(DF_REV["month"].unique())
    t_from = st.selectbox("起始月份", all_rm,
                           index=max(0, len(all_rm)-24), key="tf")
    st.divider()
    st.caption("資料：Inside Airbnb 2025-09\n© 2026 智慧旅宿 AI 平台")

# ── Filter ──────────────────────────────────────────────────────
mask = (DF_ALL["neighbourhood"].isin(sel_nb) & DF_ALL["room_type_zh"].isin(sel_rt) &
        DF_ALL["risk_level"].isin(sel_risk) & DF_ALL["price"].between(*sel_p) &
        (DF_ALL["number_of_reviews"] >= sel_min_rev))
df  = DF_ALL[mask].copy()
NB  = nb_agg(df)
CAL = DF_CAL[DF_CAL["month"] >= t_from]
REV = DF_REV[DF_REV["month"] >= t_from]

# ═══════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════
ha, hb = st.columns([3,1])
with ha:
    st.markdown(f"""
    <h1 style="font-size:1.45rem;font-weight:700;color:{P['ink']};margin:2px 0;
         letter-spacing:-.4px;">智慧旅宿「滯銷風險」預警平台</h1>
    <p style="font-size:.80rem;color:{P['muted']};margin:3px 0 0;">
      台北市 Inside Airbnb 真實資料 &ensp;｜&ensp;
      加權評分模型 × 邏輯斯回歸 × 隨機森林 &ensp;｜&ensp;
      LR&nbsp;Recall=<b style="color:{P['low']};">{LR['recall']:.3f}</b>
      &ensp;RF&nbsp;AUC=<b style="color:{P['primary']};">{RF['auc']:.3f}</b>
    </p>""", unsafe_allow_html=True)
with hb:
    st.markdown(f"""
    <div style="text-align:right;padding-top:4px;">
      <span style="font-size:.73rem;color:{P['muted']};">篩選中</span><br>
      <span style="font-size:1.7rem;font-weight:700;color:{P['primary']};
           ">{len(df):,}</span>
      <span style="font-size:.73rem;color:{P['muted']};"> 筆</span>
    </div>""", unsafe_allow_html=True)
st.markdown('<hr style="margin:10px 0 14px;">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════
T1,T2,T3,T4,T5,T6 = st.tabs([
    "概覽儀表板","風險地圖","趨勢分析","行政區比較","房源明細","🤖 AI 模型分析"
])

# ──────────────────────────────────────────────────────────────────
# TAB 1  概覽
# ──────────────────────────────────────────────────────────────────
with T1:
    total = len(df)
    h_n=(df["risk_level"]=="高風險").sum()
    m_n=(df["risk_level"]=="中風險").sum()
    l_n=(df["risk_level"]=="低風險").sum()
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("篩選房源",   f"{total:,}")
    k2.metric("🔴 高風險", f"{h_n:,}", f"{h_n/total*100:.1f}%" if total else "–",
              delta_color="inverse")
    k3.metric("🟡 中風險", f"{m_n:,}")
    k4.metric("🟢 低風險", f"{l_n:,}")
    k5.metric("平均風險分數", f"{df['risk_score'].mean():.3f}")
    k6.metric("平均入住率",  f"{df['occupancy_pct'].mean():.1f}%")
    st.divider()

    r1a,r1b,r1c = st.columns([1.1,1.5,1.8])
    with r1a:
        sec("風險等級佔比")
        mb("佔比分析 · Proportional Analysis")
        rc = df["risk_level"].value_counts().reindex(["高風險","中風險","低風險"]).reset_index()
        rc.columns=["風險等級","數量"]
        fig=px.pie(rc,values="數量",names="風險等級",color="風險等級",
                   color_discrete_map=RC,hole=0.58)
        fig.update_traces(textfont_size=11,marker_line_width=2,
                          marker_line_color=P["bg"],
                          pull=[0.05 if r=="高風險" else 0 for r in rc["風險等級"]])
        T(fig,h=252).update_layout(margin=dict(l=5,r=5,t=5,b=5),
                                    legend=dict(orientation="v",x=1,y=0.5))
        st.plotly_chart(fig,use_container_width=True)

    with r1b:
        sec("房型 × 風險等級")
        mb("列聯表分析 · Cross-tabulation")
        rt_risk=df.groupby(["room_type_zh","risk_level"]).size().reset_index(name="數量")
        fig=px.bar(rt_risk,x="room_type_zh",y="數量",color="risk_level",
                   color_discrete_map=RC,barmode="stack",
                   category_orders={"risk_level":["低風險","中風險","高風險"]},
                   labels={"room_type_zh":"","數量":"房源數","risk_level":""})
        fig.update_traces(marker_line_width=0)
        T(fig,h=252).update_layout(margin=dict(l=30,r=10,t=5,b=40))
        st.plotly_chart(fig,use_container_width=True)

    with r1c:
        sec("每晚價格 vs 滯銷風險分數")
        mb("皮爾森相關分析 + LOWESS 趨勢擬合 · Pearson Correlation + LOWESS")
        samp=df[df["price"]<df["price"].quantile(.96)].sample(min(900,len(df)),random_state=1)
        r_val=float(samp[["price","risk_score"]].corr().iloc[0,1])
        fig=px.scatter(samp,x="price",y="risk_score",color="risk_level",
                       color_discrete_map=RC,opacity=0.52,
                       hover_name="neighbourhood",
                       hover_data={"price":True,"risk_score":":.3f",
                                   "number_of_reviews":True,"risk_level":False},
                       trendline="lowess",trendline_color_override=P["primary"],
                       labels={"price":"每晚價格(TWD)","risk_score":"滯銷風險分數","risk_level":""})
        fig.update_traces(marker=dict(size=5,line=dict(width=0)),selector=dict(mode="markers"))
        T(fig,h=252).update_layout(margin=dict(l=44,r=10,t=5,b=40))
        fig.add_annotation(x=0.98,y=0.04,xref="paper",yref="paper",
                           text=f"r = {r_val:+.3f}",showarrow=False,
                           font=dict(size=11,color=P["primary"]),
                           bgcolor="rgba(255,255,255,.7)")
        st.plotly_chart(fig,use_container_width=True)

    r2a,r2b = st.columns(2)
    with r2a:
        sec("每晚價格頻率分佈")
        mb("頻率分佈直方圖 · Histogram (Freedman-Diaconis bins)")
        pdat=df[df["price"]<df["price"].quantile(.97)]
        fig=px.histogram(pdat,x="price",nbins=50,color="risk_level",
                         color_discrete_map=RC,barmode="overlay",opacity=0.68,
                         labels={"price":"每晚價格(TWD)","count":"房源數","risk_level":""})
        fig.update_traces(marker_line_width=0)
        T(fig,h=238).update_layout(margin=dict(l=44,r=10,t=5,b=36))
        st.plotly_chart(fig,use_container_width=True)

    with r2b:
        sec("入住率 × 風險等級分佈")
        mb("核密度估計 · Kernel Density Estimation (KDE) · Violin + Box")
        fig=px.violin(df,x="risk_level",y="occupancy_pct",color="risk_level",
                      color_discrete_map=RC,box=True,points=False,
                      category_orders={"risk_level":["低風險","中風險","高風險"]},
                      labels={"risk_level":"","occupancy_pct":"入住率 (%)"})
        T(fig,h=238,legend=False).update_layout(margin=dict(l=44,r=10,t=5,b=36))
        st.plotly_chart(fig,use_container_width=True)

    note("加權評分模型：滯銷風險分數 = <b>0.40</b>×可訂率 + <b>0.30</b>×評論稀疏度 "
         "+ <b>0.20</b>×近12月活躍度缺失 + <b>0.10</b>×相對價格偏高 "
         "（閾值：低&lt;0.35 ｜ 中0.35–0.60 ｜ 高≥0.60）")

# ──────────────────────────────────────────────────────────────────
# TAB 2  地圖
# ──────────────────────────────────────────────────────────────────
with T2:
    sec("台北市 Airbnb 滯銷風險熱點地圖")
    mb("地理空間分析 · Geospatial Scatter Map (Mapbox / CARTO Positron)")
    mc1,mc2,mc3=st.columns([1,1,1])
    color_by=mc1.selectbox("著色依據",["風險等級","房型","入住率(%)"],key="cb")
    size_by =mc2.radio("點大小",["一致","依風險分數"],horizontal=True,key="sb")
    n_show  =mc3.slider("顯示筆數",300,min(2000,len(df)),min(1200,len(df)),step=100,key="ns")
    geo=df.dropna(subset=["latitude","longitude"]).sample(min(n_show,len(df)),random_state=2)
    if color_by=="風險等級":
        c_kw=dict(color="risk_level",color_discrete_map=RC,
                  category_orders={"risk_level":["高風險","中風險","低風險"]})
    elif color_by=="房型":
        c_kw=dict(color="room_type_zh",color_discrete_map=RTC)
    else:
        c_kw=dict(color="occupancy_pct",
                  color_continuous_scale=["#FDECEA","#FDF5E4","#EAF5EE"],
                  range_color=[0,100])
    s_kw=dict(size="risk_score",size_max=14) if size_by=="依風險分數" else {}
    fig=px.scatter_mapbox(geo,lat="latitude",lon="longitude",hover_name="name",
        hover_data={"neighbourhood":True,"room_type_zh":True,"price":True,
                    "risk_score":":.3f","number_of_reviews":True,
                    "occupancy_pct":":.1f","latitude":False,"longitude":False},
        mapbox_style="carto-positron",zoom=11,opacity=0.82,height=540,
        center={"lat":25.047,"lon":121.517},
        labels={"risk_level":"風險","room_type_zh":"房型","occupancy_pct":"入住率%"},
        **c_kw,**s_kw)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=0,b=0),
                      legend=dict(bgcolor=P["surface"],bordercolor=P["border"],borderwidth=1))
    st.plotly_chart(fig,use_container_width=True)
    nb_mini=NB[["neighbourhood","房源數","高風險佔比","平均風險","平均入住率"]].copy()
    nb_mini.columns=["行政區","房源數","高風險佔比(%)","平均風險分數","平均入住率(%)"]
    html_table(nb_mini,fmt={"高風險佔比(%)":"{:.1f}","平均風險分數":"{:.3f}","平均入住率(%)":"{:.1f}"},
               cell_fn={"高風險佔比(%)": lambda v:
                   f"color:{P['high']};font-weight:700;" if isinstance(v,(int,float)) and v>=60
                   else (f"color:{P['medium']};font-weight:600;" if isinstance(v,(int,float)) and v>=40 else "")},
               height=230)

# ──────────────────────────────────────────────────────────────────
# TAB 3  趨勢
# ──────────────────────────────────────────────────────────────────
with T3:
    r1,r2=st.columns(2)
    with r1:
        sec("月度平均入住率 (%)")
        mb("時間序列分析 · Time Series Analysis · 面積圖 (Area Chart)")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=CAL["month"],y=CAL["avg_occupancy_pct"],
            mode="lines+markers",line=dict(color=P["low"],width=2.5),
            marker=dict(size=5,color=P["low"],line=dict(width=1.5,color=P["surface"])),
            fill="tozeroy",fillcolor="rgba(91,158,115,.10)",
            hovertemplate="%{x}<br>入住率：%{y:.1f}%<extra></extra>"))
        fig.add_hrect(y0=0,y1=35,fillcolor=P["high"],opacity=0.04,
                      annotation_text="低入住危險區",
                      annotation_font=dict(size=9,color=P["high"]),
                      annotation_position="top left")
        T(fig,h=280,legend=False).update_layout(
            yaxis=dict(range=[0,100],title="入住率 (%)"),xaxis_title="")
        st.plotly_chart(fig,use_container_width=True)

    with r2:
        sec("月度可訂率（閒置率）%")
        mb("時間序列分析 · Time Series Analysis · 面積圖 (Area Chart)")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=CAL["month"],y=CAL["avg_availability_pct"],
            mode="lines+markers",line=dict(color=P["high"],width=2.5),
            marker=dict(size=5,color=P["high"],line=dict(width=1.5,color=P["surface"])),
            fill="tozeroy",fillcolor="rgba(196,100,90,.08)",
            hovertemplate="%{x}<br>可訂率：%{y:.1f}%<extra></extra>"))
        T(fig,h=280,legend=False).update_layout(
            yaxis=dict(range=[0,100],title="可訂率 (%)"),xaxis_title="")
        st.plotly_chart(fig,use_container_width=True)

    r3,r4=st.columns(2)
    with r3:
        sec("月度評論數（市場活躍度代理指標）")
        mb("時序頻率統計 · Temporal Frequency Analysis · 月度聚合 (Monthly Aggregation)")
        rev_p=REV[REV["month"]<="2025-09"].copy()
        bar_c=[P["high"] if v<3000 else P["primary"] for v in rev_p["review_count"]]
        fig=go.Figure(go.Bar(x=rev_p["month"],y=rev_p["review_count"],
            marker=dict(color=bar_c,line=dict(width=0)),
            hovertemplate="%{x}<br>評論數：%{y:,}<extra></extra>"))
        T(fig,h=265,legend=False).update_layout(yaxis_title="評論數",xaxis_title="")
        st.plotly_chart(fig,use_container_width=True)

    with r4:
        sec("入住率 vs 可訂率 雙指標對比")
        mb("雙時序比較 · Dual-Metric Time Series Comparison")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=CAL["month"],y=CAL["avg_occupancy_pct"],
            name="入住率",mode="lines",line=dict(color=P["low"],width=2.5)))
        fig.add_trace(go.Scatter(x=CAL["month"],y=CAL["avg_availability_pct"],
            name="可訂率（滯銷）",mode="lines",
            line=dict(color=P["high"],width=2.5,dash="dot")))
        T(fig,h=265).update_layout(
            yaxis=dict(range=[0,100],title="%"),xaxis_title="",
            legend=dict(orientation="h",y=1.08,x=0))
        st.plotly_chart(fig,use_container_width=True)

    note("月度資料來源：<b>calendar.csv</b>（2025/09–2026/09 真實訂房日曆）· "
         "<b>reviews.csv</b>（評論數作為市場活躍度代理指標 Proxy Metric）")

# ──────────────────────────────────────────────────────────────────
# TAB 4  行政區比較
# ──────────────────────────────────────────────────────────────────
with T4:
    a1,a2=st.columns(2)
    with a1:
        sec("各行政區高風險比例排名")
        mb("排名比較分析 · Ranking Comparative Analysis · 水平長條圖")
        nb_s=NB.sort_values("高風險佔比",ascending=True)
        fig=go.Figure(go.Bar(x=nb_s["高風險佔比"],y=nb_s["neighbourhood"],
            orientation="h",
            marker=dict(color=nb_s["高風險佔比"],
                        colorscale=[[0,P["low"]],[0.4,P["medium"]],[1,P["high"]]],
                        cmin=0,cmax=85,line=dict(width=0)),
            text=nb_s["高風險佔比"].map("{:.1f}%".format),textposition="outside",
            hovertemplate="<b>%{y}</b><br>高風險：%{x:.1f}%<extra></extra>"))
        T(fig,h=360,legend=False).update_layout(
            xaxis=dict(range=[0,96],title="高風險比例 (%)"),
            yaxis_title="",margin=dict(l=64,r=55,t=5,b=36))
        st.plotly_chart(fig,use_container_width=True)

    with a2:
        sec("行政區多維散佈（中位價格 × 平均風險 × 房源數）")
        mb("多維散佈分析 · Multi-dimensional Scatter · 氣泡圖 (Bubble Chart)")
        fig=px.scatter(NB,x="中位價格",y="平均風險",size="房源數",color="高風險佔比",
            color_continuous_scale=[[0,P["low"]],[0.5,P["medium"]],[1,P["high"]]],
            text="neighbourhood",size_max=60,
            hover_data={"房源數":True,"平均入住率":True,"平均評論數":True},
            labels={"中位價格":"中位每晚價格(TWD)","平均風險":"平均滯銷風險分數","高風險佔比":"高風險佔比(%)"})
        fig.update_traces(textposition="top center",
                          textfont=dict(size=10,color=P["ink"]),
                          marker=dict(line=dict(width=1.5,color=P["surface"])))
        T(fig,h=360).update_layout(
            coloraxis_colorbar=dict(title="高風險%",len=0.65),
            margin=dict(l=44,r=10,t=5,b=36))
        st.plotly_chart(fig,use_container_width=True)

    sec("各行政區房型成分比例")
    mb("成分比例分析 · Compositional Analysis · 100% 正規化堆疊長條圖")
    nb_rt=df.groupby(["neighbourhood","room_type_zh"]).size().reset_index(name="n")
    nb_rt["pct"]=(nb_rt["n"]/nb_rt.groupby("neighbourhood")["n"].transform("sum")*100).round(1)
    fig=px.bar(nb_rt,x="neighbourhood",y="pct",color="room_type_zh",
               color_discrete_map=RTC,barmode="stack",
               text=nb_rt["pct"].map("{:.0f}%".format),
               labels={"neighbourhood":"","pct":"佔比 (%)","room_type_zh":"房型"},
               category_orders={"room_type_zh":["整棟出租","私人套房","共用套房","飯店客房"]})
    fig.update_traces(marker_line_width=0,textposition="inside",
                      textfont=dict(color="white",size=9))
    T(fig,h=265).update_layout(yaxis=dict(range=[0,103],title=""),
                                margin=dict(l=10,r=10,t=5,b=36))
    st.plotly_chart(fig,use_container_width=True)

    sec("行政區描述性統計彙整表")
    mb("描述性統計 · Descriptive Statistics (Mean, Median, Count)")
    tbl=NB[["neighbourhood","房源數","高風險數","高風險佔比","平均風險","中位價格","平均入住率","平均評論數"]].copy()
    tbl.columns=["行政區","房源數","高風險數","高風險佔比(%)","平均風險分數","中位價格(TWD)","平均入住率(%)","平均評論數"]
    html_table(tbl,
        fmt={"高風險佔比(%)":"{:.1f}","平均風險分數":"{:.3f}","平均入住率(%)":"{:.1f}","平均評論數":"{:.1f}"},
        cell_fn={"高風險佔比(%)": lambda v:
            f"color:{P['high']};font-weight:700;" if isinstance(v,(int,float)) and v>=60
            else (f"color:{P['medium']};font-weight:600;" if isinstance(v,(int,float)) and v>=40 else "")},
        height=310)

# ──────────────────────────────────────────────────────────────────
# TAB 5  房源明細
# ──────────────────────────────────────────────────────────────────
with T5:
    d1,d2=st.columns([2,1])
    with d1:
        sort_by=st.selectbox("排序欄位",
            ["risk_score","price","number_of_reviews","occupancy_pct","availability_365"],
            format_func=lambda x:{"risk_score":"滯銷風險分數","price":"每晚價格",
                "number_of_reviews":"評論數","occupancy_pct":"入住率",
                "availability_365":"年度可訂天數"}[x])
    with d2:
        asc=st.radio("排序",["由高→低","由低→高"],horizontal=True)=="由低→高"
    SMAP={"id":"ID","neighbourhood":"行政區","room_type_zh":"房型",
          "price":"每晚價格(TWD)","number_of_reviews":"評論數",
          "number_of_reviews_ltm":"近12月評論","occupancy_pct":"入住率(%)",
          "availability_365":"年度可訂天數","risk_score":"滯銷風險分數","risk_level":"風險等級"}
    avc=[c for c in SMAP if c in df.columns]
    df_s=(df.sort_values(sort_by,ascending=asc)[avc].head(300).reset_index(drop=True))
    df_s.index=df_s.index+1; df_s.columns=[SMAP[c] for c in avc]
    def _rc(v):
        if v=="高風險": return f"background:#FDECEA;color:{P['high']};font-weight:700;"
        if v=="中風險": return f"background:#FDF5E4;color:#A07A20;font-weight:700;"
        if v=="低風險": return f"background:#EAF5EE;color:#3D7A55;font-weight:700;"
        return ""
    def _sc(v):
        if isinstance(v,float) and 0<=v<=1:
            if v>=.60: return f"color:{P['high']};font-weight:700;"
            if v>=.35: return f"color:{P['medium']};font-weight:600;"
            return f"color:{P['low']};"
        return ""
    sec("篩選房源明細（最多顯示 300 筆）")
    mb("多條件篩選排序 · Multi-filter Sorting · 描述性統計 Descriptive Statistics")
    html_table(df_s,fmt={"滯銷風險分數":"{:.3f}","入住率(%)":"{:.1f}"},
               cell_fn={"風險等級":_rc,"滯銷風險分數":_sc},height=460)
    csv_out=df_s.to_csv(index=False,encoding="utf-8-sig").encode()
    st.download_button("⬇ 下載篩選結果 CSV",data=csv_out,
        file_name=f"risk_{datetime.now().strftime('%Y%m%d')}.csv",mime="text/csv")
    st.divider()
    e1,e2=st.columns(2)
    with e1:
        sec("高風險 TOP 10")
        mb("極值排序分析 · Extremum Ranking Analysis")
        t10=(df[df["risk_level"]=="高風險"].sort_values("risk_score",ascending=False).head(10))
        if not t10.empty:
            fig=go.Figure(go.Bar(x=t10["risk_score"],
                y=[f"{r.neighbourhood}·#{r.id}" for r in t10.itertuples()],
                orientation="h",marker=dict(color=P["high"],line=dict(width=0)),
                text=t10["risk_score"].map("{:.3f}".format),textposition="outside"))
            T(fig,h=300,legend=False).update_layout(
                xaxis=dict(range=[0,1.08],title="滯銷風險分數"),
                yaxis_title="",margin=dict(l=130,r=55,t=5,b=36))
            st.plotly_chart(fig,use_container_width=True)
    with e2:
        sec("低風險 TOP 10")
        mb("極值排序分析 · Extremum Ranking Analysis")
        t10l=(df[df["risk_level"]=="低風險"].sort_values("risk_score",ascending=True).head(10))
        if not t10l.empty:
            fig=go.Figure(go.Bar(x=t10l["risk_score"],
                y=[f"{r.neighbourhood}·#{r.id}" for r in t10l.itertuples()],
                orientation="h",marker=dict(color=P["low"],line=dict(width=0)),
                text=t10l["risk_score"].map("{:.3f}".format),textposition="outside"))
            T(fig,h=300,legend=False).update_layout(
                xaxis=dict(range=[0,.5],title="滯銷風險分數"),
                yaxis_title="",margin=dict(l=130,r=55,t=5,b=36))
            st.plotly_chart(fig,use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# TAB 6  AI 模型分析
# ──────────────────────────────────────────────────────────────────
with T6:
    # ── 模型總覽 KPI ──
    st.markdown(f"""
    <div style="background:linear-gradient(120deg,#EEF4FB,{P['surface']});
         border:1px solid {P['border']};border-radius:10px;
         padding:16px 22px;margin-bottom:14px;">
      <div style="font-size:.72rem;font-weight:700;letter-spacing:.1em;
           text-transform:uppercase;color:{P['muted']};margin-bottom:8px;">
        訓練資料：{MDL['n_train']:,} 筆 ｜ 測試資料：{MDL['n_test']:,} 筆 ｜
        高風險正類：{MDL['n_pos']:,}（{MDL['n_pos']/(MDL['n_pos']+MDL['n_neg'])*100:.1f}%）
      </div>
      <div style="display:flex;gap:40px;flex-wrap:wrap;">
        <div>
          <div style="font-size:.8rem;color:{P['muted']};margin-bottom:2px;">邏輯斯回歸 (LR)</div>
          <span style="color:{P['primary']};font-weight:700;">Recall={LR['recall']:.3f}</span>
          &ensp;<span style="color:{P['ink2']};">Precision={LR['precision']:.3f}</span>
          &ensp;<span style="color:{P['ink2']};">F1={LR['f1']:.3f}</span>
          &ensp;<span style="color:{P['accent']};">AUC={LR['auc']:.3f}</span>
        </div>
        <div>
          <div style="font-size:.8rem;color:{P['muted']};margin-bottom:2px;">隨機森林 (RF)</div>
          <span style="color:{P['primary']};font-weight:700;">Recall={RF['recall']:.3f}</span>
          &ensp;<span style="color:{P['ink2']};">Precision={RF['precision']:.3f}</span>
          &ensp;<span style="color:{P['ink2']};">F1={RF['f1']:.3f}</span>
          &ensp;<span style="color:{P['accent']};">AUC={RF['auc']:.3f}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    m1,m2=st.columns(2)

    # ── ROC 曲線 ──
    with m1:
        sec("ROC 曲線（接收者操作特徵曲線）")
        mb(f"ROC 曲線分析 · ROC Curve · LR AUC={LR['auc']:.3f} · RF AUC={RF['auc']:.3f}")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=LR["fpr"],y=LR["tpr"],mode="lines",
            name=f"邏輯斯回歸 (AUC={LR['auc']:.3f})",
            line=dict(color=P["primary"],width=2.5)))
        fig.add_trace(go.Scatter(x=RF["fpr"],y=RF["tpr"],mode="lines",
            name=f"隨機森林 (AUC={RF['auc']:.3f})",
            line=dict(color=P["low"],width=2.5,dash="dash")))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="隨機猜測",
            line=dict(color=P["muted"],width=1.5,dash="dot")))
        T(fig,h=340).update_layout(
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (Recall)",
            legend=dict(x=0.55,y=0.12))
        st.plotly_chart(fig,use_container_width=True)

    # ── PR 曲線 ──
    with m2:
        sec("Precision-Recall 曲線")
        mb(f"PR 曲線分析 · Precision-Recall Curve · LR AP={LR['ap']:.3f} · RF AP={RF['ap']:.3f}")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=LR["rec"],y=LR["prec"],mode="lines",
            name=f"邏輯斯回歸 (AP={LR['ap']:.3f})",
            line=dict(color=P["primary"],width=2.5)))
        fig.add_trace(go.Scatter(x=RF["rec"],y=RF["prec"],mode="lines",
            name=f"隨機森林 (AP={RF['ap']:.3f})",
            line=dict(color=P["low"],width=2.5,dash="dash")))
        T(fig,h=340).update_layout(
            xaxis_title="Recall (召回率)",
            yaxis_title="Precision (精確率)",
            legend=dict(x=0.02,y=0.12))
        st.plotly_chart(fig,use_container_width=True)

    m3,m4=st.columns(2)

    # ── RF 特徵重要性 ──
    with m3:
        sec("隨機森林特徵重要性（Gini 不純度）")
        mb("特徵重要性 · Feature Importance · Gini Impurity (Mean Decrease Impurity)")
        fi=pd.DataFrame({"特徵":MDL["feat_names"],
                         "重要性":list(MDL["rf_import"].values())})
        fi["特徵"]=fi["特徵"].map(lambda x: FEAT_ZH.get(x,x))
        fi=fi[~fi["特徵"].str.startswith("房型：") | (fi["重要性"]>0.002)]
        fi=fi.sort_values("重要性",ascending=True).tail(9)
        fig=go.Figure(go.Bar(x=fi["重要性"],y=fi["特徵"],orientation="h",
            marker=dict(color=fi["重要性"],
                        colorscale=[[0,P["low"]],[0.5,P["primary"]],[1,P["accent"]]],
                        line=dict(width=0)),
            text=fi["重要性"].map("{:.3f}".format),textposition="outside"))
        T(fig,h=310,legend=False).update_layout(
            xaxis=dict(range=[0,.38],title="特徵重要性（Gini）"),
            yaxis_title="",margin=dict(l=90,r=55,t=5,b=36))
        st.plotly_chart(fig,use_container_width=True)

    # ── LR 標準化係數 ──
    with m4:
        sec("邏輯斯回歸標準化係數")
        mb("邏輯斯回歸係數 · Logistic Regression Standardized Coefficients · L2 正則化")
        lc=pd.DataFrame({"特徵":MDL["feat_names"],
                         "係數":list(MDL["lr_coef"].values())})
        lc["特徵"]=lc["特徵"].map(lambda x: FEAT_ZH.get(x,x))
        lc=lc[~lc["特徵"].str.startswith("房型：") | (lc["係數"].abs()>0.05)]
        lc=lc.sort_values("係數",ascending=True).tail(9)
        bar_c=[P["high"] if v>0 else P["low"] for v in lc["係數"]]
        fig=go.Figure(go.Bar(x=lc["係數"],y=lc["特徵"],orientation="h",
            marker=dict(color=bar_c,line=dict(width=0)),
            text=lc["係數"].map("{:+.3f}".format),textposition="outside"))
        fig.add_vline(x=0,line_dash="dash",line_color=P["border2"],line_width=1.5)
        T(fig,h=310,legend=False).update_layout(
            xaxis_title="標準化係數（正→↑風險 · 負→↓風險）",
            yaxis_title="",margin=dict(l=90,r=65,t=5,b=36))
        st.plotly_chart(fig,use_container_width=True)

    # ── 混淆矩陣 ──
    m5,m6=st.columns(2)
    for col_,label_,met_,clr_ in [
        (m5,"邏輯斯回歸",LR,P["primary"]),
        (m6,"隨機森林",RF,P["low"])]:
        with col_:
            sec(f"{label_} 混淆矩陣")
            mb(f"混淆矩陣 · Confusion Matrix · Recall={met_['recall']:.3f} · Precision={met_['precision']:.3f}",
               warning=(met_["recall"]<0.80))
            cm=met_["cm"]
            z=[[cm[1,1],cm[1,0]],[cm[0,1],cm[0,0]]]
            x_lab=["預測：高風險","預測：非高風險"]
            y_lab=["實際：高風險","實際：非高風險"]
            fig=go.Figure(go.Heatmap(z=z,x=x_lab,y=y_lab,
                colorscale=[[0, '#FFFFFF'], [1, 'rgba(78, 127, 176, 0.53)']] , # 0.53 是 88(十六進位)轉為 0~1 的透明度
                text=[[str(v) for v in row] for row in z],
                texttemplate="%{text}",textfont=dict(size=18,color=P["ink"]),
                showscale=False))
            T(fig,h=230,legend=False).update_layout(margin=dict(l=10,r=10,t=5,b=36))
            st.plotly_chart(fig,use_container_width=True)

    # ── 模型對比表 ──
    st.divider()
    sec("模型效能比較表")
    mb("模型評估指標 · Model Evaluation Metrics · Accuracy / Recall / Precision / F1 / AUC-ROC / AP")
    cmp=pd.DataFrame({
        "指標":    ["Accuracy","Recall（召回率）","Precision（精確率）","F1 Score","AUC-ROC","AP（平均精確率）"],
        "邏輯斯回歸": [f"{LR['accuracy']:.3f}",f"{LR['recall']:.3f}",
                       f"{LR['precision']:.3f}",f"{LR['f1']:.3f}",
                       f"{LR['auc']:.3f}",f"{LR['ap']:.3f}"],
        "隨機森林":   [f"{RF['accuracy']:.3f}",f"{RF['recall']:.3f}",
                       f"{RF['precision']:.3f}",f"{RF['f1']:.3f}",
                       f"{RF['auc']:.3f}",f"{RF['ap']:.3f}"],
        "說明": ["整體正確率","高風險偵測率（越高越重要）","預警精準率",
                 "Recall×Precision 調和平均","ROC 曲線下面積","PR 曲線下面積"],
    })
    def _best(v):
        try:
            fv=float(v)
            if fv>=0.97: return f"color:{P['low']};font-weight:700;"
            if fv>=0.93: return f"color:{P['primary']};font-weight:600;"
        except: pass
        return ""
    html_table(cmp,cell_fn={"邏輯斯回歸":_best,"隨機森林":_best},height=220)

    note("""
    <b>為什麼 Recall 比 Precision 更重要？</b><br>
    在滯銷風險預警場景中，漏報（False Negative）的代價遠高於誤報（False Positive）：
    漏掉一間真正高風險的房源可能造成長期空置損失；誤判一間低風險房源最多只是徒增一次人工複查。
    因此模型訓練時使用 <b>class_weight='balanced'</b> 提升 Recall，並以 <b>Recall≥0.95</b> 作為核心評估標準。
    """)
