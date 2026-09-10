from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "supplied_data"
REGIMES = ["Risk-On", "Late-Cycle", "Transitional", "Risk-Off", "Post-Shock"]
PROB = {r: f"CalibratedProxy_{r}_Probability" for r in REGIMES}
RAW_PROB = {r: f"{r}_Probability" for r in REGIMES}

st.set_page_config(page_title="Regime Intelligence | Indian Equity Markets", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1600px}
[data-testid="stMetric"]{background:rgba(255,255,255,.025);border:1px solid rgba(128,128,128,.22);padding:12px 14px;border-radius:12px}
.hero{padding:24px 28px;border:1px solid rgba(128,128,128,.22);border-radius:18px;background:linear-gradient(120deg,rgba(25,45,75,.35),rgba(25,25,35,.12));margin-bottom:16px}
.hero h1{font-size:2.05rem;margin:0;letter-spacing:-.02em}.hero p{margin:6px 0 0;color:#a9b2bf}
.section{font-size:1.05rem;font-weight:700;margin:20px 0 10px}.small{font-size:.82rem;color:#9aa3af}
.status{display:inline-block;padding:5px 10px;border-radius:999px;border:1px solid rgba(128,128,128,.3);font-size:.78rem;margin-right:5px}
</style>
""", unsafe_allow_html=True)


def load_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs) if path.exists() else None
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return None


def num(value, default=np.nan):
    try:
        x = pd.to_numeric(value, errors="coerce")
        return default if pd.isna(x) else float(x)
    except Exception:
        return default


def fmt_int(value):
    x = num(value)
    return "—" if np.isnan(x) else f"{x:,.0f}"


def fmt_pct(value):
    x = num(value)
    return "—" if np.isnan(x) else f"{x:.1%}"


def fmt_num(value, digits=3):
    x = num(value)
    return "—" if np.isnan(x) else f"{x:.{digits}f}"


final = load_csv(OUT / "final_regime_output.csv", parse_dates=["Date"])
features = load_csv(DATA / "model_features.csv", parse_dates=["Date"])
master = load_csv(DATA / "master_daily_raw.csv", parse_dates=["Date"])
stats = load_csv(OUT / "regime_21d_forward_stats.csv")
meta = {}
meta_path = OUT / "model_metadata.json"
if meta_path.exists():
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        meta = {}

if final is None or features is None or master is None:
    st.error("Required model outputs are missing. Run `python run_model.py` from the project root first.")
    st.stop()

raw_cols = ["Date", "Nifty_Close", "India_VIX", "FII_Net_Cr", "DII_Net_Cr", "USDINR", "Gilt10Y", "Stocks Above 50 DMA (%)", "Stocks Above 200 DMA (%)"]
raw_available = [c for c in raw_cols if c in master.columns]
df = final.merge(features, on="Date", how="left").merge(master[raw_available], on="Date", how="left")
df = df.sort_values("Date").reset_index(drop=True)
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# Sidebar
st.sidebar.markdown("## Control Panel")
st.sidebar.caption("Interactive research interface")
start_default = max(df["Date"].min().date(), (df["Date"].max() - pd.Timedelta(days=730)).date())
start_date = st.sidebar.date_input("History start", value=start_default, min_value=df["Date"].min().date(), max_value=df["Date"].max().date())
end_date = st.sidebar.date_input("History end", value=df["Date"].max().date(), min_value=df["Date"].min().date(), max_value=df["Date"].max().date())
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()
show_raw = st.sidebar.checkbox("Show raw HMM probabilities", value=False)
show_grid = st.sidebar.checkbox("Show detailed data grid", value=False)

window = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)].copy()

st.markdown("""
<div class="hero"><h1>◈ Regime Intelligence Dashboard</h1>
<p>Bayesian Regime Detection Engine · Indian Equity Direction Forecasting · Expert research & decision-support interface</p>
<span class="status">5-State Regime Model</span><span class="status">Walk-Forward Inference</span><span class="status">Bayesian Transition Layer</span><span class="status">Synthetic Demo Data</span>
</div>
""", unsafe_allow_html=True)
st.warning("DATA INTEGRITY: The supplied archive is explicitly labelled synthetic demonstration data. Numerical outputs are for workflow demonstration only and are not real-market empirical evidence or investment advice.")

# Executive snapshot
st.markdown('<div class="section">Executive Snapshot</div>', unsafe_allow_html=True)
current_regime = str(latest.get("CalibratedProxy_Dominant_Regime", "—"))
confidence = num(latest.get("CalibratedProxy_Confidence"))
uncertainty = num(latest.get("Uncertainty"))
entropy = num(latest.get("CalibratedProxy_Entropy"))
prev_conf = num(prev.get("CalibratedProxy_Confidence"))
conf_delta = None if np.isnan(confidence) or np.isnan(prev_conf) else confidence - prev_conf

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Current Regime", current_regime)
c2.metric("Model Confidence", fmt_pct(confidence), None if conf_delta is None else f"{conf_delta:+.1%}")
c3.metric("Uncertainty", fmt_num(uncertainty))
c4.metric("Nifty Close", fmt_int(latest.get("Nifty_Close")), f"{num(latest.get('Nifty_Ret_1D')):+.2%}" if not np.isnan(num(latest.get('Nifty_Ret_1D'))) else None)
c5.metric("India VIX", fmt_num(latest.get("India_VIX"),2))

# Current probability cards
st.markdown('<div class="section">Current Regime Probability Distribution</div>', unsafe_allow_html=True)
cols = st.columns(5)
for col, r in zip(cols, REGIMES):
    col.metric(r, fmt_pct(latest.get(PROB[r])))

# Risk dashboard
risk_score = (num(latest.get("VIX_Z252"),0) + (-num(latest.get("Nifty_Ret_63D"),0)) + (-num(latest.get("Drawdown"),0))) / 3
r1,r2,r3,r4 = st.columns(4)
r1.metric("63D Momentum", fmt_pct(latest.get("Nifty_Ret_63D")))
r2.metric("Drawdown", fmt_pct(latest.get("Drawdown")))
r3.metric("Breadth", fmt_pct(latest.get("AD_Breadth")))
r4.metric("FII Net", f"₹{fmt_int(latest.get('FII_Net_Cr'))} Cr")

# Tabs
(tab1,tab2,tab3,tab4,tab5) = st.tabs(["Market Regime","Drivers & Macro","Validation","Backtest","Data & Lineage"])


def line_fig(data, x, ys, title, y_title=None):
    fig=go.Figure()
    for y in ys:
        if y in data.columns:
            fig.add_trace(go.Scatter(x=data[x],y=data[y],mode="lines",name=y.replace("_"," ")))
    fig.update_layout(title=title,height=410,margin=dict(l=20,r=20,t=55,b=20),hovermode="x unified",yaxis_title=y_title)
    return fig

with tab1:
    left,right=st.columns([1.8,1])
    with left:
        cols_prob=[PROB[r] for r in REGIMES] if not show_raw else [RAW_PROB[r] for r in REGIMES]
        fig=go.Figure()
        for r,c in zip(REGIMES,cols_prob):
            if c in window.columns:
                fig.add_trace(go.Scatter(x=window["Date"],y=window[c],mode="lines",name=r,stackgroup="one",groupnorm="fraction"))
        fig.update_layout(title="Time-Varying Regime Probability Surface",height=480,margin=dict(l=20,r=20,t=55,b=20),hovermode="x unified",yaxis=dict(tickformat=".0%",range=[0,1]))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        vals=[num(latest.get(PROB[r]),0) for r in REGIMES]
        fig=go.Figure(go.Pie(labels=REGIMES,values=vals,hole=.60,textinfo="label+percent"))
        fig.update_layout(title="Latest Posterior Mix",height=480,margin=dict(l=10,r=10,t=55,b=10),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="section">Regime Timeline</div>',unsafe_allow_html=True)
    mapping={r:i for i,r in enumerate(REGIMES)}
    timeline=window.copy(); timeline["Regime_Code"]=timeline["CalibratedProxy_Dominant_Regime"].map(mapping)
    fig=go.Figure(go.Heatmap(z=[timeline["Regime_Code"].values],x=timeline["Date"],y=["Dominant regime"],colorscale="Viridis",showscale=False,zmin=0,zmax=4,hovertemplate="%{x|%d %b %Y}<br>Regime=%{z}<extra></extra>"))
    fig.update_layout(height=145,margin=dict(l=20,r=20,t=15,b=20))
    st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="section">Nifty 50 Market Context</div>',unsafe_allow_html=True)
    st.plotly_chart(line_fig(window,"Date",["Nifty_Close"],"Nifty 50 Index Level","Index level"),use_container_width=True)
    a,b=st.columns(2)
    with a: st.plotly_chart(line_fig(window,"Date",["Nifty_Ret_21D","Nifty_Ret_63D"],"Momentum Profile","Return"),use_container_width=True)
    with b: st.plotly_chart(line_fig(window,"Date",["Nifty_Vol_21D","Nifty_Vol_63D"],"Realised Volatility","Volatility"),use_container_width=True)

with tab2:
    a,b=st.columns(2)
    with a:
        st.plotly_chart(line_fig(window,"Date",["VIX_Z252","VIX_Change_21D"],"Volatility Stress Signals"),use_container_width=True)
        st.plotly_chart(line_fig(window,"Date",["AD_Breadth","Breadth_50D_Change","Breadth_200D_Change"],"Market Breadth"),use_container_width=True)
    with b:
        st.plotly_chart(line_fig(window,"Date",["USDINR_Ret_21D","Gilt_Change_21D","US10Y_Change_21D"],"Rates & Currency Pressure"),use_container_width=True)
        st.plotly_chart(line_fig(window,"Date",["Brent_Ret_21D","DXY_Ret_21D","Gold_Ret_21D"],"Global Cross-Asset Signals"),use_container_width=True)

    st.markdown('<div class="section">Institutional Flows</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric("FII net flow",f"₹{fmt_int(latest.get('FII_Net_Cr'))} Cr")
    b.metric("DII net flow",f"₹{fmt_int(latest.get('DII_Net_Cr'))} Cr")
    c.metric("USD / INR",fmt_num(latest.get("USDINR"),2))
    fig=go.Figure()
    for col,name in [("FII_Net_Cr","FII Net"),("DII_Net_Cr","DII Net")]:
        if col in window.columns: fig.add_trace(go.Scatter(x=window["Date"],y=window[col],name=name,mode="lines"))
    fig.update_layout(title="Institutional Flow Regime Context",height=400,hovermode="x unified",margin=dict(l=20,r=20,t=55,b=20))
    st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="section">Latest Driver Panel</div>',unsafe_allow_html=True)
    driver_map={
        "Trend / MA50-MA200":latest.get("MA50_to_MA200"),"Drawdown":latest.get("Drawdown"),"Breadth":latest.get("AD_Breadth"),
        "FII flow z-score":latest.get("FII_Net_Cr_Z63"),"DII flow z-score":latest.get("DII_Net_Cr_Z63"),"VIX z-score":latest.get("VIX_Z252"),
        "INR 21D":latest.get("USDINR_Ret_21D"),"Gilt 21D":latest.get("Gilt_Change_21D"),"Brent 21D":latest.get("Brent_Ret_21D"),
        "DXY 21D":latest.get("DXY_Ret_21D"),"Gold 21D":latest.get("Gold_Ret_21D"),"GDP growth":latest.get("GDP_Growth")}
    driver_df=pd.DataFrame({"Signal":list(driver_map.keys()),"Latest value":[num(v) for v in driver_map.values()]})
    st.dataframe(driver_df.style.format({"Latest value":"{:.4f}"}),use_container_width=True,hide_index=True)

with tab3:
    st.markdown('<div class="section">Model Diagnostics</div>',unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4)
    nobs=meta.get("n_observations")
    nfeat=meta.get("n_features")
    nstates=meta.get("n_states")
    m1.metric("Observations",fmt_int(nobs)); m2.metric("Features",fmt_int(nfeat)); m3.metric("States",fmt_int(nstates))
    bic=meta.get("bic",{}) or {}
    m4.metric("Best BIC (tested)",f"{min([num(v) for v in bic.values()]):,.0f}" if bic else "—")
    if bic:
        bic_df=pd.DataFrame({"States":[int(k) for k in bic.keys()],"BIC":[num(v) for v in bic.values()]})
        fig=px.bar(bic_df,x="States",y="BIC",title="BIC Comparison Across Tested State Counts")
        fig.update_layout(height=350,margin=dict(l=20,r=20,t=55,b=20))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="section">Uncertainty & Probability Concentration</div>',unsafe_allow_html=True)
    fig=go.Figure()
    for c,n in [("CalibratedProxy_Confidence","Confidence"),("CalibratedProxy_Entropy","Entropy")]:
        if c in window.columns: fig.add_trace(go.Scatter(x=window["Date"],y=window[c],name=n,mode="lines"))
    fig.update_layout(title="Confidence / Entropy Diagnostics",height=390,hovermode="x unified",margin=dict(l=20,r=20,t=55,b=20))
    st.plotly_chart(fig,use_container_width=True)

    if stats is not None:
        st.markdown('<div class="section">21-Day Forward-Return Diagnostics</div>',unsafe_allow_html=True)
        st.dataframe(stats,use_container_width=True,hide_index=True)
        st.caption("Calculated on the supplied synthetic demonstration dataset; not empirical evidence about real Indian markets.")
    st.info("Validation note: the current probability post-processing is a demonstration layer, not a formally validated conformal guarantee. The full brief requires rolling calibration, ECE, reliability diagrams and MCMC diagnostics.")

with tab4:
    bt_path=OUT/"regime_overlay_backtest.csv"
    if not bt_path.exists():
        st.info("Backtest file not found. Run `python run_model.py` to generate the model outputs, then rerun the dashboard.")
    else:
        bt=load_csv(bt_path,parse_dates=["Date"])
        if bt is not None:
            bt=bt[(bt["Date"].dt.date>=start_date)&(bt["Date"].dt.date<=end_date)]
            if len(bt):
                q1,q2,q3,q4=st.columns(4)
                q1.metric("Strategy cumulative",fmt_pct(bt["cum_strategy"].iloc[-1]-1))
                q2.metric("Benchmark cumulative",fmt_pct(bt["cum_benchmark"].iloc[-1]-1))
                q3.metric("Average equity weight",fmt_pct(bt["equity_weight"].mean()))
                q4.metric("Observations",fmt_int(len(bt)))
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=bt["Date"],y=bt["cum_strategy"],name="Regime overlay",mode="lines"))
                fig.add_trace(go.Scatter(x=bt["Date"],y=bt["cum_benchmark"],name="Benchmark",mode="lines"))
                fig.update_layout(title="Illustrative Regime-Overlay Backtest",height=430,hovermode="x unified",margin=dict(l=20,r=20,t=55,b=20))
                st.plotly_chart(fig,use_container_width=True)
                fig2=go.Figure(go.Scatter(x=bt["Date"],y=bt["equity_weight"],mode="lines",name="Equity weight"))
                fig2.update_layout(title="Illustrative Equity Allocation Weight",height=330,yaxis=dict(tickformat=".0%"),margin=dict(l=20,r=20,t=55,b=20))
                st.plotly_chart(fig2,use_container_width=True)
                st.info("Backtest overlay is illustrative research logic. It is not investment advice and should be regenerated with verified real data before empirical claims.")

with tab5:
    st.markdown('<div class="section">Data Coverage</div>',unsafe_allow_html=True)
    d1,d2,d3,d4=st.columns(4)
    d1.metric("Daily rows",fmt_int(len(master))); d2.metric("Model rows",fmt_int(len(final))); d3.metric("Start",str(df["Date"].min().date())); d4.metric("Latest",str(df["Date"].max().date()))
    st.markdown("**Source groups available in the supplied archive**")
    st.write("Nifty 50 · Nifty Midcap 100 · Nifty Smallcap 100 · India VIX · 10Y Gilt · FII/DII · SIP · CPI · IIP · WPI · PMI · GDP · USD/INR · US10Y · DXY · Gold · Brent · breadth")
    st.markdown('<div class="section">Model Lineage</div>',unsafe_allow_html=True)
    lineage=pd.DataFrame([
        ["Raw data","data/raw/supplied_data","Supplied archive"],["Processed master","data/processed/master_daily_raw.csv","Merged daily dataset"],
        ["Features","data/processed/model_features.csv","Engineered feature matrix"],["Model","models/regime_model.joblib","Gaussian HMM + transition layer"],
        ["Predictions","outputs/final_regime_output.csv","Regime probabilities and diagnostics"],["Backtest","outputs/regime_overlay_backtest.csv","Illustrative allocation overlay"],["Dashboard","dashboard.py","Interactive Streamlit interface"]],
        columns=["Layer","Location","Purpose"])
    st.dataframe(lineage,use_container_width=True,hide_index=True)
    st.markdown('<div class="section">Latest Model Record</div>',unsafe_allow_html=True)
    display_cols=[c for c in ["Date","CalibratedProxy_Dominant_Regime","CalibratedProxy_Confidence","Uncertainty","Nifty_Close","India_VIX","FII_Net_Cr","DII_Net_Cr"] if c in df.columns]
    st.dataframe(df.tail(50 if show_grid else 10)[display_cols],use_container_width=True,hide_index=True)

st.markdown("---")
st.caption("Research prototype · Source-aligned to the supplied Bayesian Regime Detection project brief · Verify data provenance before empirical or investment use.")
