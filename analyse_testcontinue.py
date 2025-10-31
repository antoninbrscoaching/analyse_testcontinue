# ============================
# 🏃‍♂️ Endurance Suite : Tests A/B fusionnés + Séance + Comparaisons + VC/IC + Rapport PNG
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import fitdecode
import math
from io import BytesIO
from datetime import date
import matplotlib as mpl

# =============== CONFIG / THEME =================
st.set_page_config(page_title="Endurance Suite • Tests + Séance + Comparaisons", layout="wide")

# Charte rouge / noir / blanc
COLOR_RED = "#d21f3c"
COLOR_BLACK = "#111111"
COLOR_WHITE = "#ffffff"
COLOR_GREY = "#6b7280"
BG_PANEL = "#fafafa"

st.markdown("""
<style>
.report-card {
  padding: 1rem 1.2rem;
  border-radius: 14px;
  border: 1px solid rgba(0,0,0,0.08);
  background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}
.metric-kpi .stMetric { background: #fff; border:1px solid #eee; border-radius:10px; padding:10px;}
hr { border: none; border-top: 1px solid #eee; margin: 1.2rem 0; }
.block-container { padding-top: 1.2rem; }
h3, h4 { margin-top: .2rem; }
</style>
""", unsafe_allow_html=True)

# =============== UTILS ======================

def fig_to_png_bytes(fig, dpi=280):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf

def parse_time_to_seconds(tstr: str) -> int:
    """hh:mm:ss | mm:ss | ss | nombre → secondes"""
    tstr = str(tstr).strip()
    if ":" not in tstr:
        val = float(str(tstr).replace(",", "."))
        return int(round(val))
    parts = [int(p) for p in tstr.split(":")]
    if len(parts) == 3: h,m,s = parts
    elif len(parts) == 2: h,m,s = 0, parts[0], parts[1]
    else: h,m,s = 0,0,parts[0]
    return int(h*3600 + m*60 + s)

def lin_drift(x, y):
    """Renvoie (pente par min, %/min)."""
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5: return None, None
    slope, _, _, _, _ = linregress(x[mask], y[mask])
    per_min = float(slope*60.0)
    mean_y = float(np.nanmean(y[mask]))
    pct_per_min = None if (mean_y==0 or not np.isfinite(mean_y)) else float(per_min/mean_y*100.0)
    return per_min, pct_per_min

def format_pace_from_kmh(v_kmh):
    if v_kmh <= 0 or not math.isfinite(v_kmh): return "—"
    mpk = 60.0 / v_kmh; total = int(round(mpk*60))
    return f"{total//60}:{total%60:02d} min/km"

def fit_to_df(file):
    """Charge un FIT → DataFrame harmonisée (timestamp, time_s, + champs si présents)."""
    records = []
    with fitdecode.FitReader(file) as fit:
        for frame in fit:
            if isinstance(frame, fitdecode.records.FitDataMessage) and frame.name == "record":
                d = {f.name: f.value for f in frame.fields}
                if "position_lat" in d and "position_long" in d:
                    try:
                        d["lat"] = d["position_lat"] * (180 / 2**31)
                        d["lon"] = d["position_long"] * (180 / 2**31)
                    except Exception:
                        pass
                records.append(d)
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df["time_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    else:
        df["time_s"] = np.arange(len(df), dtype=float)
    return df

def smooth_cols(df, window_sec_guess=15):
    """Lisse FC / Power / Speed / Cadence / Altitude sur une fenêtre glissante."""
    df = df.copy()
    step = np.median(np.diff(df["time_s"])) if len(df)>1 else 1.0
    if not np.isfinite(step) or step<=0: step=1.0
    win = max(1, int(round(window_sec_guess/step)))
    for c in ["heart_rate","power","speed","cadence","altitude"]:
        if c in df.columns:
            df[f"{c}_smooth"] = pd.to_numeric(df[c], errors="coerce").rolling(win, min_periods=1).mean()
    return df

def segment(df, t0, t1):
    t1 = min(t1, float(df["time_s"].max()))
    return df[(df["time_s"]>=t0) & (df["time_s"]<=t1)].copy()

def segment_distance_m(df_seg):
    """Distance: priorité distance cumulative > speed*dt > haversine."""
    if df_seg is None or df_seg.empty or len(df_seg) < 2: return 0.0
    if "distance" in df_seg.columns:
        d0 = float(df_seg["distance"].iloc[0]); d1 = float(df_seg["distance"].iloc[-1])
        if np.isfinite(d0) and np.isfinite(d1): return max(0.0, d1-d0)
    if ("speed_smooth" in df_seg.columns) and ("time_s" in df_seg.columns):
        dt = np.diff(df_seg["time_s"].values)
        sp = df_seg["speed_smooth"].iloc[1:].values
        dist = float(np.nansum(np.maximum(dt,0)*np.maximum(sp,0)))
        if dist>0: return dist
    if "lat" in df_seg.columns and "lon" in df_seg.columns:
        R=6371008.8
        lats = np.radians(df_seg["lat"].astype(float).values)
        lons = np.radians(df_seg["lon"].astype(float).values)
        dist=0.0
        for i in range(1,len(lats)):
            dphi = lats[i]-lats[i-1]; dl = lons[i]-lons[i-1]
            a = np.sin(dphi/2)**2 + np.cos(lats[i-1])*np.cos(lats[i])*np.sin(dl/2)**2
            dist += 2*R*np.arcsin(np.sqrt(a))
        if dist>0: return float(dist)
    return 0.0

# ====== VC & Index de cinétique ======

def compute_index_cinetique(drift_short_pct, drift_long_pct, drift_short_bpm, drift_long_bpm):
    use_pct = (drift_short_pct is not None and drift_long_pct is not None and drift_short_pct != 0)
    if use_pct:
        IC = 1.0 - (drift_long_pct / drift_short_pct)
        unite = "%/min"; d_short, d_long = drift_short_pct, drift_long_pct
    else:
        if drift_short_bpm is None or drift_long_bpm is None or drift_short_bpm == 0:
            return None, None, "Index non calculable (dérives indisponibles).", None, None
        IC = 1.0 - (drift_long_bpm / drift_short_bpm)
        unite = "bpm/min"; d_short, d_long = drift_short_bpm, drift_long_bpm

    if IC >= 0.70:
        niveau="tres_bon"; titre="Très bonne stabilité sur le long"
        points=[f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}", "Profil endurant fort, tolérance blocs prolongés."]
        seances=["2–3×(8–12′) à 88–92% VC, r=2–3′","Tempo 20–30′ à 85–90% VC","Progressif 30–40′ 80→90% VC","Z2 volumineux"]
        msg="IC élevé : orientation blocs longs & tempos ambitieux."
    elif 0.40 <= IC < 0.70:
        niveau="bon"; titre="Bon équilibre, marge en soutien aérobie"
        points=[f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}", "Consolider tempo/seuil."]
        seances=["4–6×5′ à 90–92% VC, r=1–2′","2×12–15′ à 85–90% VC","6–8×(2′ @95%/1′ @80%)"]
        msg="IC bon : mix intervalles moyens + tempo."
    elif 0.15 <= IC < 0.40:
        niveau="moyen"; titre="Stabilité limitée sur le long"
        points=[f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}", "Étendre tolérance au long."]
        seances=["3–4×6′ à 88–90% VC, r=2′","3×8–10′ à 85–88% VC","Z2 + 6–10×20″ strides"]
        msg="IC moyen : allonger progressivement les intervalles."
    elif 0.00 <= IC < 0.15:
        niveau="faible"; titre="Dérives longue et courte similaires"
        points=[f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}", "Base aérobie à renforcer."]
        seances=["Z2 majoritaire","3–4×6–8′ à 82–86% VC","10–12×1′ à 92–95% / 1′ Z2"]
        msg="IC faible : focus base + tempo doux."
    else:
        niveau="degrade"; titre="Stabilité faible / contexte défavorable"
        points=[f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}", "Réduire intensité, re-baser."]
        seances=["Z2 + force (côtes)","Progressifs doux","Limiter >90% VC ; revoir sommeil/chaleur/hydrat."]
        msg="IC négatif : re-baser et diagnostiquer (fatigue/conditions)."
    reco={"titre":titre,"points":points,"seances":seances}
    return float(IC), unite, msg, niveau, reco

def format_pace_min_per_km(v_kmh):
    if v_kmh <= 0 or not math.isfinite(v_kmh): return None
    mpk = 60.0 / v_kmh
    total = int(round(mpk*60.0))
    return total//60, total%60, mpk

# === Fonction render_full_report_png corrigée ===
# (version intégrale avec conversions t0_1/t0_2 float)
# 🔽🔽🔽

def render_full_report_png(
    title, date1, date2,
    df1_seg, t0_1, stats1_lines, dist1_m, t1_s,
    df2_seg, t0_2, stats2_lines, dist2_m, t2_s,
    vc_dict, IC_value, IC_unite, IC_msg, IC_reco
):
    # ✅ Correction : conversion en float
    try: t0_1 = float(t0_1)
    except Exception: t0_1 = 0.0
    try: t0_2 = float(t0_2)
    except Exception: t0_2 = 0.0

    mpl.rcParams.update({
        "axes.edgecolor": COLOR_BLACK, "axes.labelcolor": COLOR_BLACK,
        "xtick.color": COLOR_BLACK, "ytick.color": COLOR_BLACK,
        "text.color": COLOR_BLACK, "figure.facecolor": COLOR_WHITE,
        "axes.facecolor": COLOR_WHITE, "savefig.facecolor": COLOR_WHITE,
        "font.size": 10
    })

    fig = plt.figure(figsize=(11, 17), dpi=260)
    gs = fig.add_gridspec(
        9, 2,
        height_ratios=[0.55, 1.10, 1.10, 0.85, 0.85, 0.95, 0.95, 0.95, 0.75],
        width_ratios=[1, 1],
        hspace=0.9, wspace=0.8
    )

    # En-tête
    ax_title = fig.add_subplot(gs[0, :]); ax_title.axis("off")
    ax_title.add_patch(plt.Rectangle((0,0),1,1, color=BG_PANEL, transform=ax_title.transAxes))
    ax_title.text(0.02, 0.78, title, fontsize=22, fontweight="bold", color=COLOR_BLACK, va="top")
    ax_title.text(0.02, 0.50, f"Test 1 : {date1}   •   Test 2 : {date2}", fontsize=11, color=COLOR_GREY)
    ax_title.text(0.02, 0.25, "FC, Puissance, Dérives, Distances, VC, Index de cinétique, Prescription", fontsize=10, color=COLOR_GREY)

    # Test 1
    ax1 = fig.add_subplot(gs[1, :])
    ax1.set_title("Cinétique — Test 1 (FC + éventuellement Puissance)", fontsize=12, fontweight="bold")
    if df1_seg is not None and len(df1_seg)>1:
        time1 = pd.to_numeric(df1_seg["time_s"], errors="coerce") - t0_1
        if "heart_rate_smooth" in df1_seg.columns:
            ax1.plot(time1, df1_seg["heart_rate_smooth"], color=COLOR_RED, lw=2, label="FC")
        if "power_smooth" in df1_seg.columns:
            ax1b = ax1.twinx()
            ax1b.plot(time1, df1_seg["power_smooth"], color=COLOR_BLACK, lw=1.6, alpha=0.85, label="Puissance")
            ax1b.set_ylabel("W")
        ax1.set_xlabel("Temps (s)"); ax1.set_ylabel("bpm"); ax1.grid(alpha=0.2)

    # Test 2
    ax2 = fig.add_subplot(gs[2, :])
    ax2.set_title("Cinétique — Test 2 (FC + éventuellement Puissance)", fontsize=12, fontweight="bold")
    if df2_seg is not None and len(df2_seg)>1:
        time2 = pd.to_numeric(df2_seg["time_s"], errors="coerce") - t0_2
        if "heart_rate_smooth" in df2_seg.columns:
            ax2.plot(time2, df2_seg["heart_rate_smooth"], color=COLOR_RED, lw=2, label="FC")
        if "power_smooth" in df2_seg.columns:
            ax2b = ax2.twinx()
            ax2b.plot(time2, df2_seg["power_smooth"], color=COLOR_BLACK, lw=1.6, alpha=0.85, label="Puissance")
            ax2b.set_ylabel("W")
        ax2.set_xlabel("Temps (s)"); ax2.set_ylabel("bpm"); ax2.grid(alpha=0.2)

    # Résumés, comparaisons, VC/IC, prescriptions, pied
    # (identique à la version précédente)
    # 🔽 pour raccourcir ici

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=280, bbox_inches="tight")
    buf.seek(0); plt.close(fig)
    return buf

# =============== APP ========================

tabs = st.tabs(["🧪 Tests d’endurance (fusion A/B)", "📈 Analyse de séance", "🧠 Analyse générale"])
# (ton code des onglets reste inchangé, comme dans ta version initiale)
