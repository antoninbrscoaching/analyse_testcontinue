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
    # pas moyen approximatif
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

# ====== Rapport PNG soigné (aucun chevauchement) ======

def render_full_report_png(
    title, date1, date2,
    df1_seg, t0_1, stats1_lines, dist1_m, t1_s,
    df2_seg, t0_2, stats2_lines, dist2_m, t2_s,
    vc_dict, IC_value, IC_unite, IC_msg, IC_reco
):
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

    # Test 1 courbe FC (+power si présent)
    ax1 = fig.add_subplot(gs[1, :])
    ax1.set_title("Cinétique — Test 1 (FC + éventuellement Puissance)", fontsize=12, fontweight="bold")
    if df1_seg is not None and len(df1_seg)>1:
        if "heart_rate_smooth" in df1_seg.columns:
            ax1.plot(df1_seg["time_s"]-t0_1, df1_seg["heart_rate_smooth"], color=COLOR_RED, lw=2, label="FC")
        if "power_smooth" in df1_seg.columns:
            ax1b = ax1.twinx()
            ax1b.plot(df1_seg["time_s"]-t0_1, df1_seg["power_smooth"], color=COLOR_BLACK, lw=1.6, alpha=0.85, label="Puissance")
            ax1b.set_ylabel("W")
        ax1.set_xlabel("Temps (s)"); ax1.set_ylabel("bpm"); ax1.grid(alpha=0.2)

    # Test 2 courbe
    ax2 = fig.add_subplot(gs[2, :])
    ax2.set_title("Cinétique — Test 2 (FC + éventuellement Puissance)", fontsize=12, fontweight="bold")
    if df2_seg is not None and len(df2_seg)>1:
        if "heart_rate_smooth" in df2_seg.columns:
            ax2.plot(df2_seg["time_s"]-t0_2, df2_seg["heart_rate_smooth"], color=COLOR_RED, lw=2, label="FC")
        if "power_smooth" in df2_seg.columns:
            ax2b = ax2.twinx()
            ax2b.plot(df2_seg["time_s"]-t0_2, df2_seg["power_smooth"], color=COLOR_BLACK, lw=1.6, alpha=0.85, label="Puissance")
            ax2b.set_ylabel("W")
        ax2.set_xlabel("Temps (s)"); ax2.set_ylabel("bpm"); ax2.grid(alpha=0.2)

    # Cartes stats Test 1 & 2
    ax4 = fig.add_subplot(gs[3, 0]); ax5 = fig.add_subplot(gs[3, 1])
    for ax in (ax4, ax5):
        ax.axis("off"); ax.add_patch(plt.Rectangle((0,0),1,1, color=BG_PANEL, transform=ax.transAxes))
    ax4.text(0.04, 0.86, "Résumé Test 1", fontsize=12, fontweight="bold")
    y=0.74
    for line in (stats1_lines or []):
        ax4.text(0.06, y, line, fontsize=10, color=COLOR_BLACK); y-=0.08
    ax5.text(0.04, 0.86, "Résumé Test 2", fontsize=12, fontweight="bold")
    y=0.74
    for line in (stats2_lines or []):
        ax5.text(0.06, y, line, fontsize=10, color=COLOR_BLACK); y-=0.08

    # Comparatif FC superposé
    ax6 = fig.add_subplot(gs[4, :])
    ax6.set_title("Comparatif FC (segments centrés t=0)", fontsize=12, fontweight="bold")
    if (df1_seg is not None) and len(df1_seg)>1:
        if "heart_rate_smooth" in df1_seg.columns:
            ax6.plot(df1_seg["time_s"]-df1_seg["time_s"].iloc[0], df1_seg["heart_rate_smooth"], color=COLOR_RED, lw=2, label="Test 1")
    if (df2_seg is not None) and len(df2_seg)>1:
        if "heart_rate_smooth" in df2_seg.columns:
            ax6.plot(df2_seg["time_s"]-df2_seg["time_s"].iloc[0], df2_seg["heart_rate_smooth"], color=COLOR_BLACK, lw=2, label="Test 2")
    ax6.legend(frameon=False); ax6.set_xlabel("Temps (s)"); ax6.set_ylabel("bpm"); ax6.grid(alpha=0.2)

    # VC + IC cartes
    ax7L = fig.add_subplot(gs[5, 0]); ax7R = fig.add_subplot(gs[5, 1])
    for ax in (ax7L, ax7R):
        ax.axis("off"); ax.add_patch(plt.Rectangle((0,0),1,1, color=BG_PANEL, transform=ax.transAxes))
    ax7L.text(0.04, 0.86, "Vitesse Critique", fontsize=12, fontweight="bold")
    if vc_dict:
        lines_vc = [f"CS : {vc_dict['CS']:.2f} m/s", f"VC : {vc_dict['V_kmh']:.2f} km/h",
                    f"Allure VC : {vc_dict['pace_str']}", f"D′ : {vc_dict['D_prime']:.0f} m"]
        y=0.74
        for l in lines_vc: ax7L.text(0.06, y, l, fontsize=10, color=COLOR_BLACK); y-=0.08
    else:
        ax7L.text(0.06, 0.74, "— Non calculable —", fontsize=10, color=COLOR_GREY)

    ax7R.text(0.04, 0.86, "Index de cinétique (IC)", fontsize=12, fontweight="bold")
    if IC_value is not None:
        ax7R.text(0.06, 0.74, f"IC : {IC_value:.3f}", fontsize=10)
        ax7R.text(0.06, 0.66, f"Unité dérives : {IC_unite}", fontsize=10)
        ax7R.text(0.06, 0.58, f"{IC_msg}", fontsize=10)
    else:
        ax7R.text(0.06, 0.74, "— Non calculable —", fontsize=10, color=COLOR_GREY)

    # Bande prescription
    ax8 = fig.add_subplot(gs[6, :]); ax8.axis("off")
    ax8.add_patch(plt.Rectangle((0,0),1,1, color=BG_PANEL, transform=ax8.transAxes))
    ax8.text(0.03, 0.86, "Prescription (4–8 semaines)", fontsize=12, fontweight="bold")
    if IC_reco:
        y=0.74; ax8.text(0.04, y, f"• {IC_reco['titre']}", fontsize=10); y-=0.08
        for p in IC_reco["points"]:
            ax8.text(0.05, y, f"– {p}", fontsize=10); y-=0.06
        y-=0.04; ax8.text(0.04, y, "Séances types :", fontsize=10); y-=0.08
        for s in IC_reco["seances"]:
            ax8.text(0.05, y, f"• {s}", fontsize=10); y-=0.06
    else:
        ax8.text(0.04, 0.74, "— Aucune recommandation —", fontsize=10, color=COLOR_GREY)

    # Bandes infos bas
    ax9L = fig.add_subplot(gs[7, 0]); ax9R = fig.add_subplot(gs[7, 1])
    for ax in (ax9L, ax9R):
        ax.axis("off"); ax.add_patch(plt.Rectangle((0,0),1,1, color=BG_PANEL, transform=ax.transAxes))
    if (dist1_m is not None) and (t1_s is not None) and t1_s>0:
        v1 = (dist1_m/t1_s)*3.6
        ax9L.text(0.05,0.75,f"Test 1 : {dist1_m:.1f} m • {v1:.2f} km/h • {format_pace_from_kmh(v1)}", fontsize=10)
    if (dist2_m is not None) and (t2_s is not None) and t2_s>0:
        v2 = (dist2_m/t2_s)*3.6
        ax9R.text(0.05,0.75,f"Test 2 : {dist2_m:.1f} m • {v2:.2f} km/h • {format_pace_from_kmh(v2)}", fontsize=10)

    # Pied rouge
    ax_footer = fig.add_subplot(gs[8, :]); ax_footer.axis("off")
    ax_footer.add_patch(plt.Rectangle((0, 0.0), 1, 0.18, color=COLOR_RED, transform=ax_footer.transAxes))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=280, bbox_inches="tight")
    buf.seek(0); plt.close(fig)
    return buf

# =============== APP ========================

tabs = st.tabs(["🧪 Tests d’endurance (fusion A/B)", "📈 Analyse de séance", "🧠 Analyse générale"])

# ---------- Onglet 1 : TESTS A & B ----------
with tabs[0]:
    st.header("🧪 Tests d’endurance (A et B)")

    colA, colB = st.columns(2)

    # Vars à pousser vers la page 3
    state = st.session_state

    for test_label, col in zip(["A","B"], [colA, colB]):
        with col:
            st.subheader(f"Test {test_label}")
            file = st.file_uploader(f"Importer Test {test_label} (.fit)", type=["fit"], key=f"file_{test_label}")
            date_test = st.date_input(f"📅 Date Test {test_label}", value=date.today(), key=f"date_{test_label}")

            df = None; df_seg=None; dist_m=None; dur_s=None
            drift_bpm = drift_pct = None

            if file:
                try:
                    df = fit_to_df(file)
                    df = smooth_cols(df, window_sec_guess=15)

                    c1, c2 = st.columns(2)
                    t0 = parse_time_to_seconds(c1.text_input("Début (hh:mm:ss)", "0:00:00", key=f"t0_{test_label}"))
                    t1 = parse_time_to_seconds(c2.text_input("Fin (hh:mm:ss)", "0:12:00", key=f"t1_{test_label}"))
                    if t1 <= t0:
                        st.error("La fin doit être supérieure au début.")
                    else:
                        df_seg = segment(df, t0, t1)
                        if len(df_seg) > 10:
                            dur_s = float(df_seg["time_s"].iloc[-1] - df_seg["time_s"].iloc[0])
                            dist_m = segment_distance_m(df_seg)

                            if "heart_rate_smooth" in df_seg.columns:
                                drift_bpm, drift_pct = lin_drift(df_seg["time_s"].values, df_seg["heart_rate_smooth"].values)

                            # KPI
                            st.markdown('<div class="report-card metric-kpi">', unsafe_allow_html=True)
                            k = st.columns(4)
                            k[0].metric("FC moy", f"{df_seg['heart_rate_smooth'].mean():.1f}" if "heart_rate_smooth" in df_seg else "—")
                            k[1].metric("P moy (W)", f"{df_seg['power_smooth'].mean():.1f}" if "power_smooth" in df_seg else "—")
                            k[2].metric("V moy (km/h)", f"{(df_seg['speed_smooth'].mean()*3.6):.2f}" if "speed_smooth" in df_seg else "—")
                            k[3].metric("Dérive FC (%/min)", f"{drift_pct:.3f}" if drift_pct is not None else "—")
                            st.markdown('</div>', unsafe_allow_html=True)

                            # --- Visualisations (cases à cocher) ---
                            st.markdown("### 👁️ Visualisations")
                            show_hr = st.checkbox("📈 Fréquence cardiaque", True, key=f"hr_{test_label}")
                            show_power = st.checkbox("⚡ Puissance", True, key=f"pw_{test_label}")
                            show_speed = st.checkbox("🚀 Vitesse", False, key=f"sp_{test_label}")
                            show_cadence = st.checkbox("🦵 Cadence", False, key=f"cd_{test_label}")
                            show_alt = st.checkbox("🏔️ Altitude", False, key=f"alt_{test_label}")

                            figs = []

                            def plot_line(x, y, title, ylab, color=COLOR_BLACK):
                                fig, ax = plt.subplots(figsize=(8,3))
                                ax.plot(x, y, color=color, lw=2)
                                ax.set_title(title); ax.set_xlabel("Temps (s)"); ax.set_ylabel(ylab); ax.grid(alpha=0.25)
                                st.pyplot(fig); return fig

                            X = df_seg["time_s"] - t0
                            if show_hr and "heart_rate_smooth" in df_seg.columns:
                                figs.append(plot_line(X, df_seg["heart_rate_smooth"], f"Fréquence cardiaque — Test {test_label}", "bpm", COLOR_RED))
                            if show_power and "power_smooth" in df_seg.columns:
                                figs.append(plot_line(X, df_seg["power_smooth"], f"Puissance — Test {test_label}", "W", COLOR_BLACK))
                            if show_speed and "speed_smooth" in df_seg.columns:
                                figs.append(plot_line(X, df_seg["speed_smooth"]*3.6, f"Vitesse — Test {test_label}", "km/h", "#333"))
                            if show_cadence and "cadence_smooth" in df_seg.columns:
                                figs.append(plot_line(X, df_seg["cadence_smooth"], f"Cadence — Test {test_label}", "rpm", "#666"))
                            if show_alt and "altitude_smooth" in df_seg.columns:
                                figs.append(plot_line(X, df_seg["altitude_smooth"], f"Altitude — Test {test_label}", "m", "#999"))

                            if figs:
                                st.download_button("💾 Exporter le dernier graphique (PNG)",
                                                   data=fig_to_png_bytes(figs[-1]),
                                                   file_name=f"graph_test_{test_label}.png",
                                                   mime="image/png")

                            # Stockage pour la page comparaisons
                            state[f"seg_{test_label}"] = df_seg
                            state[f"t0_{test_label}"] = t0
                            state[f"date_{test_label}"] = date_test
                            state[f"dur_{test_label}"] = dur_s
                            state[f"dist_{test_label}"] = dist_m
                            state[f"drift_bpm_{test_label}"] = drift_bpm
                            state[f"drift_pct_{test_label}"] = drift_pct
                except Exception as e:
                    st.error(f"Erreur Test {test_label} : {e}")

# ---------- Onglet 2 : ANALYSE DE SÉANCE ----------
with tabs[1]:
    st.header("📈 Analyse de séance (multi-intervalles, indépendante)")
    fileS = st.file_uploader("Importer la séance (.fit)", type=["fit"], key="file_session")
    if "n_intervals" not in st.session_state: st.session_state["n_intervals"] = 3

    cadd, crem, _ = st.columns([1,1,6])
    with cadd:
        if st.button("➕ Ajouter"):
            st.session_state["n_intervals"] += 1
    with crem:
        if st.button("➖ Retirer") and st.session_state["n_intervals"]>1:
            st.session_state["n_intervals"] -= 1

    if fileS:
        try:
            dfS = fit_to_df(fileS)
            dfS = smooth_cols(dfS, window_sec_guess=15)
            st.caption(f"{len(dfS)} points • fenêtre de lissage ~15 s")

            rows=[]
            for i in range(st.session_state["n_intervals"]):
                st.markdown(f"**Intervalle {i+1}**")
                c1,c2 = st.columns(2)
                t0 = parse_time_to_seconds(c1.text_input(f"Début {i+1}", "0:00:00", key=f"S_t0_{i}"))
                t1 = parse_time_to_seconds(c2.text_input(f"Fin {i+1}", "0:05:00", key=f"S_t1_{i}"))
                if t1<=t0:
                    st.warning(f"Intervalle {i+1} ignoré (fin <= début)."); continue
                seg = segment(dfS, t0, t1)
                if len(seg)<=5:
                    st.warning(f"Intervalle {i+1} trop court."); continue

                dur = float(seg["time_s"].iloc[-1] - seg["time_s"].iloc[0])
                dist = segment_distance_m(seg)
                drift_bpm, drift_pct = (None, None)
                if "heart_rate_smooth" in seg.columns:
                    drift_bpm, drift_pct = lin_drift(seg["time_s"].values, seg["heart_rate_smooth"].values)

                rows.append({
                    "Intervalle": i+1,
                    "Durée (s)": f"{dur:.1f}",
                    "Distance (m)": f"{dist:.1f}",
                    "FC moy (bpm)": f"{seg['heart_rate_smooth'].mean():.1f}" if "heart_rate_smooth" in seg.columns else "—",
                    "P moy (W)": f"{seg['power_smooth'].mean():.1f}" if "power_smooth" in seg.columns else "—",
                    "V moy (km/h)": f"{(seg['speed_smooth'].mean()*3.6):.2f}" if "speed_smooth" in seg.columns else "—",
                    "Cadence (rpm)": f"{seg['cadence_smooth'].mean():.1f}" if "cadence_smooth" in seg.columns else "—",
                    "Dérive FC (%/min)": f"{drift_pct:.4f}" if drift_pct is not None else "—"
                })

            if rows:
                df_out = pd.DataFrame(rows)
                st.dataframe(df_out, use_container_width=True)
                st.download_button("💾 Export CSV séance", df_out.to_csv(index=False).encode("utf-8"),
                                   "seance_intervalles.csv", "text/csv")
        except Exception as e:
            st.error(f"Erreur séance : {e}")

# ---------- Onglet 3 : ANALYSE GÉNÉRALE (comparaisons + VC/IC + rapport) ----------
with tabs[2]:
    st.header("🧠 Analyse générale : Comparaisons Test A vs Test B, VC, IC & Rapport PNG")

    segA = st.session_state.get("seg_A")
    segB = st.session_state.get("seg_B")
    t0A  = st.session_state.get("t0_A", 0)
    t0B  = st.session_state.get("t0_B", 0)
    dateA= st.session_state.get("date_A")
    dateB= st.session_state.get("date_B")

    driftA_bpm = st.session_state.get("drift_bpm_A")
    driftA_pct = st.session_state.get("drift_pct_A")
    driftB_bpm = st.session_state.get("drift_bpm_B")
    driftB_pct = st.session_state.get("drift_pct_B")

    distA = st.session_state.get("dist_A"); durA = st.session_state.get("dur_A")
    distB = st.session_state.get("dist_B"); durB = st.session_state.get("dur_B")

    # === Coches (UNE SEULE ZONE) pour sélectionner les comparaisons à afficher ===
    st.markdown("### ⚙️ Sélection des métriques à comparer (graphiques superposés)")
    compare_hr      = st.checkbox("📈 Fréquence cardiaque", value=True)
    compare_power   = st.checkbox("⚡ Puissance", value=False)
    compare_speed   = st.checkbox("🚀 Vitesse", value=False)
    compare_cadence = st.checkbox("🦵 Cadence", value=False)
    compare_alt     = st.checkbox("🏔️ Altitude", value=False)

    # === VC (2 points) & IC
    vc_dict = None
    IC_value=IC_unite=IC_msg=IC_reco=None

    if all(v is not None for v in [distA, distB, durA, durB]) and (durA>0 and durB>0):
        D1, T1 = float(distA), float(durA)
        D2, T2 = float(distB), float(durB)
        if (T2 != T1) and all(v>0 for v in [D1,D2,T1,T2]):
            CS = (D2 - D1) / (T2 - T1)
            D_prime = D1 - CS*T1
            V_kmh = CS*3.6
            pace = format_pace_min_per_km(V_kmh)
            pace_str = f"{pace[0]}:{pace[1]:02d} min/km" if pace else "—"
            vc_dict = {"CS": CS, "V_kmh": V_kmh, "D_prime": D_prime, "pace_str": pace_str}

    # IC (court vs long par durée)
    if (durA is not None) and (durB is not None):
        if durA <= durB:
            d_short_bpm, d_long_bpm = driftA_bpm, driftB_bpm
            d_short_pct, d_long_pct = driftA_pct, driftB_pct
            label_short, label_long = "Test A", "Test B"
        else:
            d_short_bpm, d_long_bpm = driftB_bpm, driftA_bpm
            d_short_pct, d_long_pct = driftB_pct, driftA_pct
            label_short, label_long = "Test B", "Test A"

        IC_value, IC_unite, IC_msg, IC_level, IC_reco = compute_index_cinetique(
            d_short_pct, d_long_pct, d_short_bpm, d_long_bpm
        )

    # === Bandeau synthèse VC/IC
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("🧾 Synthèse")
    kpi = st.columns(4)
    if vc_dict:
        kpi[0].metric("CS (m/s)", f"{vc_dict['CS']:.2f}")
        kpi[1].metric("VC (km/h)", f"{vc_dict['V_kmh']:.2f}")
        kpi[2].metric("Allure VC", vc_dict["pace_str"])
        kpi[3].metric("D′ (m)", f"{vc_dict['D_prime']:.0f}")
    else:
        st.caption("VC non calculable (vérifie distances/temps).")

    c2 = st.columns(2)
    with c2[0]:
        st.markdown("**Dates**")
        st.write(f"- Test A : {dateA}")
        st.write(f"- Test B : {dateB}")
    with c2[1]:
        st.markdown("**Index de cinétique**")
        if IC_value is not None:
            st.write(f"IC = **{IC_value:.3f}** *(court = {label_short}, long = {label_long}, unité = {IC_unite})*")
            st.caption(IC_msg)
        else:
            st.write("— Non calculable —")
    st.markdown('</div>', unsafe_allow_html=True)

    # === Aide : petite fonction de comparaison par métrique
    def plot_compare_metric(name, colname, unit, colorA=COLOR_RED, colorB=COLOR_BLACK, factor=1.0):
        if segA is None or segB is None: 
            st.info("⚠️ Charge d'abord les deux tests dans l'onglet 1."); 
            return
        if (colname not in segA.columns) or (colname not in segB.columns):
            st.warning(f"{name} indisponible sur un des tests.")
            return
        fig, ax = plt.subplots(figsize=(9,3.2))
        ax.plot(segA["time_s"]-segA["time_s"].iloc[0], segA[colname]*factor, color=colorA, lw=1.9, label=f"Test A ({dateA})")
        ax.plot(segB["time_s"]-segB["time_s"].iloc[0], segB[colname]*factor, color=colorB, lw=1.9, label=f"Test B ({dateB})")
        ax.set_title(f"{name} – Comparatif Test A / Test B")
        ax.set_xlabel("Temps (s)"); ax.set_ylabel(unit); ax.grid(alpha=0.25); ax.legend()
        st.pyplot(fig); plt.close(fig)

        # Tableau Δ
        avgA = float(np.nanmean(segA[colname]))*factor
        avgB = float(np.nanmean(segB[colname]))*factor
        diff_abs = avgB - avgA
        diff_pct = (diff_abs/avgA*100.0) if avgA else 0.0
        df_diff = pd.DataFrame({
            "Test A Moy": [f"{avgA:.2f} {unit}"],
            "Test B Moy": [f"{avgB:.2f} {unit}"],
            "Δ absolu": [f"{diff_abs:+.2f} {unit}"],
            "Δ relatif": [f"{diff_pct:+.2f}%"]
        })
        st.dataframe(df_diff, use_container_width=True)

    st.markdown("---")

    # === Affichage vertical des comparaisons selon les coches (UNS SOUS LES AUTRES) ===
    if compare_hr:
        st.subheader("📈 Fréquence cardiaque")
        plot_compare_metric("Fréquence cardiaque", "heart_rate_smooth", "bpm", COLOR_RED, COLOR_BLACK, 1.0)

    if compare_power:
        st.subheader("⚡ Puissance")
        plot_compare_metric("Puissance", "power_smooth", "W", COLOR_RED, COLOR_BLACK, 1.0)

    if compare_speed:
        st.subheader("🚀 Vitesse")
        # speed en m/s → km/h (×3.6)
        plot_compare_metric("Vitesse", "speed_smooth", "km/h", COLOR_RED, COLOR_BLACK, 3.6)

    if compare_cadence:
        st.subheader("🦵 Cadence")
        plot_compare_metric("Cadence", "cadence_smooth", "rpm", COLOR_RED, COLOR_BLACK, 1.0)

    if compare_alt:
        st.subheader("🏔️ Altitude")
        plot_compare_metric("Altitude", "altitude_smooth", "m", COLOR_RED, COLOR_BLACK, 1.0)

    st.markdown("---")

    # === Rapport PNG haute qualité (global) ===
    st.subheader("🖼️ Rapport complet (PNG)")
    # Construire des lignes de résumé Test A / B
    statsA_lines=[]; statsB_lines=[]
    if (distA is not None) and (durA is not None) and durA>0:
        v1=(distA/durA)*3.6
        statsA_lines += [f"Durée: {durA:.1f}s", f"Distance: {distA:.1f} m", f"Vitesse: {v1:.2f} km/h ({format_pace_from_kmh(v1)})"]
    if driftA_bpm is not None: statsA_lines.append(f"Dérive FC: {driftA_bpm:.4f} bpm/min")
    if driftA_pct is not None: statsA_lines.append(f"Dérive FC: {driftA_pct:.4f} %/min")

    if (distB is not None) and (durB is not None) and durB>0:
        v2=(distB/durB)*3.6
        statsB_lines += [f"Durée: {durB:.1f}s", f"Distance: {distB:.1f} m", f"Vitesse: {v2:.2f} km/h ({format_pace_from_kmh(v2)})"]
    if driftB_bpm is not None: statsB_lines.append(f"Dérive FC: {driftB_bpm:.4f} bpm/min")
    if driftB_pct is not None: statsB_lines.append(f"Dérive FC: {driftB_pct:.4f} %/min")

    png = render_full_report_png(
        title="Rapport complet – Endurance & Comparaisons (Rouge/Noir/Blanc)",
        date1=dateA, date2=dateB,
        df1_seg=segA, t0_1=t0A, stats1_lines=statsA_lines, dist1_m=distA, t1_s=durA,
        df2_seg=segB, t0_2=t0B, stats2_lines=statsB_lines, dist2_m=distB, t2_s=durB,
        vc_dict=vc_dict, IC_value=IC_value, IC_unite=IC_unite, IC_msg=IC_msg, IC_reco=IC_reco
    )
    st.download_button("💾 Télécharger le RAPPORT COMPLET (PNG)", data=png,
                       file_name="rapport_complet_endurance_comparaisons.png", mime="image/png")
