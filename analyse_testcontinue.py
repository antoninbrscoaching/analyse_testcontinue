# ============================
# 🏃‍♂️ Endurance Suite : Tests fusionnés + Analyse de séance + VC/IC + Rapport PNG
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

# =============== UI / THEME =================
st.set_page_config(page_title="Endurance Suite • Tests + Séances + VC/IC", layout="wide")

# Palette rouge/noir/blanc
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
.subtle { color: #6b7280; font-size: 0.92rem; }
.section-title { margin-top: .6rem; margin-bottom: .4rem; }
hr { border: none; border-top: 1px solid #eee; margin: 1.2rem 0; }
.block-container { padding-top: 1.2rem; }
.metric-kpi .stMetric { background: #fff; border:1px solid #eee; border-radius:10px; padding:10px;}
.small {font-size:.92rem;color:#6b7280}
h3, h4 { margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# =============== UTILS ======================

FIT_NAME_MAP = {
    "heart_rate": ["heart_rate","hr","heartrate"],
    "power": ["power","watts","pwr"],
    "cadence": ["cadence","cad"],
    "speed": ["speed","spd"],
    "altitude": ["altitude","alt","elev"],
    "distance": ["distance","dist"],
    "timestamp": ["timestamp","time","date_time","ts"]
}

def colname(df, target):
    """Trouve un nom de colonne probable pour un champ 'target'."""
    targets = FIT_NAME_MAP.get(target, [target])
    low = {c.lower(): c for c in df.columns}
    for t in targets:
        if t in low: return low[t]
    # fallback : contient …
    for c in df.columns:
        if target in c.lower(): return c
    return None

def load_activity(file):
    """Charge FIT/GPX/CSV. Retourne df avec colonnes harmonisées si dispo."""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)

    elif file.name.endswith(".fit"):
        data = []
        try:
            with fitdecode.FitReader(file) as fit:
                for frame in fit:
                    if isinstance(frame, fitdecode.records.FitDataMessage) and frame.name == "record":
                        record_data = {f.name: f.value for f in frame.fields}
                        # Convert lat/long semicircles → degrés si présent
                        if "position_lat" in record_data and "position_long" in record_data:
                            try:
                                record_data["lat"] = record_data["position_lat"] * (180 / 2**31)
                                record_data["lon"] = record_data["position_long"] * (180 / 2**31)
                            except Exception:
                                pass
                        data.append(record_data)
            df = pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Erreur lecture FIT : {e}")

    elif file.name.endswith(".gpx"):
        import gpxpy
        gpx = gpxpy.parse(file)
        data = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    data.append({
                        "timestamp": point.time,
                        "lat": point.latitude,
                        "lon": point.longitude,
                        "altitude": point.elevation
                    })
        df = pd.DataFrame(data)
    else:
        raise ValueError("Format non supporté (utilise .fit, .gpx ou .csv).")

    # Harmoniser timestamp si possible
    tcol = colname(df, "timestamp")
    if tcol:
        df.rename(columns={tcol: "timestamp"}, inplace=True)

    # Harmoniser principaux champs si présents
    for key in ["heart_rate","power","cadence","speed","altitude","distance"]:
        c = colname(df, key)
        if c and c != key:
            df.rename(columns={c: key}, inplace=True)

    # Normaliser
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Vérification minimale
    if "heart_rate" not in df.columns:
        st.warning("⚠️ Fréquence cardiaque absente. Les dérives FC ne seront pas calculées.")
    return df

def build_time_and_smooth(df, time_col="timestamp", hr_col="heart_rate"):
    """Temps continu (ignore grandes pauses) + lissage FC et autres colonnes numériques."""
    df = df.copy()
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

        df["delta_t"] = df[time_col].diff().dt.total_seconds().fillna(0)
        # Détection cadence d'enregistrement médiane
        median_step = np.median(df["delta_t"][df["delta_t"] > 0]) if (df["delta_t"] > 0).any() else 1
        if np.isnan(median_step) or median_step <= 0: median_step = 1
        # Limiter grosses pauses (converties en pas médian)
        df.loc[df["delta_t"] > 2*median_step, "delta_t"] = median_step
        df["time_s"] = df["delta_t"].cumsum()
    else:
        # fallback temps incrémental si pas de timestamp
        df["time_s"] = np.arange(len(df), dtype=float)
        median_step = 1

    total_dur = float(df["time_s"].iloc[-1]) if len(df) else 0.0

    # taille fenêtre de lissage selon durée
    if total_dur < 360: window_sec = 5
    elif total_dur < 900: window_sec = 10
    else: window_sec = 20
    step = np.median(np.diff(df["time_s"])) if len(df) > 1 else 1
    if not np.isfinite(step) or step <= 0: step = 1
    win = max(1, int(round(window_sec / step)))

    # Lissage colonnes numériques utiles si présentes
    for c in ["heart_rate","power","cadence","speed","altitude"]:
        if c in df.columns:
            df[f"{c}_smooth"] = pd.to_numeric(df[c], errors="coerce").rolling(win, min_periods=1).mean()

    pauses = int((df["delta_t"] > 2*median_step).sum()) if "delta_t" in df.columns else 0
    return df, window_sec, total_dur, pauses

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

def segment(df, t0, t1):
    """Sous-données entre t0 et t1 en 'time_s'."""
    if df is None or df.empty: return df.iloc[:0]
    t1 = min(t1, float(df["time_s"].max()))
    return df[(df["time_s"] >= t0) & (df["time_s"] <= t1)].copy()

def lin_drift(x, y):
    """Pente de régression (unité y / s) et %/min si applicable."""
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5: return None, None
    slope, _, _, _, _ = linregress(x[mask], y[mask])
    per_min = slope * 60.0
    pct_per_min = None
    mean_y = np.nanmean(y[mask])
    if mean_y and np.isfinite(mean_y) and mean_y != 0:
        pct_per_min = (per_min / mean_y) * 100.0
    return float(per_min), (None if pct_per_min is None else float(pct_per_min))

def metric_block(cols, stats):
    """Affiche un bloc de 4 métriques (libre)."""
    for i, (k,v) in enumerate(stats.items()):
        with cols[i % len(cols)]:
            st.metric(k, v)

def haversine_dist_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def segment_distance_m(df_seg):
    """Distance: priorité distance cumulative > speed*dt > haversine."""
    if df_seg is None or df_seg.empty or len(df_seg) < 2: return 0.0
    if "distance" in df_seg.columns:
        d0 = float(df_seg["distance"].iloc[0]); d1 = float(df_seg["distance"].iloc[-1])
        if np.isfinite(d0) and np.isfinite(d1): return max(0.0, d1 - d0)
    if ("speed_smooth" in df_seg.columns) and ("time_s" in df_seg.columns):
        dt = np.diff(df_seg["time_s"].values)
        sp = df_seg["speed_smooth"].iloc[1:].values
        dist = float(np.nansum(np.maximum(dt,0)*np.maximum(sp,0)))
        if dist > 0: return dist
    if "lat" in df_seg.columns and "lon" in df_seg.columns:
        lats = df_seg["lat"].astype(float).values
        lons = df_seg["lon"].astype(float).values
        dist = 0.0
        for i in range(1,len(lats)):
            if all(np.isfinite([lats[i-1],lats[i],lons[i-1],lons[i]])):
                dist += haversine_dist_m(lats[i-1],lons[i-1],lats[i],lons[i])
        if dist > 0: return dist
    return 0.0

def format_pace_min_per_km_from_speed_ms(v_ms):
    if v_ms <= 0 or not math.isfinite(v_ms): return "—"
    v_kmh = v_ms * 3.6
    min_per_km = 60.0 / v_kmh
    total_seconds = int(round(min_per_km * 60.0))
    return f"{total_seconds//60}:{total_seconds%60:02d} min/km"

def fig_to_png_bytes(fig, dpi=240):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf

def analyze_interval(df_seg):
    """Retourne dict de stats pour HR/Power/Cadence/Speed si dispos + dérives."""
    out = {}
    dur = float(df_seg["time_s"].iloc[-1] - df_seg["time_s"].iloc[0]) if len(df_seg)>1 else 0.0
    out["Durée (s)"] = f"{dur:.1f}"

    # HR
    if "heart_rate_smooth" in df_seg.columns:
        hr = df_seg["heart_rate_smooth"].values
        out["FC moy (bpm)"] = f"{np.nanmean(hr):.1f}"
        out["FC max (bpm)"] = f"{np.nanmax(hr):.1f}"
        drift_bpm, drift_pct = lin_drift(df_seg["time_s"].values, hr)
        out["Dérive FC (bpm/min)"] = f"{(drift_bpm or 0):.4f}"
        out["Dérive FC (%/min)"] = ("—" if drift_pct is None else f"{drift_pct:.4f}")

    # POWER
    if "power_smooth" in df_seg.columns:
        p = df_seg["power_smooth"].values
        out["Puissance moy (W)"] = f"{np.nanmean(p):.1f}"
        out["Puissance max (W)"] = f"{np.nanmax(p):.1f}"
        dp_bpm, dp_pct = lin_drift(df_seg["time_s"].values, p)
        out["Dérive Pwr (W/min)"] = f"{(dp_bpm or 0):.4f}"
        out["Dérive Pwr (%/min)"] = ("—" if dp_pct is None else f"{dp_pct:.4f}")

    # CADENCE
    if "cadence_smooth" in df_seg.columns:
        c = df_seg["cadence_smooth"].values
        out["Cadence moy (rpm)"] = f"{np.nanmean(c):.1f}"

    # SPEED & PACE
    dist_m = segment_distance_m(df_seg)
    v_ms = dist_m / dur if dur>0 else 0.0
    out["Distance (m)"] = f"{dist_m:.1f}"
    out["Vitesse moy (km/h)"] = f"{v_ms*3.6:.2f}"
    out["Allure moy"] = format_pace_min_per_km_from_speed_ms(v_ms)
    return out, dist_m, dur

# ====== Index de cinétique & VC ======

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
        niveau = "tres_bon"; titre = "Très bonne stabilité sur le long"
        points = [f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}",
                  "Profil endurant fort, tolérance aux blocs prolongés."]
        seances = ["2–3×(8–12′) à 88–92% VC, r=2–3′","Tempo 20–30′ à 85–90% VC","Progressif 30–40′ de 80→90% VC","Z2 volumineux"]
        msg = "IC élevé : orientation blocs longs & tempos ambitieux."
    elif 0.40 <= IC < 0.70:
        niveau = "bon"; titre = "Bon équilibre, marge en soutien aérobie"
        points = [f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}",
                  "Équilibre intéressant, consolider le tempo/seuil."]
        seances = ["4–6×5′ à 90–92% VC, r=1–2′ (cruise)","2×12–15′ à 85–90% VC (tempo)","6–8×(2′ @95% VC / 1′ @80%)"]
        msg = "IC bon : mix intervals moyens + tempo."
    elif 0.15 <= IC < 0.40:
        niveau = "moyen"; titre = "Stabilité limitée sur le long"
        points = [f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}",
                  "Priorité à étendre la tolérance au long, lisser la cinétique."]
        seances = ["3–4×6′ à 88–90% VC, r=2′","3×8–10′ à 85–88% VC","Z2 + 6–10×20″ strides"]
        msg = "IC moyen : allonger progressivement les intervalles."
    elif 0.00 <= IC < 0.15:
        niveau = "faible"; titre = "Dérives longue et courte similaires"
        points = [f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}",
                  "Base aérobie à renforcer, démarrer par du tempo doux."]
        seances = ["Z2 majoritaire","3–4×6–8′ à 82–86% VC","10–12×1′ à 92–95% VC / 1′ Z2"]
        msg = "IC faible : focus base + tempo doux."
    else:
        niveau = "degrade"; titre = "Stabilité faible / contexte défavorable"
        points = [f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}",
                  "Réduire l'intensité, reconstruire la base, vérifier conditions."]
        seances = ["Z2 + force (côtes)","Progressifs doux","Limiter >90% VC ; revoir sommeil/chaleur/hydrat."]
        msg = "IC négatif : re-baser et diagnostiquer."
    reco = {"titre": titre, "points": points, "seances": seances}
    return float(IC), unite, msg, niveau, reco

def format_pace_from_kmh(v_kmh):
    if v_kmh <= 0 or not math.isfinite(v_kmh): return "—"
    mpk = 60.0 / v_kmh; total = int(round(mpk*60))
    return f"{total//60}:{total%60:02d} min/km"

# ====== Rapport PNG propre (aucune superposition) ======

def render_full_report_png(
    title, date1, date2,
    df1_seg, t0_1, stats1, dist1_m, t1_s,
    df2_seg, t0_2, stats2, dist2_m, t2_s,
    vc_dict, IC_value, IC_unite, IC_msg, IC_reco
):
    mpl.rcParams.update({
        "axes.edgecolor": COLOR_BLACK, "axes.labelcolor": COLOR_BLACK,
        "xtick.color": COLOR_BLACK, "ytick.color": COLOR_BLACK,
        "text.color": COLOR_BLACK, "figure.facecolor": COLOR_WHITE,
        "axes.facecolor": COLOR_WHITE, "savefig.facecolor": COLOR_WHITE,
        "font.size": 10
    })

    fig = plt.figure(figsize=(11, 17), dpi=220)
    gs = fig.add_gridspec(
        9, 2,
        height_ratios=[0.55, 1.15, 1.15, 0.8, 0.8, 0.95, 0.95, 0.95, 0.8],
        width_ratios=[1, 1],
        hspace=0.9, wspace=0.8
    )

    # En-tête
    ax_title = fig.add_subplot(gs[0, :]); ax_title.axis("off")
    ax_title.add_patch(plt.Rectangle((0,0),1,1, color=BG_PANEL, transform=ax_title.transAxes))
    ax_title.text(0.02, 0.78, title, fontsize=22, fontweight="bold", color=COLOR_BLACK, va="top")
    ax_title.text(0.02, 0.50, f"Test 1 : {date1}   •   Test 2 : {date2}", fontsize=11, color=COLOR_GREY)
    ax_title.text(0.02, 0.25, "FC, Puissance, Dérives, Distances, VC, Index de cinétique, Prescription", fontsize=10, color=COLOR_GREY)

    # Test 1: FC
    ax1 = fig.add_subplot(gs[1, :])
    ax1.set_title("Cinétique — Test 1 (FC + Power)", fontsize=12, fontweight="bold")
    if df1_seg is not None and len(df1_seg) > 1:
        ax1.plot(df1_seg["time_s"] - t0_1, df1_seg.get("heart_rate_smooth", df1_seg.get("heart_rate", np.nan)), color=COLOR_RED, lw=2, label="FC")
        if "power_smooth" in df1_seg.columns or "power" in df1_seg.columns:
            ax1b = ax1.twinx()
            ax1b.plot(df1_seg["time_s"] - t0_1, df1_seg.get("power_smooth", df1_seg.get("power", np.nan)), color=COLOR_BLACK, lw=1.6, alpha=0.8, label="Puissance")
            ax1b.set_ylabel("Puissance (W)")
        ax1.set_xlabel("Temps (s)"); ax1.set_ylabel("FC (bpm)"); ax1.grid(True, alpha=0.2)

    # Test 2: FC
    ax2 = fig.add_subplot(gs[2, :])
    ax2.set_title("Cinétique — Test 2 (FC + Power)", fontsize=12, fontweight="bold")
    if df2_seg is not None and len(df2_seg) > 1:
        ax2.plot(df2_seg["time_s"] - t0_2, df2_seg.get("heart_rate_smooth", df2_seg.get("heart_rate", np.nan)), color=COLOR_RED, lw=2, label="FC")
        if "power_smooth" in df2_seg.columns or "power" in df2_seg.columns:
            ax2b = ax2.twinx()
            ax2b.plot(df2_seg["time_s"] - t0_2, df2_seg.get("power_smooth", df2_seg.get("power", np.nan)), color=COLOR_BLACK, lw=1.6, alpha=0.8, label="Puissance")
            ax2b.set_ylabel("Puissance (W)")
        ax2.set_xlabel("Temps (s)"); ax2.set_ylabel("FC (bpm)"); ax2.grid(True, alpha=0.2)

    # Stats cards
    ax4 = fig.add_subplot(gs[3, 0]); ax5 = fig.add_subplot(gs[3, 1])
    for ax in (ax4, ax5):
        ax.axis("off"); ax.add_patch(plt.Rectangle((0,0),1,1, color=BG_PANEL, transform=ax.transAxes))
    if stats1:
        ax4.text(0.04, 0.85, "Résumé Test 1", fontsize=12, fontweight="bold")
        for i, line in enumerate(stats1, start=0):
            ax4.text(0.06, 0.75 - i*0.08, line, fontsize=10, color=COLOR_BLACK)
    if stats2:
        ax5.text(0.04, 0.85, "Résumé Test 2", fontsize=12, fontweight="bold")
        for i, line in enumerate(stats2, start=0):
            ax5.text(0.06, 0.75 - i*0.08, line, fontsize=10, color=COLOR_BLACK)

    # Comparatif FC
    ax6 = fig.add_subplot(gs[4, :])
    ax6.set_title("Comparatif FC (segments centrés en t=0)", fontsize=12, fontweight="bold")
    if df1_seg is not None and len(df1_seg) > 1:
        ax6.plot(df1_seg["time_s"] - df1_seg["time_s"].iloc[0], df1_seg.get("heart_rate_smooth", np.nan), color=COLOR_RED, lw=2, label="Test 1")
    if df2_seg is not None and len(df2_seg) > 1:
        ax6.plot(df2_seg["time_s"] - df2_seg["time_s"].iloc[0], df2_seg.get("heart_rate_smooth", np.nan), color=COLOR_BLACK, lw=2, label="Test 2")
    ax6.legend(frameon=False); ax6.set_xlabel("Temps (s)"); ax6.set_ylabel("FC (bpm)"); ax6.grid(True, alpha=0.2)

    # VC + IC (2 cartes)
    ax7L = fig.add_subplot(gs[5, 0]); ax7R = fig.add_subplot(gs[5, 1])
    for ax in (ax7L, ax7R):
        ax.axis("off"); ax.add_patch(plt.Rectangle((0,0),1,1, color=BG_PANEL, transform=ax.transAxes))
    ax7L.text(0.04, 0.84, "Vitesse Critique", fontsize=12, fontweight="bold")
    if vc_dict:
        lines_vc = [f"CS : {vc_dict['CS']:.2f} m/s", f"VC : {vc_dict['V_kmh']:.2f} km/h", f"Allure VC : {vc_dict['pace_str']}", f"D′ : {vc_dict['D_prime']:.0f} m"]
        for i,l in enumerate(lines_vc): ax7L.text(0.06, 0.74 - i*0.08, l, fontsize=10, color=COLOR_BLACK)
    else:
        ax7L.text(0.06, 0.74, "— Non calculable —", fontsize=10, color=COLOR_GREY)

    ax7R.text(0.04, 0.84, "Index de cinétique (IC)", fontsize=12, fontweight="bold")
    if IC_value is not None:
        ax7R.text(0.06, 0.74, f"IC : {IC_value:.3f}", fontsize=10)
        ax7R.text(0.06, 0.66, f"Unité dérives : {IC_unite}", fontsize=10)
        ax7R.text(0.06, 0.58, IC_msg, fontsize=10)
    else:
        ax7R.text(0.06, 0.74, "— Non calculable —", fontsize=10, color=COLOR_GREY)

    # Reco (bandeau)
    ax8 = fig.add_subplot(gs[6, :]); ax8.axis("off")
    ax8.add_patch(plt.Rectangle((0,0),1,1, color=BG_PANEL, transform=ax8.transAxes))
    ax8.text(0.03, 0.86, "Prescription (4–8 semaines)", fontsize=12, fontweight="bold")
    if IC_reco:
        y = 0.74; ax8.text(0.04, y, f"• {IC_reco['titre']}", fontsize=10, color=COLOR_BLACK); y -= 0.08
        for p in IC_reco["points"]:
            ax8.text(0.05, y, f"– {p}", fontsize=10, color=COLOR_BLACK); y -= 0.06
        y -= 0.04; ax8.text(0.04, y, "Séances types :", fontsize=10, color=COLOR_BLACK); y -= 0.08
        for s in IC_reco["seances"]:
            ax8.text(0.05, y, f"• {s}", fontsize=10, color=COLOR_BLACK); y -= 0.06
    else:
        ax8.text(0.04, 0.74, "— Aucune recommandation —", fontsize=10, color=COLOR_GREY)

    # Bandes infos bas : distances & vitesses
    ax9L = fig.add_subplot(gs[7, 0]); ax9R = fig.add_subplot(gs[7, 1])
    for ax in (ax9L, ax9R):
        ax.axis("off"); ax.add_patch(plt.Rectangle((0,0),1,1, color=BG_PANEL, transform=ax.transAxes))
    if (dist1_m is not None) and (t1_s is not None) and t1_s>0:
        v1 = (dist1_m/t1_s)*3.6; ax9L.text(0.05,0.75,f"Test 1 : {dist1_m:.1f} m • {v1:.2f} km/h • {format_pace_from_kmh(v1)}", fontsize=10)
    if (dist2_m is not None) and (t2_s is not None) and t2_s>0:
        v2 = (dist2_m/t2_s)*3.6; ax9R.text(0.05,0.75,f"Test 2 : {dist2_m:.1f} m • {v2:.2f} km/h • {format_pace_from_kmh(v2)}", fontsize=10)

    # Pied rouge
    ax_footer = fig.add_subplot(gs[8, :]); ax_footer.axis("off")
    ax_footer.add_patch(plt.Rectangle((0, 0.0), 1, 0.18, color=COLOR_RED, transform=ax_footer.transAxes))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=280, bbox_inches="tight")
    buf.seek(0); plt.close(fig)
    return buf

# =============== APP ========================

tabs = st.tabs(["🧪 Tests d’endurance (fusion)", "📈 Analyse de séance", "🧠 Analyse générale"])

# ----- Page 1 : Tests fusionnés -----
with tabs[0]:
    st.header("🧪 Tests d’endurance (Test A et Test B)")

    colU = st.columns(2)

    # ===== Colonne gauche : Test A =====
    with colU[0]:
        st.subheader("Test A")
        uploaded_A = st.file_uploader("Importer Test A (FIT/GPX/CSV)", type=["fit","gpx","csv"], key="A_file")
        date_A = st.date_input("📅 Date Test A", value=date.today(), key="A_date")

        dfA = None; dfA_seg=None; statsA=None; distA=tA=None
        driftA_bpm = driftA_pct = None

        if uploaded_A:
            try:
                dfA = load_activity(uploaded_A)
                dfA, winA, durA, pauseA = build_time_and_smooth(dfA)
                st.caption(f"Durée détectée: {durA:.1f}s • Lissage: {winA}s • Pauses: {pauseA}")

                c1,c2 = st.columns(2)
                startA = parse_time_to_seconds(c1.text_input("Début (h:mm:ss)", "0:00:00", key="A_start"))
                endA   = parse_time_to_seconds(c2.text_input("Fin (h:mm:ss)",   "0:12:00", key="A_end"))
                if endA <= startA:
                    st.error("Fin doit être > début (Test A).")
                else:
                    dfA_seg = segment(dfA, startA, endA)
                    if len(dfA_seg)>10:
                        # stats génériques
                        outA, distA, tA = analyze_interval(dfA_seg)
                        statsA_lines = []
                        if "FC moy (bpm)" in outA:
                            statsA_lines += [
                                f"FC moy: {outA['FC moy (bpm)']}",
                                f"FC max: {outA['FC max (bpm)']}",
                                f"Dérive FC: {outA['Dérive FC (bpm/min)']} bpm/min",
                                f"Dérive FC: {outA['Dérive FC (%/min)']} %/min",
                            ]
                            driftA_bpm = float(outA["Dérive FC (bpm/min)"])
                            driftA_pct = None if outA["Dérive FC (%/min)"]=="—" else float(outA["Dérive FC (%/min)"])
                        if "Puissance moy (W)" in outA:
                            statsA_lines += [
                                f"P moy: {outA['Puissance moy (W)']}",
                                f"P max: {outA['Puissance max (W)']}",
                                f"Dérive P: {outA['Dérive Pwr (W/min)']} W/min",
                                f"Dérive P: {outA['Dérive Pwr (%/min)']} %/min",
                            ]
                        statsA_lines += [
                            f"Durée: {outA['Durée (s)']} s",
                            f"Distance: {outA['Distance (m)']}",
                            f"Vitesse moy: {outA['Vitesse moy (km/h)']}",
                            f"Allure moy: {outA['Allure moy']}",
                        ]

                        # KPIs
                        st.markdown('<div class="report-card metric-kpi">', unsafe_allow_html=True)
                        cK = st.columns(4)
                        metric_block(cK, {
                            "FC moy": outA.get("FC moy (bpm)","—"),
                            "Dérive FC (%/min)": outA.get("Dérive FC (%/min)","—"),
                            "Vitesse moy (km/h)": outA.get("Vitesse moy (km/h)","—"),
                            "Puissance moy (W)": outA.get("Puissance moy (W)","—"),
                        })
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Graph FC + Power
                        figA, axA = plt.subplots(figsize=(8,3))
                        xA = dfA_seg["time_s"] - startA
                        if "heart_rate_smooth" in dfA_seg.columns:
                            axA.plot(xA, dfA_seg["heart_rate_smooth"], color=COLOR_RED, label="FC (bpm)")
                            axA.set_ylabel("FC (bpm)")
                        axA.set_xlabel("Temps (s)")
                        axA.grid(alpha=0.25)
                        if "power_smooth" in dfA_seg.columns or "power" in dfA_seg.columns:
                            ax2 = axA.twinx()
                            ax2.plot(xA, dfA_seg.get("power_smooth", dfA_seg.get("power")), color=COLOR_BLACK, alpha=0.85, label="Puissance (W)")
                            ax2.set_ylabel("Puissance (W)")
                        axA.set_title(f"Test A — {date_A}")
                        st.pyplot(figA)
                        st.download_button("💾 PNG Test A", data=fig_to_png_bytes(figA), file_name="test_A.png", mime="image/png")
                        plt.close(figA)

                        # Infos listées
                        with st.expander("Détails Test A"):
                            st.write("\n".join(statsA_lines))

            except Exception as e:
                st.error(f"Erreur Test A : {e}")

    # ===== Colonne droite : Test B =====
    with colU[1]:
        st.subheader("Test B")
        uploaded_B = st.file_uploader("Importer Test B (FIT/GPX/CSV)", type=["fit","gpx","csv"], key="B_file")
        date_B = st.date_input("📅 Date Test B", value=date.today(), key="B_date")

        dfB = None; dfB_seg=None; statsB=None; distB=tB=None
        driftB_bpm = driftB_pct = None

        if uploaded_B:
            try:
                dfB = load_activity(uploaded_B)
                dfB, winB, durB, pauseB = build_time_and_smooth(dfB)
                st.caption(f"Durée détectée: {durB:.1f}s • Lissage: {winB}s • Pauses: {pauseB}")

                c1,c2 = st.columns(2)
                startB = parse_time_to_seconds(c1.text_input("Début (h:mm:ss)", "0:00:00", key="B_start"))
                endB   = parse_time_to_seconds(c2.text_input("Fin (h:mm:ss)",   "0:12:00", key="B_end"))
                if endB <= startB:
                    st.error("Fin doit être > début (Test B).")
                else:
                    dfB_seg = segment(dfB, startB, endB)
                    if len(dfB_seg)>10:
                        outB, distB, tB = analyze_interval(dfB_seg)
                        statsB_lines = []
                        if "FC moy (bpm)" in outB:
                            statsB_lines += [
                                f"FC moy: {outB['FC moy (bpm)']}",
                                f"FC max: {outB['FC max (bpm)']}",
                                f"Dérive FC: {outB['Dérive FC (bpm/min)']} bpm/min",
                                f"Dérive FC: {outB['Dérive FC (%/min)']} %/min",
                            ]
                            driftB_bpm = float(outB["Dérive FC (bpm/min)"])
                            driftB_pct = None if outB["Dérive FC (%/min)"]=="—" else float(outB["Dérive FC (%/min)"])
                        if "Puissance moy (W)" in outB:
                            statsB_lines += [
                                f"P moy: {outB['Puissance moy (W)']}",
                                f"P max: {outB['Puissance max (W)']}",
                                f"Dérive P: {outB['Dérive Pwr (W/min)']} W/min",
                                f"Dérive P: {outB['Dérive Pwr (%/min)']} %/min",
                            ]
                        statsB_lines += [
                            f"Durée: {outB['Durée (s)']} s",
                            f"Distance: {outB['Distance (m)']}",
                            f"Vitesse moy: {outB['Vitesse moy (km/h)']}",
                            f"Allure moy: {outB['Allure moy']}",
                        ]

                        st.markdown('<div class="report-card metric-kpi">', unsafe_allow_html=True)
                        cK = st.columns(4)
                        metric_block(cK, {
                            "FC moy": outB.get("FC moy (bpm)","—"),
                            "Dérive FC (%/min)": outB.get("Dérive FC (%/min)","—"),
                            "Vitesse moy (km/h)": outB.get("Vitesse moy (km/h)","—"),
                            "Puissance moy (W)": outB.get("Puissance moy (W)","—"),
                        })
                        st.markdown('</div>', unsafe_allow_html=True)

                        figB, axB = plt.subplots(figsize=(8,3))
                        xB = dfB_seg["time_s"] - startB
                        if "heart_rate_smooth" in dfB_seg.columns:
                            axB.plot(xB, dfB_seg["heart_rate_smooth"], color=COLOR_RED, label="FC (bpm)")
                            axB.set_ylabel("FC (bpm)")
                        axB.set_xlabel("Temps (s)")
                        axB.grid(alpha=0.25)
                        if "power_smooth" in dfB_seg.columns or "power" in dfB_seg.columns:
                            ax2 = axB.twinx()
                            ax2.plot(xB, dfB_seg.get("power_smooth", dfB_seg.get("power")), color=COLOR_BLACK, alpha=0.85, label="Puissance (W)")
                            ax2.set_ylabel("Puissance (W)")
                        axB.set_title(f"Test B — {date_B}")
                        st.pyplot(figB)
                        st.download_button("💾 PNG Test B", data=fig_to_png_bytes(figB), file_name="test_B.png", mime="image/png")
                        plt.close(figB)

                        with st.expander("Détails Test B"):
                            st.write("\n".join(statsB_lines))

            except Exception as e:
                st.error(f"Erreur Test B : {e}")

    # Stocker pour la page 3
    st.session_state["dfA_seg"] = dfA_seg
    st.session_state["dfB_seg"] = dfB_seg
    st.session_state["date_A"] = date_A
    st.session_state["date_B"] = date_B
    st.session_state["startA"] = (startA if uploaded_A else 0)
    st.session_state["startB"] = (startB if uploaded_B else 0)
    st.session_state["driftA_bpm"] = driftA_bpm
    st.session_state["driftA_pct"] = driftA_pct
    st.session_state["driftB_bpm"] = driftB_bpm
    st.session_state["driftB_pct"] = driftB_pct
    st.session_state["distA"] = distA; st.session_state["tA"] = tA
    st.session_state["distB"] = distB; st.session_state["tB"] = tB

# ----- Page 2 : Analyse de séance -----
with tabs[1]:
    st.header("📈 Analyse de séance (multi-intervalles)")
    fileS = st.file_uploader("Importer la séance (FIT/GPX/CSV)", type=["fit","gpx","csv"], key="S_file")
    if "n_intervals" not in st.session_state: st.session_state["n_intervals"] = 1

    ca, cb, cc = st.columns([1,1,4])
    with ca:
        if st.button("➕ Ajouter un intervalle"):
            st.session_state["n_intervals"] += 1
    with cb:
        if st.button("➖ Retirer un intervalle") and st.session_state["n_intervals"]>1:
            st.session_state["n_intervals"] -= 1

    if fileS:
        try:
            dfS = load_activity(fileS)
            dfS, winS, durS, pauseS = build_time_and_smooth(dfS)
            st.caption(f"Durée détectée: {durS:.1f}s • Lissage: {winS}s • Pauses: {pauseS}")

            rows=[]
            for i in range(st.session_state["n_intervals"]):
                st.markdown(f"**Intervalle {i+1}**")
                c1,c2 = st.columns(2)
                t0 = parse_time_to_seconds(c1.text_input(f"Début {i+1} (h:mm:ss)", "0:00:00", key=f"S_t0_{i}"))
                t1 = parse_time_to_seconds(c2.text_input(f"Fin {i+1} (h:mm:ss)",   "0:05:00", key=f"S_t1_{i}"))
                if t1 <= t0:
                    st.warning(f"Intervalle {i+1}: fin <= début, ignoré.")
                    continue
                seg = segment(dfS, t0, t1)
                if len(seg)<=5:
                    st.warning(f"Intervalle {i+1}: trop court.")
                    continue
                stats, d_m, dur = analyze_interval(seg)
                row = {"Intervalle": i+1, **stats}
                rows.append(row)

            if rows:
                df_out = pd.DataFrame(rows)
                st.dataframe(df_out, use_container_width=True)
                # Export
                csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Export CSV séance", data=csv, file_name="seance_intervalles.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Erreur séance : {e}")

# ----- Page 3 : Analyse générale -----
with tabs[2]:
    st.header("🧠 Analyse générale : VC, IC & Rapport PNG")

    dfA_seg = st.session_state.get("dfA_seg")
    dfB_seg = st.session_state.get("dfB_seg")
    startA  = st.session_state.get("startA",0)
    startB  = st.session_state.get("startB",0)
    date_A  = st.session_state.get("date_A")
    date_B  = st.session_state.get("date_B")

    driftA_bpm = st.session_state.get("driftA_bpm")
    driftA_pct = st.session_state.get("driftA_pct")
    driftB_bpm = st.session_state.get("driftB_bpm")
    driftB_pct = st.session_state.get("driftB_pct")

    distA = st.session_state.get("distA"); tA = st.session_state.get("tA")
    distB = st.session_state.get("distB"); tB = st.session_state.get("tB")

    vc_dict = None; IC_value=IC_unite=IC_msg=IC_reco=None

    if (dfA_seg is not None) and (dfB_seg is not None) and (tA and tB) and (distA and distB):
        # Déterminer court/long par durée
        if tA <= tB:
            d_short_bpm, d_long_bpm = driftA_bpm, driftB_bpm
            d_short_pct, d_long_pct = driftA_pct, driftB_pct
            label_short, label_long = "Test A", "Test B"
        else:
            d_short_bpm, d_long_bpm = driftB_bpm, driftA_bpm
            d_short_pct, d_long_pct = driftB_pct, driftA_pct
            label_short, label_long = "Test B", "Test A"

        # VC 2 points
        D1, T1 = float(distA), float(tA)
        D2, T2 = float(distB), float(tB)
        if (T2 != T1) and all(v>0 for v in [D1,D2,T1,T2]):
            CS = (D2 - D1) / (T2 - T1)
            D_prime = D1 - CS*T1
            V_kmh = CS*3.6
            pace_str = format_pace_from_kmh(V_kmh)
            vc_dict = {"CS": CS, "V_kmh": V_kmh, "D_prime": D_prime, "pace_str": pace_str}

        # IC
        IC_value, IC_unite, IC_msg, IC_level, IC_reco = compute_index_cinetique(
            d_short_pct, d_long_pct, d_short_bpm, d_long_bpm
        )

        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("🧾 Synthèse")

        kpi = st.columns(4)
        if vc_dict:
            kpi[0].metric("CS (m/s)", f"{vc_dict['CS']:.2f}")
            kpi[1].metric("VC (km/h)", f"{vc_dict['V_kmh']:.2f}")
            kpi[2].metric("Allure VC", vc_dict["pace_str"])
            kpi[3].metric("D′ (m)", f"{vc_dict['D_prime']:.0f}")
        else:
            st.warning("VC non calculable (vérifie distances/temps).")

        c2 = st.columns(2)
        with c2[0]:
            st.markdown("**Dates**")
            st.write(f"- Test A : {date_A}")
            st.write(f"- Test B : {date_B}")
        with c2[1]:
            st.markdown("**Index de cinétique**")
            if IC_value is not None:
                st.write(f"IC = **{IC_value:.3f}** *(court = {label_short}, long = {label_long}, unité = {IC_unite})*")
                st.caption(IC_msg)
            else:
                st.write("— Non calculable —")

        # Graph comparatif FC
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("📈 Comparatif FC")
        if (dfA_seg is not None) and (dfB_seg is not None):
            figC, axC = plt.subplots(figsize=(9,3.2))
            if "heart_rate_smooth" in dfA_seg.columns:
                axC.plot(dfA_seg["time_s"]-dfA_seg["time_s"].iloc[0], dfA_seg["heart_rate_smooth"], color=COLOR_RED, label=f"Test A ({date_A})")
            if "heart_rate_smooth" in dfB_seg.columns:
                axC.plot(dfB_seg["time_s"]-dfB_seg["time_s"].iloc[0], dfB_seg["heart_rate_smooth"], color=COLOR_BLACK, label=f"Test B ({date_B})")
            axC.set_xlabel("Temps (s)"); axC.set_ylabel("FC (bpm)"); axC.grid(alpha=0.25); axC.legend()
            st.pyplot(figC)
            st.download_button("💾 PNG Comparatif", data=fig_to_png_bytes(figC), file_name="comparatif_fc.png", mime="image/png")
            plt.close(figC)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("🖼️ Rapport PNG (qualité élevée)")
        # Construire des listes de lignes résumées pour Test A/B
        statsA_lines = []
        statsB_lines = []
        if (distA is not None) and (tA is not None) and tA>0:
            v1 = (distA/tA)*3.6
            statsA_lines += [f"Durée: {tA:.1f}s", f"Distance: {distA:.1f} m", f"Vitesse: {v1:.2f} km/h ({format_pace_from_kmh(v1)})"]
        if driftA_bpm is not None: statsA_lines.append(f"Dérive FC: {driftA_bpm:.4f} bpm/min")
        if driftA_pct is not None: statsA_lines.append(f"Dérive FC: {driftA_pct:.4f} %/min")
        if (distB is not None) and (tB is not None) and tB>0:
            v2 = (distB/tB)*3.6
            statsB_lines += [f"Durée: {tB:.1f}s", f"Distance: {distB:.1f} m", f"Vitesse: {v2:.2f} km/h ({format_pace_from_kmh(v2)})"]
        if driftB_bpm is not None: statsB_lines.append(f"Dérive FC: {driftB_bpm:.4f} bpm/min")
        if driftB_pct is not None: statsB_lines.append(f"Dérive FC: {driftB_pct:.4f} %/min")

        png = render_full_report_png(
            title="Rapport complet – Endurance & VC (Rouge/Noir/Blanc)",
            date1=date_A, date2=date_B,
            df1_seg=dfA_seg, t0_1=startA, stats1=statsA_lines, dist1_m=distA, t1_s=tA,
            df2_seg=dfB_seg, t0_2=startB, stats2=statsB_lines, dist2_m=distB, t2_s=tB,
            vc_dict=vc_dict, IC_value=IC_value, IC_unite=IC_unite, IC_msg=IC_msg, IC_reco=IC_reco
        )
        st.download_button("💾 Télécharger le RAPPORT COMPLET (PNG)", data=png, file_name="rapport_complet_endurance_vc.png", mime="image/png")

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Importe et paramètre d’abord Test A et Test B dans l’onglet 1 pour activer la synthèse.")
