# ============================
# 🏃‍♂️ Analyse Endurance + VC + Index de cinétique + Rapport PNG
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
st.set_page_config(page_title="Analyse Endurance + VC", layout="wide")

# Palette rouge/noir/blanc
COLOR_RED = "#d21f3c"
COLOR_BLACK = "#111111"
COLOR_WHITE = "#ffffff"
COLOR_GREY = "#6b7280"
BG_PANEL = "#fafafa"

st.markdown(f"""
<style>
.report-card {{
  padding: 1rem 1.2rem;
  border-radius: 14px;
  border: 1px solid rgba(0,0,0,0.08);
  background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}}
.subtle {{ color: #6b7280; font-size: 0.92rem; }}
.section-title {{ margin-top: .6rem; margin-bottom: .4rem; }}
hr {{ border: none; border-top: 1px solid #eee; margin: 1.2rem 0; }}
.block-container {{ padding-top: 1.4rem; }}
</style>
""", unsafe_allow_html=True)

# =============== UTILS ======================

def load_activity(file):
    """Charge un fichier FIT, GPX ou CSV (robuste avec fitdecode)."""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)

    elif file.name.endswith(".fit"):
        data = []
        try:
            with fitdecode.FitReader(file) as fit:
                for frame in fit:
                    if isinstance(frame, fitdecode.records.FitDataMessage) and frame.name == "record":
                        record_data = {f.name: f.value for f in frame.fields}
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
                        "time": point.time,
                        "lat": point.latitude,
                        "lon": point.longitude,
                        "alt": point.elevation
                    })
        df = pd.DataFrame(data)
    else:
        raise ValueError("Format non supporté.")

    # Harmoniser la colonne temps -> 'timestamp'
    for c in df.columns:
        if "time" in c.lower():
            df.rename(columns={c: "timestamp"}, inplace=True)
            break

    # Validation minimale
    if "heart_rate" not in df.columns:
        raise ValueError("Le fichier ne contient pas de fréquence cardiaque ('heart_rate').")

    df = df.dropna(subset=["heart_rate"]).reset_index(drop=True)
    return df


def smooth_hr(df, time_col="timestamp", hr_col="heart_rate"):
    """Temps continu (ignore les pauses) + lissage FC."""
    df = df.copy().sort_values(by=time_col).reset_index(drop=True)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # Δt et temps continu
    df["delta_t"] = df[time_col].diff().dt.total_seconds().fillna(0)
    median_step = np.median(df["delta_t"][df["delta_t"] > 0])
    if np.isnan(median_step) or median_step == 0:
        median_step = 1
    df.loc[df["delta_t"] > 2 * median_step, "delta_t"] = median_step

    df["time_s"] = df["delta_t"].cumsum()
    total_dur = df["time_s"].iloc[-1]

    # fenêtre lissage
    if total_dur < 360:
        window_sec = 5
    elif total_dur < 900:
        window_sec = 10
    else:
        window_sec = 20

    step = np.median(np.diff(df["time_s"]))
    if step <= 0 or np.isnan(step):
        step = 1
    window_size = max(1, int(window_sec / step))

    df["hr_smooth"] = df[hr_col].rolling(window_size, min_periods=1).mean()
    pauses = (df["delta_t"] > 2 * median_step).sum()
    return df, window_sec, total_dur, pauses


def analyze_heart_rate(df):
    """Stats FC + dérive (bpm/min et %/min) via régression."""
    hr = df["hr_smooth"].dropna()
    mean_hr = hr.mean()
    max_hr = hr.max()
    min_hr = hr.min()

    slope, _, _, _, _ = linregress(df["time_s"], df["hr_smooth"])
    drift_per_min = slope * 60  # bpm/min
    drift_percent = (drift_per_min / mean_hr) * 100 if mean_hr > 0 else np.nan  # %/min

    stats = {
        "FC moyenne (bpm)": round(mean_hr, 1),
        "FC max (bpm)": round(max_hr, 1),
        "FC min (bpm)": round(min_hr, 1),
        "Dérive (bpm/min)": round(drift_per_min, 4),
        "Dérive (%/min)": round(drift_percent, 4) if not np.isnan(drift_percent) else None,
        "Durée segment (s)": round(df["time_s"].iloc[-1] - df["time_s"].iloc[0], 1),
    }
    return stats, float(drift_per_min), (None if np.isnan(drift_percent) else float(drift_percent))


def parse_time_to_seconds(tstr: str) -> int:
    """hh:mm:ss | mm:ss | ss | nombre avec virgule/point => secondes int."""
    tstr = tstr.strip()
    if ":" not in tstr:
        try:
            val = float(tstr.replace(",", "."))
            return int(round(val))
        except:
            raise ValueError("Format temps invalide")
    parts = [int(p) for p in tstr.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = 0, 0, parts[0]
    return int(h * 3600 + m * 60 + s)


def haversine_dist_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def segment_distance_m(df_seg):
    """Distance prioritaire: distance FIT (m) > speed*Δt > Haversine lat/lon."""
    if df_seg is None or df_seg.empty or len(df_seg) < 2:
        return 0.0

    # 1) distance cumulative
    for cname in df_seg.columns:
        if cname.lower() == "distance":
            d0 = float(df_seg[cname].iloc[0])
            d1 = float(df_seg[cname].iloc[-1])
            if np.isfinite(d0) and np.isfinite(d1):
                return max(0.0, d1 - d0)

    # 2) speed * delta_t
    speed_col = next((c for c in df_seg.columns if c.lower() == "speed"), None)
    if speed_col is not None and "delta_t" in df_seg.columns:
        dist = float(np.nansum(df_seg[speed_col].fillna(0).values * df_seg["delta_t"].fillna(0).values))
        if dist > 0:
            return dist

    # 3) Haversine
    lc = {c.lower(): c for c in df_seg.columns}
    if "lat" in lc and "lon" in lc:
        lats = df_seg[lc["lat"]].astype(float).values
        lons = df_seg[lc["lon"]].astype(float).values
        dist = 0.0
        for i in range(1, len(df_seg)):
            if all(np.isfinite([lats[i-1], lats[i], lons[i-1], lons[i]])):
                dist += haversine_dist_m(lats[i-1], lons[i-1], lats[i], lons[i])
        if dist > 0:
            return dist

    return 0.0


def format_pace_min_per_km(v_kmh):
    if v_kmh <= 0 or not math.isfinite(v_kmh):
        return None
    min_per_km = 60.0 / v_kmh
    total_seconds = int(round(min_per_km * 60.0))
    return total_seconds // 60, total_seconds % 60, min_per_km


def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


# ====== Interprétation Index de cinétique (barème & recommandations) ======

def compute_index_cinetique(drift_short_pct, drift_long_pct, drift_short_bpm, drift_long_bpm):
    """
    Retourne:
      - IC (float)
      - unite ('%/min' ou 'bpm/min')
      - message (texte court)
      - niveau ('tres_bon', 'bon', 'moyen', 'faible', 'degrade')
      - reco (dict) -> 'titre', 'points' (liste de puces), 'seances' (liste)
    On privilégie les dérives en %/min si disponibles.
    """
    use_pct = (drift_short_pct is not None and drift_long_pct is not None and drift_short_pct != 0)
    if use_pct:
        IC = 1.0 - (drift_long_pct / drift_short_pct)
        unite = "%/min"
        d_short, d_long = drift_short_pct, drift_long_pct
    else:
        if drift_short_bpm is None or drift_long_bpm is None or drift_short_bpm == 0:
            return None, None, "Index non calculable (dérives indisponibles).", None, None
        IC = 1.0 - (drift_long_bpm / drift_short_bpm)
        unite = "bpm/min"
        d_short, d_long = drift_short_bpm, drift_long_bpm

    # Classification
    if IC >= 0.70:
        niveau = "tres_bon"
        titre = "Très bonne stabilité sur le long"
        points = [
            f"Dérive {unite} courte: {d_short:.3f}",
            f"Dérive {unite} longue: {d_long:.3f}",
            "Profil endurant fort, tolérance aux blocs prolongés."
        ]
        seances = [
            "2–3×(8–12′) à 88–92% VC, r=2–3′",
            "Tempo 20–30′ à 85–90% VC",
            "Progressif 30–40′ de 80→90% VC",
            "Z2 volumineux"
        ]
        msg = "IC élevé : orientation blocs longs & tempos ambitieux."
    elif 0.40 <= IC < 0.70:
        niveau = "bon"
        titre = "Bon équilibre, marge en soutien aérobie"
        points = [
            f"Dérive {unite} courte: {d_short:.3f}",
            f"Dérive {unite} longue: {d_long:.3f}",
            "Équilibre intéressant, consolider le tempo/seuil."
        ]
        seances = [
            "4–6×5′ à 90–92% VC, r=1–2′ (cruise)",
            "2×12–15′ à 85–90% VC (tempo)",
            "6–8×(2′ @95% VC / 1′ @80%) (mix)"
        ]
        msg = "IC bon : mix intervals moyens + tempo."
    elif 0.15 <= IC < 0.40:
        niveau = "moyen"
        titre = "Stabilité limitée sur le long"
        points = [
            f"Dérive {unite} courte: {d_short:.3f}",
            f"Dérive {unite} longue: {d_long:.3f}",
            "Priorité à étendre la tolérance au long, lisser la cinétique."
        ]
        seances = [
            "3–4×6′ à 88–90% VC, r=2′",
            "3×8–10′ à 85–88% VC (tempo court)",
            "Z2 conséquent + 6–10×20″ strides"
        ]
        msg = "IC moyen : allonger progressivement les intervalles."
    elif 0.00 <= IC < 0.15:
        niveau = "faible"
        titre = "Dérives longue et courte similaires"
        points = [
            f"Dérive {unite} courte: {d_short:.3f}",
            f"Dérive {unite} longue: {d_long:.3f}",
            "Base aérobie à renforcer, démarrer par du tempo doux."
        ]
        seances = [
            "Z2 majoritaire",
            "3–4×6–8′ à 82–86% VC (tempo doux)",
            "10–12×1′ à 92–95% VC / 1′ Z2 (fartlek contrôlé)"
        ]
        msg = "IC faible : focus base + tempo doux, peu de >92% VC."
    else:  # IC < 0
        niveau = "degrade"
        titre = "Stabilité faible / contexte défavorable"
        points = [
            f"Dérive {unite} courte: {d_short:.3f}",
            f"Dérive {unite} longue: {d_long:.3f}",
            "Réduire l'intensité, reconstruire la base, vérifier conditions."
        ]
        seances = [
            "Z2 + force (côtes)",
            "Progressifs doux",
            "Limiter >90% VC ; revoir sommeil, chaleur, hydratation"
        ]
        msg = "IC négatif : re-baser et diagnostiquer (fatigue/conditions)."

    reco = {"titre": titre, "points": points, "seances": seances}
    return float(IC), unite, msg, niveau, reco


def render_full_report_png(
    title: str,
    date1, date2,
    interval_df1, start_sec1, stats1, dist1_m, t1_s,
    interval_df2, start_sec2, stats2, dist2_m, t2_s,
    vc_dict, IC_value, IC_unite, IC_msg, IC_reco
):
    """
    Un seul PNG 'rapport complet' (graph Test1, Test2, Comparatif + cartes stats + VC + Index de cinétique + prescription).
    Mise en page agrandie, sans chevauchements.
    """

    # Style matplotlib
    mpl.rcParams.update({
        "axes.edgecolor": COLOR_BLACK,
        "axes.labelcolor": COLOR_BLACK,
        "xtick.color": COLOR_BLACK,
        "ytick.color": COLOR_BLACK,
        "text.color": COLOR_BLACK,
        "figure.facecolor": COLOR_WHITE,
        "axes.facecolor": COLOR_WHITE,
        "savefig.facecolor": COLOR_WHITE,
        "font.size": 10
    })

    # Figure plus haute et plus respirante
    fig = plt.figure(figsize=(10.5, 16), dpi=170, constrained_layout=False)
    # 7 lignes : Titre, T1, T2, Comparatif, Cartes Stats (2 colonnes), VC/IC, Prescription
    gs = fig.add_gridspec(
        7, 2,
        height_ratios=[0.55, 1.15, 1.15, 1.15, 1.0, 0.95, 1.10],
        width_ratios=[1, 1],
        hspace=1.0, wspace=0.7
    )

    # En-tête
    ax_title = fig.add_subplot(gs[0, :]); ax_title.axis("off")
    ax_title.add_patch(plt.Rectangle((0,0),1,1, color=BG_PANEL, transform=ax_title.transAxes))
    ax_title.text(0.02, 0.80, title, fontsize=22, fontweight="bold", color=COLOR_BLACK, va="top")
    ax_title.text(0.02, 0.50, f"Test 1 : {date1}   •   Test 2 : {date2}", fontsize=11, color=COLOR_GREY)
    ax_title.text(0.02, 0.25, "FC, Dérive, Distances, Vitesse Critique, Index de cinétique, Prescription", fontsize=10, color=COLOR_GREY)

    # Graph Test 1
    ax1 = fig.add_subplot(gs[1, :])
    ax1.set_title("Cinétique cardiaque — Test 1", fontsize=12, fontweight="bold")
    if interval_df1 is not None and len(interval_df1) > 1:
        ax1.plot(interval_df1["time_s"] - start_sec1, interval_df1["hr_smooth"], linewidth=2, color=COLOR_RED, label="FC Test 1")
    ax1.set_xlabel("Temps segment (s)"); ax1.set_ylabel("FC (bpm)"); ax1.grid(True, alpha=0.15)

    # Graph Test 2
    ax2 = fig.add_subplot(gs[2, :])
    ax2.set_title("Cinétique cardiaque — Test 2", fontsize=12, fontweight="bold")
    if interval_df2 is not None and len(interval_df2) > 1:
        ax2.plot(interval_df2["time_s"] - start_sec2, interval_df2["hr_smooth"], linewidth=2, color=COLOR_BLACK, label="FC Test 2")
    ax2.set_xlabel("Temps segment (s)"); ax2.set_ylabel("FC (bpm)"); ax2.grid(True, alpha=0.15)

    # Graph Comparatif
    ax3 = fig.add_subplot(gs[3, :])
    ax3.set_title("Comparatif des cinétiques (segments centrés en t=0)", fontsize=12, fontweight="bold")
    if interval_df1 is not None and len(interval_df1) > 1:
        ax3.plot(interval_df1["time_s"] - interval_df1["time_s"].iloc[0],
                 interval_df1["hr_smooth"], linewidth=2, color=COLOR_RED, label="Test 1")
    if interval_df2 is not None and len(interval_df2) > 1:
        ax3.plot(interval_df2["time_s"] - interval_df2["time_s"].iloc[0],
                 interval_df2["hr_smooth"], linewidth=2, color=COLOR_BLACK, label="Test 2")
    ax3.set_xlabel("Temps segment (s)"); ax3.set_ylabel("FC (bpm)"); ax3.grid(True, alpha=0.15); ax3.legend(frameon=False)

    # Cartes stats Test 1 & Test 2 (2 colonnes)
    ax4 = fig.add_subplot(gs[4, 0]); ax5 = fig.add_subplot(gs[4, 1])
    for ax in (ax4, ax5):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0,0),1,1, color=BG_PANEL, transform=ax.transAxes))

    # Carte Test 1
    ax4.text(0.03, 0.90, "Résumé Test 1", fontsize=12, fontweight="bold")
    if stats1 is not None and t1_s:
        v1_kmh = 3.6 * (dist1_m / t1_s) if t1_s > 0 else 0.0
        lines1 = [
            f"FC moy : {stats1['FC moyenne (bpm)']} bpm",
            f"FC max : {stats1['FC max (bpm)']} bpm",
            f"Dérive : {stats1['Dérive (bpm/min)']} bpm/min",
            f"Dérive : {stats1['Dérive (%/min)']} %/min" if stats1['Dérive (%/min)'] is not None else "Dérive %/min : —",
            f"Durée  : {stats1['Durée segment (s)']} s",
            f"Distance : {dist1_m:.1f} m",
            f"Vitesse moy : {v1_kmh:.2f} km/h"
        ]
        ax4.text(0.05, 0.80, "\n".join(lines1), fontsize=11, color=COLOR_BLACK)

    # Carte Test 2
    ax5.text(0.03, 0.90, "Résumé Test 2", fontsize=12, fontweight="bold")
    if stats2 is not None and t2_s:
        v2_kmh = 3.6 * (dist2_m / t2_s) if t2_s > 0 else 0.0
        lines2 = [
            f"FC moy : {stats2['FC moyenne (bpm)']} bpm",
            f"FC max : {stats2['FC max (bpm)']} bpm",
            f"Dérive : {stats2['Dérive (bpm/min)']} bpm/min",
            f"Dérive : {stats2['Dérive (%/min)']} %/min" if stats2['Dérive (%/min)'] is not None else "Dérive %/min : —",
            f"Durée  : {stats2['Durée segment (s)']} s",
            f"Distance : {dist2_m:.1f} m",
            f"Vitesse moy : {v2_kmh:.2f} km/h"
        ]
        ax5.text(0.05, 0.80, "\n".join(lines2), fontsize=11, color=COLOR_BLACK)

    # Carte VC + Index de cinétique (2 colonnes)
    ax6L = fig.add_subplot(gs[5, 0]); ax6R = fig.add_subplot(gs[5, 1])
    for ax in (ax6L, ax6R):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0,0),1,1, color=BG_PANEL, transform=ax.transAxes))

    # VC
    ax6L.text(0.03, 0.90, "Vitesse Critique", fontsize=12, fontweight="bold")
    if vc_dict is not None:
        lines_vc = [
            f"CS : {vc_dict['CS']:.2f} m/s",
            f"VC : {vc_dict['V_kmh']:.2f} km/h",
            f"Allure VC : {vc_dict['pace_str']}",
            f"D′ : {vc_dict['D_prime']:.0f} m",
        ]
        ax6L.text(0.05, 0.80, "\n".join(lines_vc), fontsize=11, color=COLOR_BLACK)
    else:
        ax6L.text(0.05, 0.80, "— Non calculable —", fontsize=11, color=COLOR_GREY)

    # Index de cinétique
    ax6R.text(0.03, 0.90, "Index de cinétique (IC)", fontsize=12, fontweight="bold")
    if IC_value is not None:
        lines_ic = [
            f"IC : {IC_value:.3f}",
            f"Unité dérives : {IC_unite}",
            f"Note : {IC_msg}",
        ]
        ax6R.text(0.05, 0.80, "\n".join(lines_ic), fontsize=11, color=COLOR_BLACK)
    else:
        ax6R.text(0.05, 0.80, "— Non calculable —", fontsize=11, color=COLOR_GREY)

    # Prescription (bandeau bas)
    ax7 = fig.add_subplot(gs[6, :]); ax7.axis("off")
    ax7.add_patch(plt.Rectangle((0,0),1,1, color=BG_PANEL, transform=ax7.transAxes))
    ax7.text(0.03, 0.90, "Prescription (4–8 semaines)", fontsize=12, fontweight="bold")
    if IC_reco is not None:
        # Puces courtes pour éviter chevauchement
        y = 0.78
        ax7.text(0.04, y, f"• {IC_reco['titre']}", fontsize=11, color=COLOR_BLACK); y -= 0.08
        for p in IC_reco["points"]:
            ax7.text(0.05, y, f"– {p}", fontsize=10, color=COLOR_BLACK); y -= 0.06
        y -= 0.04
        ax7.text(0.04, y, "Séances types :", fontsize=11, color=COLOR_BLACK); y -= 0.08
        for s in IC_reco["seances"]:
            ax7.text(0.05, y, f"• {s}", fontsize=10, color=COLOR_BLACK); y -= 0.06
    else:
        ax7.text(0.04, 0.78, "— Aucune recommandation (IC indisponible) —", fontsize=11, color=COLOR_GREY)

    # Bande rouge pied de page
    ax7.add_patch(plt.Rectangle((0, -0.06), 1, 0.06, color=COLOR_RED, transform=ax7.transAxes))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# =============== APP ========================

st.title("🏃‍♂️ Analyse de Tests d'Endurance + Vitesse Critique")

tabs = st.tabs(["Test 1", "Test 2", "Analyse générale"])

# Variables partagées
interval_df1 = stats1 = None
interval_df2 = stats2 = None
drift1_bpm = drift2_bpm = None
drift1_pct = drift2_pct = None
dist1_m = dist2_m = None
t1_s = t2_s = None
test1_date = test2_date = None
start_sec1 = start_sec2 = 0

# ---------- Onglet Test 1 ----------
with tabs[0]:
    st.header("🧪 Test 1")
    coltop = st.columns([2,1])
    with coltop[0]:
        uploaded_file1 = st.file_uploader("Importe le premier test (FIT, GPX ou CSV)", type=["fit", "gpx", "csv"], key="file1")
    with coltop[1]:
        test1_date = st.date_input("📅 Date du test 1", value=date.today(), key="date1")

    if uploaded_file1:
        try:
            df1 = load_activity(uploaded_file1)
        except Exception as e:
            st.error(f"Erreur fichier 1 : {e}")
            st.stop()

        df1["timestamp"] = pd.to_datetime(df1["timestamp"], errors="coerce")
        df1 = df1.dropna(subset=["timestamp"])

        lag1 = st.slider("Correction du décalage capteur (s)", 0, 10, 0, key="lag1")
        df1["timestamp"] = df1["timestamp"] - pd.to_timedelta(lag1, unit="s")

        df1, window_sec1, total_dur1, pauses1 = smooth_hr(df1)
        st.markdown(f"""
        <div class="report-card">
          <div class="subtle">Durée détectée : {total_dur1:.1f}s • Lissage : {window_sec1}s • Pauses : {pauses1}</div>
          <h4 class="section-title">🎯 Sélection du segment (format hh:mm:ss)</h4>
        </div>
        """, unsafe_allow_html=True)

        c11, c12 = st.columns(2)
        with c11:
            start_str1 = st.text_input("Début", value="0:00:00", key="start1")
        with c12:
            end_str1 = st.text_input("Fin", value="0:12:00", key="end1")

        try:
            start_sec1 = parse_time_to_seconds(start_str1)
            end_sec1 = parse_time_to_seconds(end_str1)
        except:
            st.error("Format temps invalide (hh:mm:ss).")
            st.stop()

        if end_sec1 <= start_sec1:
            st.error("La fin doit être supérieure au début.")
            st.stop()

        if end_sec1 > df1["time_s"].max():
            st.warning("⚠️ Fin > données disponibles. Limitation automatique (Test 1).")
            end_sec1 = df1["time_s"].max()

        interval_df1 = df1[(df1["time_s"] >= start_sec1) & (df1["time_s"] <= end_sec1)]

        if len(interval_df1) > 10:
            stats1, drift1_bpm, drift1_pct = analyze_heart_rate(interval_df1)
            dist1_m = segment_distance_m(interval_df1)
            t1_s = float(end_sec1 - start_sec1)
            v1_kmh = 3.6 * (dist1_m / t1_s) if t1_s > 0 else 0.0

            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.subheader(f"📊 Résultats ({start_str1} → {end_str1}) — Test 1")
            cols = st.columns(4)
            cols[0].metric("FC moyenne", f"{stats1['FC moyenne (bpm)']} bpm")
            cols[1].metric("FC max", f"{stats1['FC max (bpm)']} bpm")
            cols[2].metric("Dérive", f"{stats1['Dérive (bpm/min)']} bpm/min")
            cols[3].metric("Durée", f"{stats1['Durée segment (s)']} s")

            cols2 = st.columns(3)
            cols2[0].metric("Distance segment", f"{dist1_m:.1f} m")
            cols2[1].metric("Temps segment", f"{t1_s:.1f} s")
            cols2[2].metric("Vitesse moy.", f"{v1_kmh:.2f} km/h")

            fig1, ax1 = plt.subplots()
            ax1.plot(interval_df1["time_s"] - start_sec1, interval_df1["hr_smooth"], label="FC Test 1", color=COLOR_RED)
            ax1.set_xlabel("Temps se_
