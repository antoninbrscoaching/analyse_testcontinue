# ============================
# 🏃‍♂️ Analyse Endurance (Tests fusionnés) + VC + Index de cinétique
# + Entraînement (multi-séances + IC par intervalle)
# + Superposition FC / Allure / Puissance (double axe Y)
# + Export PDF (uniquement)
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress
import fitdecode
import math
from io import BytesIO
from datetime import date
import matplotlib as mpl
import xml.etree.ElementTree as ET  # ✅ lecture TCX
import gpxpy

# =============== CONFIG / THEME =================
st.set_page_config(page_title="Analyse Endurance + VC (PDF)", layout="wide")

# Couleurs principales
COLOR_RED_T1 = "#d21f3c"
COLOR_RED_T2 = "#8b0a1a"
COLOR_RED_SES = "#f57c92"
COLOR_BLUE_T1 = "#0066cc"
COLOR_BLUE_T2 = "#003366"
COLOR_BLUE_SES = "#66a3ff"
COLOR_ORANGE_T1 = "#ff8c00"
COLOR_ORANGE_T2 = "#cc6600"
COLOR_ORANGE_SES = "#ffb84d"
COLOR_GREY = "#6b7280"

st.markdown("""
<style>
.report-card {
  padding: 1rem 1.2rem;
  border-radius: 14px;
  border: 1px solid rgba(0,0,0,0.08);
  background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
  margin-bottom: 0.8rem;
}
.subtle { color: #6b7280; font-size: 0.92rem; }
.section-title { margin-top: .6rem; margin-bottom: .4rem; }
hr { border: none; border-top: 1px solid #eee; margin: 1.2rem 0; }
.block-container { padding-top: 1.4rem; }
.table-box {
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 10px;
  padding: 0.4rem 0.6rem;
  background: #fff;
  box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)


# =============== EXTENSIONS ACCEPTÉES ======================
ACCEPTED_TYPES = ["fit", "FIT", "gpx", "GPX", "csv", "CSV", "tcx", "TCX"]

# =============== LECTURE FICHIERS ==========================
def load_activity(file):
    """Charge un fichier FIT, GPX, CSV ou TCX."""
    name = file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(file)

    elif name.endswith(".fit"):
        data = []
        try:
            with fitdecode.FitReader(file) as fit:
                for frame in fit:
                    if isinstance(frame, fitdecode.records.FitDataMessage) and frame.name == "record":
                        data.append({f.name: f.value for f in frame.fields})
            df = pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Erreur lecture FIT : {e}")

    elif name.endswith(".gpx"):
        gpx = gpxpy.parse(file)
        data = []
        for track in gpx.tracks:
            for seg in track.segments:
                for p in seg.points:
                    data.append({
                        "time": p.time,
                        "lat": p.latitude,
                        "lon": p.longitude,
                        "alt": p.elevation
                    })
        df = pd.DataFrame(data)

    elif name.endswith(".tcx"):
        # ✅ Lecture TCX
        try:
            tree = ET.parse(file)
            root = tree.getroot()
            ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
            data = []
            for tp in root.findall(".//tcx:Trackpoint", ns):
                time = tp.find("tcx:Time", ns)
                hr = tp.find("tcx:HeartRateBpm/tcx:Value", ns)
                dist = tp.find("tcx:DistanceMeters", ns)
                alt = tp.find("tcx:AltitudeMeters", ns)
                power = tp.find("tcx:Extensions//tcx:Watts", ns)
                pos = tp.find("tcx:Position", ns)
                lat = pos.find("tcx:LatitudeDegrees", ns).text if pos is not None and pos.find("tcx:LatitudeDegrees", ns) is not None else None
                lon = pos.find("tcx:LongitudeDegrees", ns).text if pos is not None and pos.find("tcx:LongitudeDegrees", ns) is not None else None

                data.append({
                    "timestamp": time.text if time is not None else None,
                    "heart_rate": float(hr.text) if hr is not None else None,
                    "distance": float(dist.text) if dist is not None else None,
                    "alt": float(alt.text) if alt is not None else None,
                    "power": float(power.text) if power is not None else None,
                    "lat": float(lat) if lat else None,
                    "lon": float(lon) if lon else None
                })
            df = pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Erreur lecture TCX : {e}")

    else:
        raise ValueError("Format non supporté (.fit, .gpx, .csv, .tcx uniquement)")

    # Harmonisation de la colonne temps
    for c in df.columns:
        if "time" in c.lower():
            df.rename(columns={c: "timestamp"}, inplace=True)
            break

    if "heart_rate" not in df.columns:
        raise ValueError("Aucune donnée de fréquence cardiaque détectée")

    df = df.dropna(subset=["heart_rate"]).reset_index(drop=True)

    for c in ["heart_rate", "speed", "enhanced_speed", "power", "distance", "lat", "lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
# =============== UTILITAIRES CALCUL / LISSAGE ======================

def get_speed_col(df):
    """Retourne le nom de la colonne de vitesse (m/s) si disponible."""
    if "enhanced_speed" in df.columns:
        return "enhanced_speed"
    if "speed" in df.columns:
        return "speed"
    return None


def smooth_hr(df, time_col="timestamp", hr_col="heart_rate"):
    """Lissage FC et signaux associés, gère les pauses."""
    df = df.copy().sort_values(by=time_col).reset_index(drop=True)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # Delta temps
    df["delta_t"] = df[time_col].diff().dt.total_seconds().fillna(0)
    median_step = np.median(df["delta_t"][df["delta_t"] > 0])
    if np.isnan(median_step) or median_step == 0:
        median_step = 1
    df.loc[df["delta_t"] > 2 * median_step, "delta_t"] = median_step

    df["time_s"] = df["delta_t"].cumsum()
    total_dur = df["time_s"].iloc[-1]

    # Fenêtre de lissage selon durée
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
    speed_col = get_speed_col(df)
    if speed_col:
        df["speed_smooth"] = df[speed_col].rolling(window_size, min_periods=1).mean()
    if "power" in df.columns:
        df["power_smooth"] = df["power"].rolling(window_size, min_periods=1).mean()

    pauses = (df["delta_t"] > 2 * median_step).sum()
    return df, window_sec, total_dur, pauses


def analyze_heart_rate(df):
    """Analyse de la FC : moyenne, dérive bpm/min et %/min."""
    hr = df["hr_smooth"].dropna()
    mean_hr = hr.mean()
    max_hr = hr.max()
    min_hr = hr.min()

    slope, _, _, _, _ = linregress(df["time_s"], df["hr_smooth"])
    drift_per_min = slope * 60
    drift_percent = (drift_per_min / mean_hr) * 100 if mean_hr > 0 else np.nan

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
    """Convertit hh:mm:ss | mm:ss | ss en secondes."""
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


# =============== DISTANCE & GÉO ======================

def haversine_dist_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def segment_distance_m(df_seg):
    """Calcule la distance du segment, priorise distance FIT > speed*dt > Haversine."""
    if df_seg is None or df_seg.empty or len(df_seg) < 2:
        return 0.0

    for cname in df_seg.columns:
        if cname.lower() == "distance":
            d0 = float(df_seg[cname].iloc[0])
            d1 = float(df_seg[cname].iloc[-1])
            if np.isfinite(d0) and np.isfinite(d1):
                return max(0.0, d1 - d0)

    speed_col = next((c for c in df_seg.columns if c.lower() in ("speed", "enhanced_speed")), None)
    if speed_col is not None and "delta_t" in df_seg.columns:
        dist = float(np.nansum(df_seg[speed_col].fillna(0).values * df_seg["delta_t"].fillna(0).values))
        if dist > 0:
            return dist

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


# =============== FORMATTAGE ALLURE / VITESSE ======================

def format_pace_min_per_km(v_kmh):
    if v_kmh <= 0 or not math.isfinite(v_kmh):
        return None
    min_per_km = 60.0 / v_kmh
    total_seconds = int(round(min_per_km * 60.0))
    return total_seconds // 60, total_seconds % 60, min_per_km


# =============== EXPORT PDF ======================

def fig_to_pdf_bytes(figs):
    """Convertit un ou plusieurs graphs matplotlib en PDF en mémoire."""
    if not isinstance(figs, (list, tuple)):
        figs = [figs]
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        for f in figs:
            f.tight_layout()
            pdf.savefig(f, bbox_inches="tight")
    buf.seek(0)
    return buf


# =============== INDEX CINÉTIQUE ======================

def compute_index_cinetique(drift_short_pct, drift_long_pct, drift_short_bpm, drift_long_bpm):
    """Calcule l'index de cinétique (IC)."""
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

    if IC >= 0.70:
        titre = "Très bonne stabilité sur le long"
        msg = "IC élevé : blocs longs & tempos ambitieux."
        seances = ["2–3×(8–12′) à 88–92% VC, r=2–3′", "Tempo 20–30′ à 85–90% VC", "Progressif 30–40′ de 80→90% VC", "Z2 volumineux"]
    elif 0.40 <= IC < 0.70:
        titre = "Bon équilibre, marge en soutien aérobie"
        msg = "IC bon : mix intervals moyens + tempo."
        seances = ["4–6×5′ à 90–92% VC, r=1–2′", "2×12–15′ à 85–90% VC", "6–8×(2′ @95% VC / 1′ @80%)"]
    elif 0.15 <= IC < 0.40:
        titre = "Stabilité limitée sur le long"
        msg = "IC moyen : allonger progressivement les intervalles."
        seances = ["3–4×6′ à 88–90% VC", "3×8–10′ à 85–88% VC", "Z2 + 6–10×20″ strides"]
    elif 0.00 <= IC < 0.15:
        titre = "Dérives longue et courte similaires"
        msg = "IC faible : base + tempo doux, peu de >92% VC."
        seances = ["Z2 majoritaire", "3–4×6–8′ à 82–86% VC", "10–12×1′ à 92–95% VC / 1′ Z2"]
    else:
        titre = "Stabilité faible / contexte défavorable"
        msg = "IC négatif : re-baser et diagnostiquer (fatigue/conditions)."
        seances = ["Z2 + force (côtes)", "Progressifs doux", "Limiter >90% VC ; vérifier récupération"]

    reco = {"titre": titre, "seances": seances}
    return float(IC), unite, msg, None, reco


# =============== AIDES GRAPHIQUES ======================

def pace_formatter(v, pos):
    if v is None or not math.isfinite(v) or v <= 0:
        return ""
    m = int(v)
    s = int(round((v - m) * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d}"


def add_pace_axis(ax):
    ax_pace = ax.twinx()
    ax_pace.set_ylabel("Allure (min/km)")
    ax_pace.yaxis.set_major_formatter(FuncFormatter(pace_formatter))
    ax_pace.invert_yaxis()
    return ax_pace


def add_power_axis(ax, offset=60):
    ax_pow = ax.twinx()
    ax_pow.spines["right"].set_position(("outward", offset))
    ax_pow.set_frame_on(True)
    ax_pow.patch.set_visible(False)
    ax_pow.set_ylabel("Puissance (W)")
    return ax_pow


def compute_pace_series(df):
    if "speed_smooth" not in df.columns:
        return None
    speed = df["speed_smooth"].astype(float).replace([np.inf, -np.inf], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pace_min_per_km = 1000.0 / speed / 60.0
    pace_min_per_km[~np.isfinite(pace_min_per_km)] = np.nan
    return pace_min_per_km


def plot_multi_signals(ax, df, t0=0.0, who="T1",
                       show_fc=True, show_pace=False, show_power=False,
                       linewidth=1.8):
    """Trace FC, Allure, Puissance sur des axes séparés selon le type (T1, T2, SES)."""
    if who == "T1":
        c_fc, c_pace, c_pow = "#d21f3c", "#0066cc", "#ff8c00"
    elif who == "T2":
        c_fc, c_pace, c_pow = "#8b0a1a", "#003366", "#cc6600"
    else:
        c_fc, c_pace, c_pow = "#f57c92", "#66a3ff", "#ffb84d"

    ax_pace = None
    ax_pow = None
    tt = df["time_s"].values - t0

    if show_fc and "hr_smooth" in df.columns:
        ax.plot(tt, df["hr_smooth"], color=c_fc, linewidth=linewidth, label=f"{who} • FC (bpm)")
        ax.set_ylabel("FC (bpm)")

    if show_pace and "speed_smooth" in df.columns:
        pace_series = compute_pace_series(df)
        if pace_series is not None:
            ax_pace = add_pace_axis(ax)
            ax_pace.plot(tt, pace_series, color=c_pace, linewidth=linewidth, label=f"{who} • Allure (min/km)")

    if show_power and "power_smooth" in df.columns:
        ax_pow = add_power_axis(ax, offset=60)
        ax_pow.plot(tt, df["power_smooth"], color=c_pow, linewidth=linewidth, label=f"{who} • Puissance (W)")

    return ax, ax_pace, ax_pow
# =============== APP PRINCIPALE ========================
st.title("🏃‍♂️ Analyse de Tests d'Endurance + Vitesse Critique (Export PDF)")

tabs = st.tabs(["🧪 Tests d'endurance", "⚙️ Analyse entraînement", "📊 Analyse générale"])

# ---------- Variables globales ----------
interval_df1 = interval_df2 = None
stats1 = stats2 = None
drift1_bpm = drift2_bpm = None
drift1_pct = drift2_pct = None
dist1_m = dist2_m = None
t1_s = t2_s = None
test1_date = test2_date = None
start_sec1 = start_sec2 = 0

# ---------- Onglet 1 : Tests d'endurance ----------
with tabs[0]:
    st.header("🧪 Tests d'endurance")

    ctop = st.columns(2)

    # ---- Test 1 ----
    with ctop[0]:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("Test 1")
        uploaded_file1 = st.file_uploader(
            "Fichier Test 1 (FIT, GPX, CSV, TCX)",
            type=ACCEPTED_TYPES,
            key="file1"
        )
        test1_date = st.date_input("📅 Date du test 1", value=date.today(), key="date1")

        show_t1_fc = st.checkbox("☑️ FC (Test 1)", value=True, key="t1_fc")
        show_t1_pace = st.checkbox("☑️ Allure (Test 1)", value=False, key="t1_pace")
        show_t1_power = st.checkbox("☑️ Puissance (Test 1)", value=False, key="t1_power")

        if uploaded_file1:
            try:
                df1 = load_activity(uploaded_file1)
            except Exception as e:
                st.error(f"Erreur fichier 1 : {e}")
                st.markdown('</div>', unsafe_allow_html=True)
                st.stop()

            df1["timestamp"] = pd.to_datetime(df1["timestamp"], errors="coerce")
            df1 = df1.dropna(subset=["timestamp"])
            lag1 = st.slider("Correction du décalage capteur (s)", 0, 10, 0, key="lag1")
            df1["timestamp"] = df1["timestamp"] - pd.to_timedelta(lag1, unit="s")

            df1, window_sec1, total_dur1, pauses1 = smooth_hr(df1)
            st.caption(f"Durée détectée : {total_dur1:.1f}s • Lissage : {window_sec1}s • Pauses : {pauses1}")

            c11, c12 = st.columns(2)
            with c11:
                start_str1 = st.text_input("Début (hh:mm:ss)", value="0:00:00", key="start1")
            with c12:
                end_str1 = st.text_input("Fin (hh:mm:ss)", value="0:12:00", key="end1")

            try:
                start_sec1 = parse_time_to_seconds(start_str1)
                end_sec1 = parse_time_to_seconds(end_str1)
            except:
                st.error("Format temps invalide (hh:mm:ss).")
                st.stop()

            if end_sec1 <= start_sec1:
                st.error("La fin doit être supérieure au début.")
            else:
                if end_sec1 > df1["time_s"].max():
                    st.warning("⚠️ Fin > données disponibles – limitation automatique (Test 1).")
                    end_sec1 = df1["time_s"].max()

                interval_df1 = df1[(df1["time_s"] >= start_sec1) & (df1["time_s"] <= end_sec1)]
                if len(interval_df1) > 10:
                    stats1, drift1_bpm, drift1_pct = analyze_heart_rate(interval_df1)
                    dist1_m = segment_distance_m(interval_df1)
                    t1_s = float(end_sec1 - start_sec1)
                    v1_kmh = 3.6 * (dist1_m / t1_s) if t1_s > 0 else 0.0

                    table1 = pd.DataFrame({
                        "Métrique": [
                            "FC moyenne (bpm)", "FC max (bpm)",
                            "Dérive (bpm/min)", "Dérive (%/min)",
                            "Durée (s)", "Distance (m)", "Vitesse moy (km/h)"
                        ],
                        "Valeur": [
                            stats1["FC moyenne (bpm)"], stats1["FC max (bpm)"],
                            stats1["Dérive (bpm/min)"], stats1["Dérive (%/min)"],
                            stats1["Durée segment (s)"], round(dist1_m, 1), round(v1_kmh, 2)
                        ]
                    })
                    st.markdown('<div class="table-box">', unsafe_allow_html=True)
                    st.dataframe(table1, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    fig1, ax1 = plt.subplots(figsize=(9, 4.8))
                    plot_multi_signals(
                        ax1, interval_df1, t0=start_sec1, who="T1",
                        show_fc=show_t1_fc,
                        show_pace=show_t1_pace and (get_speed_col(interval_df1) is not None),
                        show_power=show_t1_power and ("power_smooth" in interval_df1.columns),
                        linewidth=1.9
                    )
                    ax1.set_xlabel("Temps segment (s)")
                    ax1.set_title(f"Cinétique – Test 1 ({test1_date})")
                    ax1.grid(True, alpha=0.15)

                    handles, labels = [], []
                    for a in fig1.axes:
                        h, l = a.get_legend_handles_labels()
                        handles += h; labels += l
                    if handles:
                        ax1.legend(handles, labels, fontsize=8, loc="upper left", frameon=False)
                    st.pyplot(fig1)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Test 2 ----
    with ctop[1]:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("Test 2")
        uploaded_file2 = st.file_uploader(
            "Fichier Test 2 (FIT, GPX, CSV, TCX)",
            type=ACCEPTED_TYPES,
            key="file2"
        )
        test2_date = st.date_input("📅 Date du test 2", value=date.today(), key="date2")

        show_t2_fc = st.checkbox("☑️ FC (Test 2)", value=True, key="t2_fc")
        show_t2_pace = st.checkbox("☑️ Allure (Test 2)", value=False, key="t2_pace")
        show_t2_power = st.checkbox("☑️ Puissance (Test 2)", value=False, key="t2_power")

        if uploaded_file2:
            try:
                df2 = load_activity(uploaded_file2)
            except Exception as e:
                st.error(f"Erreur fichier 2 : {e}")
                st.stop()

            df2["timestamp"] = pd.to_datetime(df2["timestamp"], errors="coerce")
            df2 = df2.dropna(subset=["timestamp"])
            lag2 = st.slider("Correction du décalage capteur (s)", 0, 10, 0, key="lag2")
            df2["timestamp"] = df2["timestamp"] - pd.to_timedelta(lag2, unit="s")

            df2, window_sec2, total_dur2, pauses2 = smooth_hr(df2)
            st.caption(f"Durée détectée : {total_dur2:.1f}s • Lissage : {window_sec2}s • Pauses : {pauses2}")

            c21, c22 = st.columns(2)
            with c21:
                start_str2 = st.text_input("Début (hh:mm:ss)", value="0:00:00", key="start2")
            with c22:
                end_str2 = st.text_input("Fin (hh:mm:ss)", value="0:12:00", key="end2")

            try:
                start_sec2 = parse_time_to_seconds(start_str2)
                end_sec2 = parse_time_to_seconds(end_str2)
            except:
                st.error("Format temps invalide (hh:mm:ss).")
                st.stop()

            if end_sec2 <= start_sec2:
                st.error("La fin doit être supérieure au début.")
            else:
                if end_sec2 > df2["time_s"].max():
                    st.warning("⚠️ Fin > données disponibles – limitation automatique (Test 2).")
                    end_sec2 = df2["time_s"].max()

                interval_df2 = df2[(df2["time_s"] >= start_sec2) & (df2["time_s"] <= end_sec2)]
                if len(interval_df2) > 10:
                    stats2, drift2_bpm, drift2_pct = analyze_heart_rate(interval_df2)
                    dist2_m = segment_distance_m(interval_df2)
                    t2_s = float(end_sec2 - start_sec2)
                    v2_kmh = 3.6 * (dist2_m / t2_s) if t2_s > 0 else 0.0

                    table2 = pd.DataFrame({
                        "Métrique": [
                            "FC moyenne (bpm)", "FC max (bpm)",
                            "Dérive (bpm/min)", "Dérive (%/min)",
                            "Durée (s)", "Distance (m)", "Vitesse moy (km/h)"
                        ],
                        "Valeur": [
                            stats2["FC moyenne (bpm)"], stats2["FC max (bpm)"],
                            stats2["Dérive (bpm/min)"], stats2["Dérive (%/min)"],
                            stats2["Durée segment (s)"], round(dist2_m, 1), round(v2_kmh, 2)
                        ]
                    })
                    st.markdown('<div class="table-box">', unsafe_allow_html=True)
                    st.dataframe(table2, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    fig2, ax2 = plt.subplots(figsize=(9, 4.8))
                    plot_multi_signals(
                        ax2, interval_df2, t0=start_sec2, who="T2",
                        show_fc=show_t2_fc,
                        show_pace=show_t2_pace and (get_speed_col(interval_df2) is not None),
                        show_power=show_t2_power and ("power_smooth" in interval_df2.columns),
                        linewidth=1.9
                    )
                    ax2.set_xlabel("Temps segment (s)")
                    ax2.set_title(f"Cinétique – Test 2 ({test2_date})")
                    ax2.grid(True, alpha=0.15)

                    handles, labels = [], []
                    for a in fig2.axes:
                        h, l = a.get_legend_handles_labels()
                        handles += h; labels += l
                    if handles:
                        ax2.legend(handles, labels, fontsize=8, loc="upper left", frameon=False)
                    st.pyplot(fig2)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Graphique combiné (T1 vs T2) ----
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("Graphique combiné — sélectionne les séries à afficher")

    show_c_fc_t1 = st.checkbox("☑️ FC Test 1", True)
    show_c_pace_t1 = st.checkbox("☑️ Allure Test 1", False)
    show_c_pow_t1 = st.checkbox("☑️ Puissance Test 1", False)
    show_c_fc_t2 = st.checkbox("☑️ FC Test 2", True)
    show_c_pace_t2 = st.checkbox("☑️ Allure Test 2", False)
    show_c_pow_t2 = st.checkbox("☑️ Puissance Test 2", False)

    if (interval_df1 is not None) or (interval_df2 is not None):
        figC, axC = plt.subplots(figsize=(9.5, 5.2))
        if interval_df1 is not None:
            plot_multi_signals(
                axC, interval_df1, t0=interval_df1["time_s"].iloc[0], who="T1",
                show_fc=show_c_fc_t1,
                show_pace=show_c_pace_t1 and (get_speed_col(interval_df1) is not None),
                show_power=show_c_pow_t1 and ("power_smooth" in interval_df1.columns),
                linewidth=1.9
            )
        if interval_df2 is not None:
            plot_multi_signals(
                axC, interval_df2, t0=interval_df2["time_s"].iloc[0], who="T2",
                show_fc=show_c_fc_t2,
                show_pace=show_c_pace_t2 and (get_speed_col(interval_df2) is not None),
                show_power=show_c_pow_t2 and ("power_smooth" in interval_df2.columns),
                linewidth=1.9
            )
        axC.set_xlabel("Temps segment (s)")
        axC.set_title("Comparaison des cinétiques (FC + Allure + Puissance)")
        axC.grid(True, alpha=0.15)

        handles, labels = [], []
        for a in figC.axes:
            h, l = a.get_legend_handles_labels()
            handles += h; labels += l
        if handles:
            axC.legend(handles, labels, fontsize=8, loc="upper left", frameon=False)

        st.pyplot(figC)
    else:
        st.info("Importe au moins un test et coche les options d’affichage.")
    st.markdown('</div>', unsafe_allow_html=True)
# ---------- Onglet 2 : Analyse entraînement ----------
with tabs[1]:
    st.header("⚙️ Analyse entraînement (multi-séances + IC local + FC/Allure/Puissance)")

    uploaded_sessions = st.file_uploader(
        "Importer un ou plusieurs fichiers (FIT, GPX, CSV, TCX)",
        type=ACCEPTED_TYPES,
        accept_multiple_files=True,
        key="multi_sessions"
    )

    if uploaded_sessions:
        all_summaries = []
        figs = []

        for file in uploaded_sessions:
            try:
                df = load_activity(file)
            except Exception as e:
                st.error(f"Erreur lors de la lecture de {file.name} : {e}")
                continue

            df, window, dur, pauses = smooth_hr(df)
            st.markdown(f"### 📂 {file.name}")
            st.caption(f"Durée : {dur:.1f}s • Lissage {window}s • Pauses : {pauses}")

            cA, cB = st.columns(2)
            with cA:
                s_str = st.text_input(f"Début (hh:mm:ss) – {file.name}", value="0:00:00", key=f"start_{file.name}")
            with cB:
                e_str = st.text_input(f"Fin (hh:mm:ss) – {file.name}", value="0:10:00", key=f"end_{file.name}")

            try:
                s_sec = parse_time_to_seconds(s_str)
                e_sec = parse_time_to_seconds(e_str)
            except:
                st.warning("⛔ Format temps invalide, ignoré.")
                continue

            if e_sec <= s_sec:
                st.warning("⛔ Fin ≤ début, segment ignoré.")
                continue

            seg = df[(df["time_s"] >= s_sec) & (df["time_s"] <= e_sec)]
            if seg.empty:
                st.warning("⚠️ Segment vide.")
                continue

            stats, d_bpm, d_pct = analyze_heart_rate(seg)
            dist_m = segment_distance_m(seg)
            t_s = e_sec - s_sec
            v_kmh = 3.6 * (dist_m / t_s) if t_s > 0 else 0.0

            table = pd.DataFrame({
                "Métrique": [
                    "FC moyenne (bpm)", "Dérive (bpm/min)", "Dérive (%/min)",
                    "Durée (s)", "Distance (m)", "Vitesse moy (km/h)"
                ],
                "Valeur": [
                    stats["FC moyenne (bpm)"], stats["Dérive (bpm/min)"],
                    stats["Dérive (%/min)"], stats["Durée segment (s)"],
                    round(dist_m, 1), round(v_kmh, 2)
                ]
            })
            st.markdown('<div class="table-box">', unsafe_allow_html=True)
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(9, 4.5))
            plot_multi_signals(
                ax, seg, t0=s_sec, who=file.name[:3],
                show_fc=True,
                show_pace=(get_speed_col(seg) is not None),
                show_power=("power_smooth" in seg.columns),
                linewidth=1.8
            )
            ax.set_title(f"Cinétique – {file.name}")
            ax.set_xlabel("Temps segment (s)")
            ax.grid(True, alpha=0.2)
            handles, labels = [], []
            for a in fig.axes:
                h, l = a.get_legend_handles_labels()
                handles += h; labels += l
            if handles:
                ax.legend(handles, labels, fontsize=8, loc="upper left", frameon=False)
            st.pyplot(fig)

            all_summaries.append({
                "Fichier": file.name,
                "FC moy": stats["FC moyenne (bpm)"],
                "Dérive bpm/min": d_bpm,
                "Dérive %/min": d_pct,
                "Durée (s)": t_s,
                "Dist (m)": dist_m,
                "Vit (km/h)": v_kmh
            })
            figs.append(fig)

        if all_summaries:
            df_sum = pd.DataFrame(all_summaries)
            st.markdown("### 📊 Synthèse des segments importés")
            st.dataframe(df_sum, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun segment valide analysé.")


# ---------- Onglet 3 : Analyse générale ----------
with tabs[2]:
    st.header("📊 Analyse générale (Vitesse Critique + Index Cinétique)")

    if interval_df1 is not None and interval_df2 is not None and drift1_pct is not None and drift2_pct is not None:
        st.subheader("⚙️ Calcul de la Vitesse Critique (VC)")

        if dist1_m and dist2_m and t1_s and t2_s and t1_s != t2_s:
            vc_m_s = (dist2_m - dist1_m) / (t2_s - t1_s)
            d_prime = dist1_m - vc_m_s * t1_s
            vc_kmh = vc_m_s * 3.6
            st.success(f"**Vitesse Critique (VC) = {vc_kmh:.2f} km/h** • D′ = {d_prime:.1f} m")

            st.caption("La VC est la vitesse soutenable sans dérive majeure.")
        else:
            st.warning("Impossible de calculer la VC (valeurs invalides).")

        st.subheader("⚙️ Index de Cinétique (IC)")
        ic_val, unite, msg, _, reco = compute_index_cinetique(
            drift1_pct, drift2_pct, drift1_bpm, drift2_bpm
        )

        if ic_val is not None:
            st.markdown(f"**IC = {ic_val*100:.1f}%** ({unite})")
            st.info(msg)
            st.markdown(f"**{reco['titre']}**")
            for s in reco["seances"]:
                st.markdown(f"• {s}")
        else:
            st.warning("Impossible de calculer l'Index de Cinétique (IC).")

        # --------- Export PDF ---------
        st.subheader("📄 Export PDF")
        if st.button("Générer le rapport PDF"):
            figs_export = []
            if interval_df1 is not None:
                fig_export1, axE1 = plt.subplots(figsize=(9, 4.5))
                plot_multi_signals(
                    axE1, interval_df1, t0=start_sec1, who="T1",
                    show_fc=True,
                    show_pace=(get_speed_col(interval_df1) is not None),
                    show_power=("power_smooth" in interval_df1.columns)
                )
                axE1.set_title(f"Test 1 – {test1_date}")
                figs_export.append(fig_export1)

            if interval_df2 is not None:
                fig_export2, axE2 = plt.subplots(figsize=(9, 4.5))
                plot_multi_signals(
                    axE2, interval_df2, t0=start_sec2, who="T2",
                    show_fc=True,
                    show_pace=(get_speed_col(interval_df2) is not None),
                    show_power=("power_smooth" in interval_df2.columns)
                )
                axE2.set_title(f"Test 2 – {test2_date}")
                figs_export.append(fig_export2)

            buf_pdf = fig_to_pdf_bytes(figs_export)
            st.download_button(
                label="📥 Télécharger le rapport PDF",
                data=buf_pdf,
                file_name=f"rapport_endurance_{date.today()}.pdf",
                mime="application/pdf"
            )
    else:
        st.info("🧩 Pour accéder à l’analyse générale, importe deux tests valides.")
