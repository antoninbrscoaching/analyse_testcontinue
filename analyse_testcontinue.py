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
import xml.etree.ElementTree as ET  # ✅ pour lecture TCX

# =============== UI / THEME =================
st.set_page_config(page_title="Analyse Endurance + VC (PDF)", layout="wide")

# Palette rouge/noir/blanc (fond clair) + nuances
COLOR_RED_T1 = "#d21f3c"   # FC Test 1
COLOR_RED_T2 = "#8b0a1a"   # FC Test 2
COLOR_RED_SES = "#f57c92"  # FC Entraînement

COLOR_BLUE_T1 = "#0066cc"  # Allure Test 1
COLOR_BLUE_T2 = "#003366"  # Allure Test 2
COLOR_BLUE_SES = "#66a3ff" # Allure Entraînement

COLOR_ORANGE_T1 = "#ff8c00"  # Puissance Test 1
COLOR_ORANGE_T2 = "#cc6600"  # Puissance Test 2
COLOR_ORANGE_SES = "#ffb84d" # Puissance Entraînement

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


# =============== UTILS ======================

def load_activity(file):
    """Charge un fichier FIT, GPX, CSV ou TCX (robuste avec fitdecode)."""
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

    elif file.name.endswith(".gpx") or file.name.endswith(".GPX"):
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

    elif file.name.endswith(".tcx") or file.name.endswith(".TCX"):
        # ✅ Lecture TCX (Garmin / Coros / Strava)
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
        raise ValueError("Format non supporté (.fit, .gpx, .csv, .tcx uniquement).")

    # Harmoniser la colonne temps -> 'timestamp'
    for c in df.columns:
        if "time" in c.lower():
            df.rename(columns={c: "timestamp"}, inplace=True)
            break

    if "heart_rate" not in df.columns:
        raise ValueError("Le fichier ne contient pas de fréquence cardiaque ('heart_rate').")

    df = df.dropna(subset=["heart_rate"]).reset_index(drop=True)

    # Normaliser types
    for c in ["heart_rate", "speed", "enhanced_speed", "power", "distance", "lat", "lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# =============== AJOUTS : extensions reconnues partout ======================
# ✅ Extensions universelles (avec majuscules/minuscules)
ACCEPTED_TYPES = ["fit", "FIT", "gpx", "GPX", "csv", "CSV", "tcx", "TCX"]

# =============== APP ========================
st.title("🏃‍♂️ Analyse de Tests d'Endurance + Vitesse Critique (Export PDF)")

tabs = st.tabs(["🧪 Tests d'endurance", "⚙️ Analyse entraînement", "📊 Analyse générale"])

# ---------- Onglet 1 : Tests fusionnés ----------
with tabs[0]:
    st.header("🧪 Tests d'endurance")

    ctop = st.columns(2)
    # ---- Test 1
    with ctop[0]:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("Test 1")
        uploaded_file1 = st.file_uploader(
            "Fichier Test 1 (FIT, GPX, CSV, TCX)",
            type=ACCEPTED_TYPES,
            key="file1"
        )
        # (reste de ton bloc test 1 inchangé…)

    # ---- Test 2
    with ctop[1]:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("Test 2")
        uploaded_file2 = st.file_uploader(
            "Fichier Test 2 (FIT, GPX, CSV, TCX)",
            type=ACCEPTED_TYPES,
            key="file2"
        )
        # (reste du bloc test 2 inchangé…)

# ---------- Onglet 2 : Analyse entraînement ----------
with tabs[1]:
    st.header("⚙️ Analyse entraînement (multi-séances + IC local + FC/Allure/Puissance)")
    uploaded_sessions = st.file_uploader(
        "Importer un ou plusieurs fichiers (FIT, GPX, CSV, TCX)",
        type=ACCEPTED_TYPES,
        accept_multiple_files=True,
        key="multi_sessions"
    )
    # (reste inchangé ici…)

# ---------- Onglet 3 : Analyse générale ----------
# (ton code original inchangé)
  

def get_speed_col(df):
    """Retourne le nom de la vitesse (m/s) si dispo."""
    if "enhanced_speed" in df.columns:
        return "enhanced_speed"
    if "speed" in df.columns:
        return "speed"
    return None


def smooth_hr(df, time_col="timestamp", hr_col="heart_rate"):
    """Temps continu (ignore les pauses) + lissage FC (et autres signaux si présents)."""
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

    # Fenêtre lissage
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

    # Lissages
    df["hr_smooth"] = df[hr_col].rolling(window_size, min_periods=1).mean()

    speed_col = get_speed_col(df)
    if speed_col:
        df["speed_smooth"] = df[speed_col].rolling(window_size, min_periods=1).mean()
    if "power" in df.columns:
        df["power_smooth"] = df["power"].rolling(window_size, min_periods=1).mean()

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
    speed_col = next((c for c in df_seg.columns if c.lower() in ("speed", "enhanced_speed")), None)
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


def fig_to_pdf_bytes(figs):
    """Prend un fig (ou une liste de figs) et renvoie un PDF en mémoire."""
    if not isinstance(figs, (list, tuple)):
        figs = [figs]
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        for f in figs:
            f.tight_layout()
            pdf.savefig(f, bbox_inches="tight")
    buf.seek(0)
    return buf


# ====== IC & Rapport ======

def compute_index_cinetique(drift_short_pct, drift_long_pct, drift_short_bpm, drift_long_bpm):
    """
    Retourne:
      - IC (float)
      - unite ('%/min' ou 'bpm/min')
      - message (texte court)
      - niveau (unused)
      - reco (dict)
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

    if IC >= 0.70:
        titre = "Très bonne stabilité sur le long"
        points = [f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}", "Profil endurant fort."]
        seances = ["2–3×(8–12′) à 88–92% VC, r=2–3′", "Tempo 20–30′ à 85–90% VC", "Progressif 30–40′ de 80→90% VC", "Z2 volumineux"]
        msg = "IC élevé : blocs longs & tempos ambitieux."
    elif 0.40 <= IC < 0.70:
        titre = "Bon équilibre, marge en soutien aérobie"
        points = [f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}", "Consolider le tempo/seuil."]
        seances = ["4–6×5′ à 90–92% VC, r=1–2′ (cruise)", "2×12–15′ à 85–90% VC (tempo)", "6–8×(2′ @95% VC / 1′ @80%) (mix)"]
        msg = "IC bon : mix intervals moyens + tempo."
    elif 0.15 <= IC < 0.40:
        titre = "Stabilité limitée sur le long"
        points = [f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}", "Étendre tolérance au long."]
        seances = ["3–4×6′ à 88–90% VC, r=2′", "3×8–10′ à 85–88% VC (tempo court)", "Z2 + 6–10×20″ strides"]
        msg = "IC moyen : allonger progressivement les intervalles."
    elif 0.00 <= IC < 0.15:
        titre = "Dérives longue et courte similaires"
        points = [f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}", "Base aérobie à renforcer."]
        seances = ["Z2 majoritaire", "3–4×6–8′ à 82–86% VC (tempo doux)", "10–12×1′ à 92–95% VC / 1′ Z2"]
        msg = "IC faible : base + tempo doux, peu de >92% VC."
    else:  # IC < 0
        titre = "Stabilité faible / contexte défavorable"
        points = [f"Dérive {unite} courte: {d_short:.3f}", f"Dérive {unite} longue: {d_long:.3f}", "Réduire intensité, re-baser."]
        seances = ["Z2 + force (côtes)", "Progressifs doux", "Limiter >90% VC ; vérifier récupération/chaleur"]
        msg = "IC négatif : re-baser et diagnostiquer (fatigue/conditions)."

    reco = {"titre": titre, "points": points, "seances": seances}
    return float(IC), unite, msg, None, reco


# =============== HELPERS GRAPHIQUES (FC + Allure + Puissance) ======================

def pace_formatter(v, pos):
    """Formateur d'axe pour allure en min:sec/km depuis une valeur en min/km."""
    if v is None or not math.isfinite(v) or v <= 0:
        return ""
    m = int(v)
    s = int(round((v - m) * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d}"

def add_pace_axis(ax):
    """Crée un axe droit pour l'allure (min/km) inversé (plus bas = plus rapide)."""
    ax_pace = ax.twinx()
    ax_pace.set_ylabel("Allure (min/km)")
    ax_pace.yaxis.set_major_formatter(FuncFormatter(pace_formatter))
    ax_pace.invert_yaxis()
    return ax_pace

def add_power_axis(ax, offset=60):
    """Ajoute un second axe droit décalé pour la puissance (W)."""
    ax_pow = ax.twinx()
    ax_pow.spines["right"].set_position(("outward", offset))
    ax_pow.set_frame_on(True)
    ax_pow.patch.set_visible(False)
    ax_pow.set_ylabel("Puissance (W)")
    return ax_pow

def compute_pace_series(df):
    """Allure (min/km) depuis speed_smooth (m/s)."""
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
    """
    Trace FC sur axe gauche, Allure sur axe droit (inversé), Puissance sur axe droit décalé.
    who: "T1" | "T2" | "SES" (pour palette)
    """
    # Couleurs par type et par test/séance
    if who == "T1":
        c_fc, c_pace, c_pow = COLOR_RED_T1, COLOR_BLUE_T1, COLOR_ORANGE_T1
    elif who == "T2":
        c_fc, c_pace, c_pow = COLOR_RED_T2, COLOR_BLUE_T2, COLOR_ORANGE_T2
    else:  # "SES"
        c_fc, c_pace, c_pow = COLOR_RED_SES, COLOR_BLUE_SES, COLOR_ORANGE_SES

    ax_pace = None
    ax_pow = None
    tt = df["time_s"].values - t0

    # FC
    if show_fc and "hr_smooth" in df.columns:
        ax.plot(tt, df["hr_smooth"], color=c_fc, linewidth=linewidth, label=f"{who} • FC (bpm)")
        ax.set_ylabel("FC (bpm)")

    # Allure
    if show_pace and "speed_smooth" in df.columns:
        pace_series = compute_pace_series(df)
        if pace_series is not None:
            ax_pace = add_pace_axis(ax)
            ax_pace.plot(tt, pace_series, color=c_pace, linewidth=linewidth, label=f"{who} • Allure (min/km)")

    # Puissance
    if show_power and "power_smooth" in df.columns:
        ax_pow = add_power_axis(ax, offset=60)
        ax_pow.plot(tt, df["power_smooth"], color=c_pow, linewidth=linewidth, label=f"{who} • Puissance (W)")

    return ax, ax_pace, ax_pow


# =============== APP ========================
st.title("🏃‍♂️ Analyse de Tests d'Endurance + Vitesse Critique (Export PDF)")

tabs = st.tabs(["🧪 Tests d'endurance", "⚙️ Analyse entraînement", "📊 Analyse générale"])

# Variables partagées (onglets)
interval_df1 = stats1 = None
interval_df2 = stats2 = None
drift1_bpm = drift2_bpm = None
drift1_pct = drift2_pct = None
dist1_m = dist2_m = None
t1_s = t2_s = None
test1_date = test2_date = None
start_sec1 = start_sec2 = 0

# ---------- Onglet 1 : Tests fusionnés ----------
with tabs[0]:
    st.header("🧪 Tests d'endurance")

    ctop = st.columns(2)
    # ---- Test 1
    with ctop[0]:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("Test 1")
        uploaded_file1 = st.file_uploader("Fichier Test 1 (FIT, GPX, CSV)", type=["fit", "gpx", "csv"], key="file1")
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
                st.markdown('</div>', unsafe_allow_html=True)
                st.stop()

            if end_sec1 <= start_sec1:
                st.error("La fin doit être supérieure au début.")
            else:
                if end_sec1 > df1["time_s"].max():
                    st.warning("⚠️ Fin > données disponibles. Limitation automatique (Test 1).")
                    end_sec1 = df1["time_s"].max()

                interval_df1 = df1[(df1["time_s"] >= start_sec1) & (df1["time_s"] <= end_sec1)]

                if len(interval_df1) > 10:
                    stats1, drift1_bpm, drift1_pct = analyze_heart_rate(interval_df1)
                    dist1_m = segment_distance_m(interval_df1)
                    t1_s = float(end_sec1 - start_sec1)
                    v1_kmh = 3.6 * (dist1_m / t1_s) if t1_s > 0 else 0.0

                    # Tableau métriques
                    table1 = pd.DataFrame({
                        "Métrique": ["FC moyenne (bpm)", "FC max (bpm)", "Dérive (bpm/min)", "Dérive (%/min)", "Durée (s)", "Distance (m)", "Vitesse moy (km/h)"],
                        "Valeur": [stats1['FC moyenne (bpm)'], stats1['FC max (bpm)'], stats1['Dérive (bpm/min)'],
                                   stats1['Dérive (%/min)'], stats1['Durée segment (s)'], round(dist1_m,1), round(v1_kmh,2)]
                    })
                    st.markdown('<div class="table-box">', unsafe_allow_html=True)
                    st.dataframe(table1, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Graphique (FC + Allure + Puissance, double axe)
                    fig1, ax1 = plt.subplots(figsize=(9, 4.8))
                    plot_multi_signals(
                        ax1, interval_df1, t0=start_sec1, who="T1",
                        show_fc=show_t1_fc,
                        show_pace=show_t1_pace and (get_speed_col(interval_df1) is not None),
                        show_power=show_t1_power and ("power_smooth" in interval_df1.columns),
                        linewidth=1.9
                    )
                    ax1.set_xlabel("Temps segment (s)")
                    ax1.set_title(f"Cinétique - Test 1 ({test1_date})")
                    ax1.grid(True, alpha=0.15)

                    # Légende fusionnée
                    handles, labels = [], []
                    for a in fig1.axes:
                        h, l = a.get_legend_handles_labels()
                        handles += h; labels += l
                    if handles:
                        ax1.legend(handles, labels, fontsize=8, loc="upper left", frameon=False)

                    st.pyplot(fig1)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Test 2
    with ctop[1]:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("Test 2")
        uploaded_file2 = st.file_uploader("Fichier Test 2 (FIT, GPX, CSV)", type=["fit", "gpx", "csv"], key="file2")
        test2_date = st.date_input("📅 Date du test 2", value=date.today(), key="date2")

        show_t2_fc = st.checkbox("☑️ FC (Test 2)", value=True, key="t2_fc")
        show_t2_pace = st.checkbox("☑️ Allure (Test 2)", value=False, key="t2_pace")
        show_t2_power = st.checkbox("☑️ Puissance (Test 2)", value=False, key="t2_power")

        if uploaded_file2:
            try:
                df2 = load_activity(uploaded_file2)
            except Exception as e:
                st.error(f"Erreur fichier 2 : {e}")
                st.markdown('</div>', unsafe_allow_html=True)
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
                st.markdown('</div>', unsafe_allow_html=True)
                st.stop()

            if end_sec2 <= start_sec2:
                st.error("La fin doit être supérieure au début.")
            else:
                if end_sec2 > df2["time_s"].max():
                    st.warning("⚠️ Fin > données disponibles. Limitation automatique (Test 2).")
                    end_sec2 = df2["time_s"].max()

                interval_df2 = df2[(df2["time_s"] >= start_sec2) & (df2["time_s"] <= end_sec2)]

                if len(interval_df2) > 10:
                    stats2, drift2_bpm, drift2_pct = analyze_heart_rate(interval_df2)
                    dist2_m = segment_distance_m(interval_df2)
                    t2_s = float(end_sec2 - start_sec2)
                    v2_kmh = 3.6 * (dist2_m / t2_s) if t2_s > 0 else 0.0

                    # Tableau métriques
                    table2 = pd.DataFrame({
                        "Métrique": ["FC moyenne (bpm)", "FC max (bpm)", "Dérive (bpm/min)", "Dérive (%/min)", "Durée (s)", "Distance (m)", "Vitesse moy (km/h)"],
                        "Valeur": [stats2['FC moyenne (bpm)'], stats2['FC max (bpm)'], stats2['Dérive (bpm/min)'],
                                   stats2['Dérive (%/min)'], stats2['Durée segment (s)'], round(dist2_m,1), round(v2_kmh,2)]
                    })
                    st.markdown('<div class="table-box">', unsafe_allow_html=True)
                    st.dataframe(table2, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Graphique
                    fig2, ax2 = plt.subplots(figsize=(9, 4.8))
                    plot_multi_signals(
                        ax2, interval_df2, t0=start_sec2, who="T2",
                        show_fc=show_t2_fc,
                        show_pace=show_t2_pace and (get_speed_col(interval_df2) is not None),
                        show_power=show_t2_power and ("power_smooth" in interval_df2.columns),
                        linewidth=1.9
                    )
                    ax2.set_xlabel("Temps segment (s)")
                    ax2.set_title(f"Cinétique - Test 2 ({test2_date})")
                    ax2.grid(True, alpha=0.15)

                    handles, labels = [], []
                    for a in fig2.axes:
                        h, l = a.get_legend_handles_labels()
                        handles += h; labels += l
                    if handles:
                        ax2.legend(handles, labels, fontsize=8, loc="upper left", frameon=False)

                    st.pyplot(fig2)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Graphique combiné unique (sélections)
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("Graphique combiné — sélectionne les séries à afficher (PDF via onglet 3)")
    show_c_fc_t1 = st.checkbox("☑️ FC Test 1", value=True)
    show_c_pace_t1 = st.checkbox("☑️ Allure Test 1", value=False)
    show_c_pow_t1 = st.checkbox("☑️ Puissance Test 1", value=False)
    show_c_fc_t2 = st.checkbox("☑️ FC Test 2", value=True)
    show_c_pace_t2 = st.checkbox("☑️ Allure Test 2", value=False)
    show_c_pow_t2 = st.checkbox("☑️ Puissance Test 2", value=False)

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

    # État persistant
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}  # {nom_fichier: dataframe}
    if "training_intervals" not in st.session_state:
        st.session_state.training_intervals = []  # liste de dicts d’intervalles

    # --- Import multi-séances ---
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("📂 Importer des séances d'entraînement")
    uploaded_sessions = st.file_uploader(
        "Importer un ou plusieurs fichiers (FIT, GPX, CSV)",
        type=["fit", "gpx", "csv"],
        accept_multiple_files=True,
        key="multi_sessions"
    )
    if uploaded_sessions:
        for f in uploaded_sessions:
            if f.name not in st.session_state.sessions:
                try:
                    df = load_activity(f)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    df = df.dropna(subset=["timestamp"])
                    df, w, dur, pauses = smooth_hr(df)
                    st.session_state.sessions[f.name] = df
                    st.success(f"✅ {f.name} importé ({dur/60:.1f} min, {len(df)} points)")
                except Exception as e:
                    st.error(f"Erreur fichier {f.name} : {e}")

    if st.session_state.sessions:
        info = []
        for name, df in st.session_state.sessions.items():
            speed_col = get_speed_col(df)
            info.append({
                "Séance": name,
                "Durée (min)": round(df["time_s"].iloc[-1] / 60, 1),
                "Nb points": len(df),
                "FC moy (bpm)": round(df["heart_rate"].mean(), 1),
                "Allure dispo": "Oui" if speed_col else "Non",
                "Puissance dispo": "Oui" if "power_smooth" in df.columns or "power" in df.columns else "Non"
            })
        st.markdown('<div class="table-box">', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(info), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Importe une ou plusieurs séances pour commencer.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Ajouter un intervalle (illimité) ---
    if st.session_state.sessions:
        with st.form("add_interval_form_train"):
            st.markdown("### ➕ Ajouter un intervalle à analyser")
            session_name = st.selectbox("Séance source", options=list(st.session_state.sessions.keys()))
            c1, c2 = st.columns(2)
            with c1:
                start_str = st.text_input("Début (hh:mm:ss)", value="0:00:00", key="train_start")
            with c2:
                end_str = st.text_input("Fin (hh:mm:ss)", value="0:05:00", key="train_end")

            c3, c4, c5 = st.columns(3)
            with c3:
                show_fc_it = st.checkbox("FC", value=True, key="it_fc")
            with c4:
                show_pace_it = st.checkbox("Allure", value=False, key="it_pace")
            with c5:
                show_power_it = st.checkbox("Puissance", value=False, key="it_pow")

            add_btn = st.form_submit_button("Ajouter l'intervalle")

        def analyze_interval(session_name, start_str, end_str, show_fc_it, show_pace_it, show_power_it):
            df = st.session_state.sessions.get(session_name)
            if df is None:
                st.warning("Séance introuvable.")
                return None
            try:
                s = parse_time_to_seconds(start_str)
                e = parse_time_to_seconds(end_str)
            except:
                st.warning("Format temps invalide.")
                return None
            if e <= s:
                st.warning("La fin doit être > début.")
                return None

            df_seg = df[(df["time_s"] >= s) & (df["time_s"] <= e)].copy()
            if len(df_seg) < 10:
                st.warning("Segment trop court.")
                return None

            stats, drift_bpm, drift_pct = analyze_heart_rate(df_seg)
            dist_m = segment_distance_m(df_seg)
            t_s = e - s
            v_kmh = 3.6 * (dist_m / t_s) if t_s > 0 else 0.0

            # IC local = dérive locale (bpm/min & %/min)
            ic_local_bpm = drift_bpm
            ic_local_pct = drift_pct

            curve_time = (df_seg["time_s"] - df_seg["time_s"].iloc[0]).values
            speed_col = get_speed_col(df_seg)

            series = {
                "_curve_time": curve_time,
                "_curve_bpm": df_seg["hr_smooth"].values if "hr_smooth" in df_seg.columns else None,
                "_curve_pace": compute_pace_series(df_seg).values if speed_col else None,
                "_curve_power": df_seg["power_smooth"].values if "power_smooth" in df_seg.columns else None
            }

            return {
                "Séance": session_name,
                "Début": start_str,
                "Fin": end_str,
                "Durée (s)": round(t_s, 1),
                "FC moy (bpm)": stats["FC moyenne (bpm)"],
                "FC max (bpm)": stats["FC max (bpm)"],
                "Dérive (bpm/min)": stats["Dérive (bpm/min)"],
                "Dérive (%/min)": stats["Dérive (%/min)"],
                "Distance (m)": round(dist_m, 1),
                "Vitesse (km/h)": round(v_kmh, 2),
                "IC local (bpm/min)": round(ic_local_bpm, 4) if ic_local_bpm is not None else None,
                "IC local (%/min)": round(ic_local_pct, 4) if ic_local_pct is not None else None,
                **series,
                "_id": f"{session_name} {start_str}→{end_str}",
                "Afficher": True,
                "show_fc": bool(show_fc_it),
                "show_pace": bool(show_pace_it),
                "show_power": bool(show_power_it)
            }

        if add_btn:
            res = analyze_interval(session_name, start_str, end_str, show_fc_it, show_pace_it, show_power_it)
            if res:
                st.session_state.training_intervals.append(res)
                st.success(f"Intervalle ajouté ({res['_id']})")

    # --- Tableau global des intervalles + export CSV (optionnel) ---
    if st.session_state.training_intervals:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("📋 Intervalles analysés (IC local inclus)")
        df_int = pd.DataFrame([
            {k:v for k,v in d.items()
             if not k.startswith("_curve_") and not k.startswith("_id") and not k.startswith("show_")}
            for d in st.session_state.training_intervals
        ])
        st.markdown('<div class="table-box">', unsafe_allow_html=True)
        st.dataframe(df_int, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- 3 panneaux graphiques superposables ---
        st.markdown("### 📈 Panneaux de graphiques (superpose les intervalles choisis)")

        def render_panel(panel_title, key_prefix):
            st.markdown(f'#### {panel_title}')
            ids = [d["_id"] for d in st.session_state.training_intervals]
            default_sel = [d["_id"] for d in st.session_state.training_intervals if d.get("Afficher", True)]
            selected = st.multiselect("Intervalles à afficher", options=ids, default=default_sel, key=f"{key_prefix}_sel")

            cfc, cpa, cpw = st.columns(3)
            with cfc:
                show_fc = st.checkbox("FC (bpm)", value=True, key=f"{key_prefix}_fc")
            with cpa:
                show_pace = st.checkbox("Allure (min/km)", value=False, key=f"{key_prefix}_pace")
            with cpw:
                show_power = st.checkbox("Puissance (W)", value=False, key=f"{key_prefix}_pow")

            if not selected:
                st.info("Sélectionne au moins un intervalle.")
                return

            fig, ax = plt.subplots(figsize=(9.5, 5.2))

            for d in st.session_state.training_intervals:
                if d["_id"] not in selected:
                    continue
                tt = d["_curve_time"]
                # Palette "SES"
                # FC
                if show_fc and d["_curve_bpm"] is not None:
                    ax.plot(tt, d["_curve_bpm"], label=f"{d['_id']} • FC", color=COLOR_RED_SES, linewidth=1.6)
                # Allure
                if show_pace and d["_curve_pace"] is not None:
                    ax_pace = add_pace_axis(ax) if not hasattr(ax, "_pace_added") else ax._pace_axis
                    if not hasattr(ax, "_pace_added"):
                        ax._pace_added = True
                        ax._pace_axis = ax_pace
                    ax_pace.plot(tt, d["_curve_pace"], label=f"{d['_id']} • Allure", color=COLOR_BLUE_SES, linewidth=1.4)
                # Puissance
                if show_power and d["_curve_power"] is not None:
                    ax_pow = add_power_axis(ax, offset=60 if not hasattr(ax, "_pow_added") else 60)
                    ax._pow_added = True
                    ax_pow.plot(tt, d["_curve_power"], label=f"{d['_id']} • Puissance", color=COLOR_ORANGE_SES, linewidth=1.4)

            ax.set_xlabel("Temps intervalle (s)")
            ax.set_ylabel("FC (bpm)")
            ax.set_title(panel_title)
            ax.grid(True, alpha=0.15)

            # Légende combinée
            handles, labels = [], []
            for a in fig.axes:
                h, l = a.get_legend_handles_labels()
                handles += h; labels += l
            if handles:
                ax.legend(handles, labels, fontsize=8, loc="upper left", frameon=False)

            st.pyplot(fig)

        render_panel("Panneau A", "panelA")
        render_panel("Panneau B", "panelB")
        render_panel("Panneau C", "panelC")

        c1, c2 = st.columns(2)
        if c1.button("🗑️ Vider les intervalles"):
            st.session_state.training_intervals = []
            st.experimental_rerun()
        if c2.button("🗑️ Vider les séances"):
            st.session_state.sessions = {}
            st.experimental_rerun()
    else:
        st.info("Ajoute des intervalles (depuis les séances importées) pour afficher les résultats et l'IC local.")

# ---------- Onglet 3 : Analyse générale (VC + IC + Export PDF) ----------
with tabs[2]:
    st.header("📊 Analyse générale : VC, Index de cinétique & Export PDF")

    vc_dict = None
    IC_value = None
    IC_unite = None
    IC_msg = None
    IC_reco = None

    # Bouton global pour créer un PDF multi-pages des figures actuellement affichées
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("🖨️ Export PDF")
    st.caption("Le PDF inclut les figures créées depuis cette session (tests, combinés, entraînement…).")

    if st.button("📄 Générer un PDF de toutes les figures ouvertes"):
        # On ne peut pas récupérer rétroactivement toutes les figs Streamlit,
        # donc on génère une page de synthèse dédiée + on propose de régénérer les principaux graphiques.
        figs = []

        # Page de garde
        fig0 = plt.figure(figsize=(8.5, 11))
        fig0.text(0.05, 0.92, "Rapport d'analyse (PDF)", fontsize=20, weight="bold")
        fig0.text(0.05, 0.86, "Contenu :", fontsize=12)
        fig0.text(0.07, 0.82, "• Tests d'endurance (T1/T2) — courbes FC/Allure/Puissance", fontsize=10)
        fig0.text(0.07, 0.78, "• Graphique combiné (sélection)", fontsize=10)
        fig0.text(0.07, 0.74, "• Analyse générale : VC, IC et comparatifs", fontsize=10)
        fig0.text(0.05, 0.06, "Conseil : Utilise ensuite la fonction d'impression du lecteur PDF pour \"A4 portrait\".", fontsize=9, color=COLOR_GREY)
        figs.append(fig0)

        # (Re)générer quelques figures clés si disponibles
        if interval_df1 is not None:
            f1, ax = plt.subplots(figsize=(9, 4.8))
            plot_multi_signals(ax, interval_df1, t0=interval_df1["time_s"].iloc[0], who="T1",
                               show_fc=True,
                               show_pace=(get_speed_col(interval_df1) is not None),
                               show_power=("power_smooth" in interval_df1.columns),
                               linewidth=1.8)
            ax.set_title("Test 1 — courbes")
            ax.set_xlabel("Temps (s)")
            ax.grid(True, alpha=0.15)
            handles, labels = [], []
            for a in f1.axes:
                h, l = a.get_legend_handles_labels()
                handles += h; labels += l
            if handles: ax.legend(handles, labels, fontsize=8, loc="upper left", frameon=False)
            figs.append(f1)

        if interval_df2 is not None:
            f2, ax = plt.subplots(figsize=(9, 4.8))
            plot_multi_signals(ax, interval_df2, t0=interval_df2["time_s"].iloc[0], who="T2",
                               show_fc=True,
                               show_pace=(get_speed_col(interval_df2) is not None),
                               show_power=("power_smooth" in interval_df2.columns),
                               linewidth=1.8)
            ax.set_title("Test 2 — courbes")
            ax.set_xlabel("Temps (s)")
            ax.grid(True, alpha=0.15)
            handles, labels = [], []
            for a in f2.axes:
                h, l = a.get_legend_handles_labels()
                handles += h; labels += l
            if handles: ax.legend(handles, labels, fontsize=8, loc="upper left", frameon=False)
            figs.append(f2)

        # Export
        buf = fig_to_pdf_bytes(figs)
        st.download_button("⬇️ Télécharger le PDF", data=buf, file_name="rapport_endurance.pdf", mime="application/pdf")

        # Fermer les figs créées pour le PDF
        for f in figs:
            plt.close(f)

    st.markdown('</div>', unsafe_allow_html=True)

    # === Calculs VC + IC (comparatif T1 vs T2) ===
    if ("interval_df1" not in locals()) or ("interval_df2" not in locals()):
        pass

    if ('interval_df1' in locals() and interval_df1 is not None) and ('interval_df2' in locals() and interval_df2 is not None) and \
       (t1_s and t2_s) and (dist1_m and dist2_m):

        # Définir "court" et "long" par durée
        if t1_s <= t2_s:
            drift_short_bpm, drift_long_bpm = drift1_bpm, drift2_bpm
            drift_short_pct, drift_long_pct = drift1_pct, drift2_pct
        else:
            drift_short_bpm, drift_long_bpm = drift2_bpm, drift1_bpm
            drift_short_pct, drift_long_pct = drift2_pct, drift1_pct

        # VC 2 points
        D1, T1 = float(dist1_m), float(t1_s)
        D2, T2 = float(dist2_m), float(t2_s)

        vc_dict = None
        if (T2 != T1) and (D1 > 0 and D2 > 0 and T1 > 0 and T2 > 0):
            CS = (D2 - D1) / (T2 - T1)
            D_prime = D1 - CS * T1
            V_kmh = CS * 3.6
            if V_kmh > 0 and math.isfinite(V_kmh):
                pace = format_pace_min_per_km(V_kmh)
                pace_str = f"{pace[0]}:{pace[1]:02d} min/km" if pace else "—"
                vc_dict = {"CS": CS, "V_kmh": V_kmh, "D_prime": D_prime, "pace_str": pace_str}

        # IC comparatif
        IC_value, IC_unite, IC_msg, _, IC_reco = compute_index_cinetique(
            drift_short_pct, drift_long_pct, drift_short_bpm, drift_long_bpm
        )

        # Tableau de synthèse
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("🧾 Synthèse VC & IC")
        tab_synth = []
        if vc_dict:
            tab_synth += [
                {"Bloc":"VC/CS","Clé":"CS (m/s)","Valeur":f"{vc_dict['CS']:.2f}"},
                {"Bloc":"VC/CS","Clé":"VC (km/h)","Valeur":f"{vc_dict['V_kmh']:.2f}"},
                {"Bloc":"VC/CS","Clé":"Allure VC","Valeur":vc_dict["pace_str"]},
                {"Bloc":"VC/CS","Clé":"D′ (m)","Valeur":f"{vc_dict['D_prime']:.0f}"}
            ]
        if IC_value is not None:
            tab_synth += [
                {"Bloc":"Index de cinétique","Clé":"IC (comparatif)","Valeur":f"{IC_value:.3f}"},
                {"Bloc":"Index de cinétique","Clé":"Unité","Valeur":IC_unite},
                {"Bloc":"Index de cinétique","Clé":"Note","Valeur":IC_msg}
            ]
        if tab_synth:
            st.markdown('<div class="table-box">', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(tab_synth), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Graphique comparatif final (FC+Allure+Puissance) avec cases
        st.markdown("### Graphique comparatif — sélection (à inclure dans PDF via bouton ci-dessus)")
        gc1, gc2, gc3, gc4, gc5, gc6 = st.columns(6)
        with gc1:
            g_fc_t1 = st.checkbox("FC T1", True, key="g_fc_t1")
        with gc2:
            g_pace_t1 = st.checkbox("Allure T1", False, key="g_pace_t1")
        with gc3:
            g_pow_t1 = st.checkbox("Puissance T1", False, key="g_pow_t1")
        with gc4:
            g_fc_t2 = st.checkbox("FC T2", True, key="g_fc_t2")
        with gc5:
            g_pace_t2 = st.checkbox("Allure T2", False, key="g_pace_t2")
        with gc6:
            g_pow_t2 = st.checkbox("Puissance T2", False, key="g_pow_t2")

        figC, axC = plt.subplots(figsize=(9.5, 5.2))
        plot_multi_signals(
            axC, interval_df1, t0=interval_df1["time_s"].iloc[0], who="T1",
            show_fc=g_fc_t1,
            show_pace=g_pace_t1 and (get_speed_col(interval_df1) is not None),
            show_power=g_pow_t1 and ("power_smooth" in interval_df1.columns),
            linewidth=1.9
        )
        plot_multi_signals(
            axC, interval_df2, t0=interval_df2["time_s"].iloc[0], who="T2",
            show_fc=g_fc_t2,
            show_pace=g_pace_t2 and (get_speed_col(interval_df2) is not None),
            show_power=g_pow_t2 and ("power_smooth" in interval_df2.columns),
            linewidth=1.9
        )
        axC.set_xlabel("Temps segment (s)")
        axC.set_title("Comparaison (FC + Allure + Puissance)")
        axC.grid(True, alpha=0.15)

        handles, labels = [], []
        for a in figC.axes:
            h, l = a.get_legend_handles_labels()
            handles += h; labels += l
        if handles:
            axC.legend(handles, labels, fontsize=8, loc="upper left", frameon=False)

        st.pyplot(figC)

    else:
        st.info("Importe et définis les segments des deux tests pour activer la synthèse VC/IC et l'export PDF.")
