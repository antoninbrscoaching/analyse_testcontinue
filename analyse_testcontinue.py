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
import xml.etree.ElementTree as ET
import gpxpy

# ========================= CONFIG / THEME ==============================
st.set_page_config(page_title="Analyse Tests Endurance + VC", layout="wide")

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
.table-box {
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 10px;
  padding: 0.4rem 0.6rem;
  background: #fff;
  box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.block-container { padding-top: 1.4rem; }
</style>
""", unsafe_allow_html=True)

ACCEPTED_TYPES = ["fit","FIT","gpx","GPX","csv","CSV","tcx","TCX"]

# ========================= LECTURE FICHIERS ==============================

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
            raise ValueError(f"Erreur FIT : {e}")

    elif name.endswith(".gpx"):
        gpx = gpxpy.parse(file)
        data = []
        for trk in gpx.tracks:
            for seg in trk.segments:
                for pt in seg.points:
                    data.append({
                        "timestamp": pt.time,
                        "lat": pt.latitude,
                        "lon": pt.longitude,
                        "alt": pt.elevation
                    })
        df = pd.DataFrame(data)

    elif name.endswith(".tcx"):
        try:
            content = file.read().decode("utf-8", errors="ignore")
            root = ET.fromstring(content)

            data = []
            for tp in root.findall(".//{*}Trackpoint"):
                t = tp.find(".//{*}Time")
                hr = tp.find(".//{*}HeartRateBpm/{*}Value")
                dist = tp.find(".//{*}DistanceMeters")
                alt = tp.find(".//{*}AltitudeMeters")
                lat = tp.find(".//{*}Position/{*}LatitudeDegrees")
                lon = tp.find(".//{*}Position/{*}LongitudeDegrees")
                powv = tp.find(".//{*}Watts")

                data.append({
                    "timestamp": t.text if t is not None else None,
                    "heart_rate": float(hr.text) if hr is not None else None,
                    "distance": float(dist.text) if dist is not None else None,
                    "alt": float(alt.text) if alt is not None else None,
                    "power": float(powv.text) if powv is not None else None,
                    "lat": float(lat.text) if lat is not None else None,
                    "lon": float(lon.text) if lon is not None else None,
                })

            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError("TCX vide")
        except Exception as e:
            raise ValueError(f"Erreur TCX : {e}")

    else:
        raise ValueError("Format non supporté (.fit, .gpx, .csv, .tcx uniquement)")

    # Harmonisation timestamp
    for c in df.columns:
        if "time" in c.lower():
            df = df.rename(columns={c: "timestamp"})
            break

    if "heart_rate" not in df.columns:
        raise ValueError("Pas de FC détectée")

    df = df.dropna(subset=["heart_rate"]).reset_index(drop=True)

    # Nettoyage
    for c in ["heart_rate","speed","enhanced_speed","power","distance","lat","lon","alt"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# ========================= OUTILS CALCUL ==============================

def get_speed_col(df):
    if "enhanced_speed" in df.columns: return "enhanced_speed"
    if "speed" in df.columns: return "speed"
    return None

# ------------------------------------------------------------
# Lissage Cardio
# ------------------------------------------------------------
def smooth_hr(df, time_col="timestamp", hr_col="heart_rate"):
    df = df.copy().sort_values(by=time_col).reset_index(drop=True)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    df["delta_t"] = df[time_col].diff().dt.total_seconds().fillna(0)
    median_step = np.median(df["delta_t"][df["delta_t"] > 0])
    if np.isnan(median_step) or median_step == 0:
        median_step = 1
    df.loc[df["delta_t"] > 2 * median_step, "delta_t"] = median_step

    df["time_s"] = df["delta_t"].cumsum()
    total_dur = df["time_s"].iloc[-1]

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
    sp = get_speed_col(df)
    if sp:
        df["speed_smooth"] = df[sp].rolling(window_size, min_periods=1).mean()
    if "power" in df.columns:
        df["power_smooth"] = df["power"].rolling(window_size, min_periods=1).mean()

    pauses = (df["delta_t"] > 2 * median_step).sum()
    return df, window_sec, total_dur, pauses


# ------------------------------------------------------------
# Analyse FC
# ------------------------------------------------------------
def analyze_heart_rate(df):
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


# ------------------------------------------------------------
# Outils divers
# ------------------------------------------------------------
def parse_time_to_seconds(tstr: str) -> int:
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


def segment_elevation_gain_m(df_seg):
    """Gain D+ sur le segment (somme des hausses d'altitude), robuste au bruit."""
    if df_seg is None or df_seg.empty:
        return 0.0
    if "alt" not in df_seg.columns:
        return 0.0

    alt = pd.to_numeric(df_seg["alt"], errors="coerce").astype(float)
    alt = alt.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
    alt = alt.rolling(5, min_periods=1, center=True).median()

    d = alt.diff().fillna(0.0)
    gain = float(d[d > 0].sum())
    if not math.isfinite(gain):
        return 0.0
    return max(0.0, gain)


def segment_grade_percent(df_seg):
    """Pente moyenne en % = 100 * (D+ net) / distance horizontale."""
    dist_m = segment_distance_m(df_seg)
    if dist_m <= 0:
        return None

    if "alt" not in df_seg.columns or df_seg["alt"].dropna().empty:
        return None

    alt = pd.to_numeric(df_seg["alt"], errors="coerce").astype(float)
    alt = alt.replace([np.inf, -np.inf], np.nan)
    if alt.notna().sum() < 2:
        return None

    alt0 = float(alt.dropna().iloc[0])
    alt1 = float(alt.dropna().iloc[-1])
    deniv_net = alt1 - alt0

    grade = 100.0 * (deniv_net / dist_m)
    if not math.isfinite(grade):
        return None
    return float(grade)


def format_pace_min_per_km(v_kmh):
    if v_kmh <= 0 or not math.isfinite(v_kmh):
        return None
    min_per_km = 60.0 / v_kmh
    total_seconds = int(round(min_per_km * 60.0))
    return total_seconds // 60, total_seconds % 60, min_per_km


def fig_to_pdf_bytes(figs):
    if not isinstance(figs, (list, tuple)):
        figs = [figs]
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        for f in figs:
            f.tight_layout()
            pdf.savefig(f, bbox_inches="tight")
    buf.seek(0)
    return buf


# ------------------------------------------------------------
# NOUVEAUX OUTILS — FC vs Allure
# ------------------------------------------------------------

def compute_pace_series(df):
    """Convertit la vitesse m/s → min/km (retourne une série pandas)."""
    sp = get_speed_col(df)
    if not sp:
        return None

    v = df[sp].astype(float)
    v_kmh = v * 3.6
    pace = 60.0 / v_kmh.replace(0, np.nan)
    return pace


def compare_fc_pace(df):
    """
    Analyse la relation FC ↗️ / Allure ↘️ (ou inverse)
    Retourne :
        - corr : coefficient de corrélation
        - slope : pente linéaire FC = a·allure + b
        - msg : interprétation automatique
    """
    if "speed_smooth" not in df.columns:
        return None, None, "Allure non disponible"

    pace = compute_pace_series(df)
    if pace is None:
        return None, None, "Allure non disponible"

    mask = np.isfinite(df["hr_smooth"]) & np.isfinite(pace)
    if mask.sum() < 30:
        return None, None, "Données insuffisantes"

    x = pace[mask]
    y = df["hr_smooth"][mask]

    corr = np.corrcoef(x, y)[0, 1]
    slope, intercept, _, _, _ = linregress(x, y)

    if corr > 0.5:
        msg = "La FC augmente lorsque l’allure ralentit → dérive cardiaque ou fatigue."
    elif corr < -0.5:
        msg = "La FC augmente quand l’allure accélère → comportement normal."
    else:
        msg = "Relation faible : allure et FC peu liées sur ce segment."

    return float(corr), float(slope), msg

# ========================= CINÉTIQUE VITESSE =========================
def analyze_speed_kinetics(df):
    """
    Retourne la dérive de vitesse en km/h/min et en %/min
    df doit contenir 'speed_smooth' ou 'enhanced_speed'.
    """
    sp_col = get_speed_col(df)
    if sp_col is None or df[sp_col].dropna().empty:
        return None, None

    slope, _, _, _, _ = linregress(df["time_s"], df[sp_col])
    drift_per_min = slope * 60
    mean_speed = df[sp_col].mean()
    drift_percent = (drift_per_min / mean_speed * 100) if mean_speed > 0 else None

    return round(drift_per_min, 4), round(drift_percent, 4) if drift_percent is not None else None

# ========================= INDEX CINÉTIQUE ==============================

def compute_index_cinetique(drift_short_pct, drift_long_pct, drift_short_bpm, drift_long_bpm):
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
        seances = [
            "2–3×(8–12′) à 88–92% VC, r=2–3′",
            "Tempo 20–30′ à 85–90% VC",
            "Progressif 30–40′ de 80→90% VC",
            "Z2 volumineux"
        ]
    elif 0.40 <= IC < 0.70:
        titre = "Bon équilibre, marge en soutien aérobie"
        msg = "IC bon : mix intervals moyens + tempo."
        seances = [
            "4–6×5′ à 90–92% VC, r=1–2′",
            "2×12–15′ à 85–90% VC",
            "6–8×(2′ @95% VC / 1′ @80%)"
        ]
    elif 0.15 <= IC < 0.40:
        titre = "Stabilité limitée sur le long"
        msg = "IC moyen : allonger progressivement les intervalles."
        seances = [
            "3–4×6′ à 88–90% VC",
            "3×8–10′ à 85–88% VC",
            "Z2 + 6–10×20″ strides"
        ]
    elif 0.00 <= IC < 0.15:
        titre = "Dérives longue et courte similaires"
        msg = "IC faible : base + tempo doux, peu de >92% VC."
        seances = [
            "Z2 majoritaire",
            "3–4×6–8′ à 82–86% VC",
            "10–12×1′ à 92–95% VC / 1′ Z2"
        ]
    else:
        titre = "Stabilité faible / contexte défavorable"
        msg = "IC négatif : re-baser et diagnostiquer (fatigue/conditions)."
        seances = [
            "Z2 + force (côtes)",
            "Progressifs doux",
            "Limiter >90% VC ; vérifier récupération"
        ]

    reco = {"titre": titre, "seances": seances}
    return float(IC), unite, msg, None, reco

# ========================= AIDES GRAPHIQUES ==============================

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
    if who == "T1":
        c_fc, c_pace, c_pow = COLOR_RED_T1, COLOR_BLUE_T1, COLOR_ORANGE_T1
    elif who == "T2":
        c_fc, c_pace, c_pow = COLOR_RED_T2, COLOR_BLUE_T2, COLOR_ORANGE_T2
    else:
        c_fc, c_pace, c_pow = COLOR_RED_SES, COLOR_BLUE_SES, COLOR_ORANGE_SES

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

# ========================= APP PRINCIPALE ==============================

st.title("🏃‍♂️ Analyse de Tests d'Endurance + Vitesse Critique (Export PDF)")

tabs = st.tabs(["🧪 Tests d'endurance", "⚙️ Analyse entraînement"])

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "tests"

# ---------------------------------------------------------------------
# ONGLET 1 : TESTS D’ENDURANCE (2 à 6 tests + VC + LOG + D′)
# ---------------------------------------------------------------------
with tabs[0]:
    st.session_state.active_tab = "tests"
    st.header("🧪 Tests d'endurance (2 à 6 tests)")

    # Gestion nombre de tests
    if "nb_tests" not in st.session_state:
        st.session_state.nb_tests = 2  # minimum 2

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("➕ Ajouter un test", use_container_width=True):
            if st.session_state.nb_tests < 6:
                st.session_state.nb_tests += 1
    with colB:
        if st.button("➖ Supprimer un test", use_container_width=True):
            if st.session_state.nb_tests > 2:
                st.session_state.nb_tests -= 1

    st.markdown(f"### Nombre de tests sélectionnés : **{st.session_state.nb_tests}**")

    tests_data = []  # tous les tests analysés
    VC_kmh = None
    D_prime = None
    A = None
    k_log = None  # pour ne pas écraser k() de python

    # >>> AJOUT CONDITIONS : paramètres globaux de recalibrage (pente + température)
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("🌡️ Recalibrage des références (pente + température)")

    colR1, colR2, colR3 = st.columns(3)
    with colR1:
        temp_ref_c = st.number_input("Température de référence (°C)", value=15.0, step=0.5, key="temp_ref")
    with colR2:
        temp_act_c = st.number_input("Température du test (°C)", value=15.0, step=0.5, key="temp_act")
    with colR3:
        temp_coef_pct_per_c = st.number_input("Impact température (% / °C)", value=0.30, step=0.05, key="temp_coef")

    colR4, colR5 = st.columns(2)
    with colR4:
        grade_ref_pct = st.number_input("Pente de référence (%)", value=0.0, step=0.1, key="grade_ref")
    with colR5:
        grade_coef_pct_per_pct = st.number_input("Impact pente (% / %)", value=1.00, step=0.05, key="grade_coef")

    st.caption(
        "Principe : on calcule une *vitesse équivalente* corrigée des conditions pour comparer les tests entre eux. "
        "Tu peux ajuster les coefficients sans toucher au reste du code."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    def apply_conditions_correction(v_kmh_raw, grade_pct, temp_act, temp_ref, temp_coef_pct_per_c, grade_ref, grade_coef_pct_per_pct):
        """Retourne v_kmh_eq (vitesse équivalente) corrigée pente + température.
        Convention : si conditions plus difficiles (plus chaud / plus de pente), v_eq augmente.
        """
        if v_kmh_raw is None or not math.isfinite(v_kmh_raw) or v_kmh_raw <= 0:
            return None

        # Température : pénalité proportionnelle à (temp_act - temp_ref)
        dT = float(temp_act - temp_ref)
        temp_factor = 1.0 - (temp_coef_pct_per_c / 100.0) * dT

        # Pente : pénalité proportionnelle à (grade_pct - grade_ref)
        if grade_pct is None or not math.isfinite(grade_pct):
            dG = 0.0
        else:
            dG = float(grade_pct - grade_ref)
        grade_factor = 1.0 - (grade_coef_pct_per_pct / 100.0) * dG

        # Eviter division par 0 / facteurs absurdes
        eps = 1e-6
        temp_factor = max(eps, temp_factor)
        grade_factor = max(eps, grade_factor)

        v_eq = v_kmh_raw / (temp_factor * grade_factor)
        if not math.isfinite(v_eq) or v_eq <= 0:
            return None
        return float(v_eq)

    # --------- CARTES TESTS EN GRILLE (2 par ligne) ---------
    n = st.session_state.nb_tests
    indices = list(range(1, n + 1))
    cols = st.columns(2)

    for idx, i in enumerate(indices):
        col = cols[idx % 2]
        with col:
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.subheader(f"📌 Test {i}")

            uploaded = st.file_uploader(
                f"Fichier Test {i} (FIT, GPX, CSV, TCX)",
                type=ACCEPTED_TYPES,
                key=f"file_{i}"
            )
            test_date = st.date_input(f"📅 Date du test {i}", value=date.today(), key=f"date_{i}")

            show_fc = st.checkbox(f"☑️ FC (Test {i})", value=True, key=f"fc_{i}")
            show_pace = st.checkbox(f"☑️ Allure (Test {i})", value=False, key=f"pace_{i}")
            show_power = st.checkbox(f"☑️ Puissance (Test {i})", value=False, key=f"power_{i}")

            if uploaded:
                try:
                    df = load_activity(uploaded)
                except Exception as e:
                    st.error(f"Erreur dans le fichier du Test {i} : {e}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.dropna(subset=["timestamp"])

                lag = st.slider(
                    f"Correction du décalage capteur (s) — Test {i}",
                    0, 10, 0, key=f"lag_{i}"
                )
                df["timestamp"] = df["timestamp"] - pd.to_timedelta(lag, unit="s")

                df, window, total_dur, pauses = smooth_hr(df)
                st.caption(
                    f"Durée détectée : {total_dur:.1f}s • "
                    f"Lissage : {window}s • Pauses détectées : {pauses}"
                )

                col_start, col_end = st.columns(2)
                with col_start:
                    start_str = st.text_input(
                        f"Début (hh:mm:ss) — Test {i}",
                        value="0:00:00",
                        key=f"start_{i}"
                    )
                with col_end:
                    end_str = st.text_input(
                        f"Fin (hh:mm:ss) — Test {i}",
                        value="0:12:00",
                        key=f"end_{i}"
                    )

                try:
                    start_sec = parse_time_to_seconds(start_str)
                    end_sec = parse_time_to_seconds(end_str)
                except:
                    st.error("Format temps invalide.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                if end_sec <= start_sec:
                    st.error("Fin doit être > début")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                if end_sec > df["time_s"].max():
                    st.warning(f"⚠️ Fin > fichier ({df['time_s'].max():.0f}s). Limitation auto.")
                    end_sec = df["time_s"].max()

                segment = df[(df["time_s"] >= start_sec) & (df["time_s"] <= end_sec)]

                if len(segment) < 10:
                    st.warning("Segment trop court pour analyse.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                # ---- ANALYSE FC ----
                stats, drift_bpm, drift_pct = analyze_heart_rate(segment)

                # >>> CORRECTION v_kmh : calcul distance robuste + pente + correction température
                dist_m = segment_distance_m(segment)
                t_s = float(end_sec - start_sec)
                v_kmh = 3.6 * dist_m / t_s if t_s > 0 else 0.0

                grade_pct = segment_grade_percent(segment)
                dplus_m = segment_elevation_gain_m(segment)

                v_kmh_eq = apply_conditions_correction(
                    v_kmh_raw=v_kmh,
                    grade_pct=grade_pct,
                    temp_act=temp_act_c,
                    temp_ref=temp_ref_c,
                    temp_coef_pct_per_c=temp_coef_pct_per_c,
                    grade_ref=grade_ref_pct,
                    grade_coef_pct_per_pct=grade_coef_pct_per_pct
                )

                pace = format_pace_min_per_km(v_kmh)
                if pace:
                    pace_str = f"{int(pace[0])}:{int(pace[1]):02d} min/km"
                else:
                    pace_str = "–"

                pace_eq = format_pace_min_per_km(v_kmh_eq) if v_kmh_eq is not None else None
                if pace_eq:
                    pace_eq_str = f"{int(pace_eq[0])}:{int(pace_eq[1]):02d} min/km"
                else:
                    pace_eq_str = "–"

                # ---- CINÉTIQUE VITESSE ----
                d_v_kmh, d_v_pct = analyze_speed_kinetics(segment)

                df_table = pd.DataFrame({
                    "Métrique": [
                        "FC moyenne (bpm)", "FC max (bpm)",
                        "Dérive FC (bpm/min)", "Dérive FC (%/min)",
                        "Dérive vitesse (km/h/min)", "Dérive vitesse (%/min)",
                        "Durée segment (s)", "Distance (m)",
                        "Vitesse (km/h)", "Allure (min/km)",
                        "Pente moyenne (%)",
                        "D+ (m)",
                        "Température (°C)",
                        "Vitesse équivalente (km/h)",
                        "Allure équivalente (min/km)"
                    ],
                    "Valeur": [
                        stats["FC moyenne (bpm)"], stats["FC max (bpm)"],
                        drift_bpm, drift_pct,
                        d_v_kmh, d_v_pct,
                        t_s, round(dist_m, 1),
                        round(v_kmh, 2), pace_str,
                        (round(grade_pct, 3) if grade_pct is not None else None),
                        round(dplus_m, 1),
                        float(temp_act_c),
                        (round(v_kmh_eq, 2) if v_kmh_eq is not None else None),
                        pace_eq_str
                    ]
                })
                st.dataframe(df_table, hide_index=True, use_container_width=True)

                fig, ax = plt.subplots(figsize=(9, 4.6))
                plot_multi_signals(
                    ax, segment, t0=start_sec, who=f"T{i}",
                    show_fc=show_fc,
                    show_pace=show_pace and (get_speed_col(segment) is not None),
                    show_power=show_power and ("power_smooth" in segment.columns),
                    linewidth=1.9
                )
                ax.set_title(f"Cinétique — Test {i} ({test_date})")
                ax.set_xlabel("Temps segment (s)")
                ax.grid(True, alpha=0.2)

                handles, labels = [], []
                for a in fig.axes:
                    h, l = a.get_legend_handles_labels()
                    handles += h; labels += l
                if handles:
                    ax.legend(handles, labels, fontsize=8, loc="upper left", frameon=False)

                st.pyplot(fig)

                tests_data.append({
                    "i": i,
                    "df": df,
                    "segment": segment,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "stats": stats,
                    "drift_bpm": drift_bpm,
                    "drift_pct": drift_pct,
                    "d_v_kmh": d_v_kmh,
                    "d_v_pct": d_v_pct,
                    "dist_m": dist_m,
                    "t_s": t_s,
                    "v_kmh": v_kmh,
                    "v_kmh_eq": v_kmh_eq,
                    "grade_pct": grade_pct,
                    "dplus_m": dplus_m,
                    "pace_str": pace_str,
                    "pace_eq_str": pace_eq_str,
                    "date": test_date,
                })

            st.markdown('</div>', unsafe_allow_html=True)

        if idx % 2 == 1 and idx < len(indices) - 1:
            cols = st.columns(2)

    # ============================================================
    # =============== GRAPHIQUE COMBINÉ DES TESTS =================
    # ============================================================
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("📊 Graphique combiné — FC / Allure / Puissance")

    show_c_fc = st.checkbox("☑️ FC", True, key="comb_fc")
    show_c_pace = st.checkbox("☑️ Allure", False, key="comb_pace")
    show_c_power = st.checkbox("☑️ Puissance", False, key="comb_power")

    if len(tests_data) > 0:
        figC, axC = plt.subplots(figsize=(10, 5))

        for t in tests_data:
            seg = t["segment"]
            t0 = seg["time_s"].iloc[0]

            plot_multi_signals(
                axC, seg, t0=t0, who=f"T{t['i']}",
                show_fc=show_c_fc,
                show_pace=show_c_pace and (get_speed_col(seg) is not None),
                show_power=show_c_power and ("power_smooth" in seg.columns)
            )

        axC.set_xlabel("Temps segment (s)")
        axC.set_title("Superposition des cinétiques")
        axC.grid(True, alpha=0.15)

        handles, labels = [], []
        for a in figC.axes:
            h, l = a.get_legend_handles_labels()
            handles += h; labels += l
        if handles:
            axC.legend(handles, labels, fontsize=8, loc="upper left", frameon=False)

        st.pyplot(figC)

    st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # ===================== VITESSE CRITIQUE ======================
    # ============================================================
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Modèle Hyperbolique — Vitesse Critique (VC)")

    # >>> AJOUT : utiliser la vitesse équivalente si disponible pour la calibration
    valid_tests = [t for t in tests_data if t["dist_m"] > 0 and t["t_s"] > 0]

    if len(valid_tests) >= 2:
        D = np.array([t["dist_m"] for t in valid_tests])
        T = np.array([t["t_s"] for t in valid_tests])

        slope, intercept = np.polyfit(T, D, 1)
        VC_m_s = slope
        D_prime = float(intercept)
        VC_kmh = VC_m_s * 3.6

        if VC_kmh > 0:
            pace_min_km = 60.0 / VC_kmh
            total_pace_s = int(round(pace_min_km * 60))
            pm = total_pace_s // 60
            ps = total_pace_s % 60
            vc_pace_str = f"{pm}:{ps:02d} min/km"
        else:
            vc_pace_str = "–"

        st.success(
            f"**VC = {VC_kmh:.2f} km/h**  \n"
            f"➡️ soit **{vc_pace_str}**  \n"
            f"**D′ = {D_prime:.1f} m**  \n"
            f"(Régression hyperbolique sur {len(valid_tests)} tests)"
        )
    else:
        st.info("Il faut au moins deux tests valides (distance & durée) pour calculer la VC.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ===================== MODÈLE POWER LAW ======================
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("📈 Modèle Power Law (T = A · V^{-k})")

    if len(valid_tests) >= 2:
        V = np.array([t["dist_m"] / t["t_s"] for t in valid_tests if t["t_s"] > 0])
        TT = np.array([t["t_s"] for t in valid_tests if t["t_s"] > 0])

        positive_mask = V > 0
        V = V[positive_mask]
        TT = TT[positive_mask]

        if len(V) >= 2:
            X = np.log(V)
            Y = np.log(TT)

            slope_pl, intercept_pl = np.polyfit(X, Y, 1)
            k_log = -slope_pl
            A = float(np.exp(intercept_pl))

            st.write(f"**k = {k_log:.3f}**, **A = {A:.2f}** (modèle Power Law)")
        else:
            st.info("Pas assez de vitesses positives pour ajuster le modèle Power Law.")
    else:
        st.info("Au moins 2 tests requis pour le modèle Power Law.")

    st.markdown('</div>', unsafe_allow_html=True)

    # =============== TABLEAU PRÉDICTIF (CHOIX DU MODÈLE) =========
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("📊 Prédictions selon intensité")

    model_choice = st.radio(
        "Choisir le modèle utilisé pour le tableau :",
        ("Modèle Power Law (<100% VC)", "Modèle D′ (>100% VC)"),
        index=0,
        horizontal=True
    )

    VC_ms = VC_kmh / 3.6 if (VC_kmh is not None and VC_kmh > 0) else None

    if model_choice.startswith("Modèle Power"):

        if VC_ms is not None and A is not None and k_log is not None:

            pourcentages = list(range(80, 100, 2))
            rows = []

            for p in pourcentages:
                v_kmh = VC_kmh * (p / 100.0)
                v_ms = v_kmh / 3.6
                if v_ms <= 0:
                    continue

                Tlim = A * (v_ms ** (-k_log))

                if Tlim <= 0 or not math.isfinite(Tlim):
                    continue

                m = int(Tlim // 60)
                s = int(Tlim % 60)
                T_str = f"{m}:{s:02d}"

                pace_min = 60.0 / v_kmh
                sec = int(round(pace_min * 60))
                pm, ps = sec // 60, sec % 60
                pace_str = f"{pm}:{ps:02d}"

                rows.append({
                    "% VC": f"{p}%",
                    "Modèle": "Power Law",
                    "Temps limite (mm:ss)": T_str,
                    "Allure (min/km)": pace_str
                })

            if rows:
                df_pred = pd.DataFrame(rows)
                st.dataframe(df_pred, hide_index=True, use_container_width=True)
            else:
                st.info("Aucune prédiction exploitable avec le modèle Power Law (paramètres invalides).")

        else:
            st.info("⚠️ Impossible : paramètres Power Law (A, k) ou VC non disponibles.")

    else:

        if VC_ms is not None and D_prime is not None and D_prime > 0:

            pourcentages = list(range(102, 132, 2))
            rows = []

            for p in pourcentages:
                v_kmh = VC_kmh * (p / 100.0)
                v_ms = v_kmh / 3.6

                denom = v_ms - VC_ms
                if denom <= 0:
                    continue

                Tlim = D_prime / denom

                if Tlim <= 0 or not math.isfinite(Tlim):
                    continue

                m = int(Tlim // 60)
                s = int(Tlim % 60)
                T_str = f"{m}:{s:02d}"

                pace_min = 60.0 / v_kmh
                sec = int(round(pace_min * 60))
                pm, ps = sec // 60, sec % 60
                pace_str = f"{pm}:{ps:02d}"

                rows.append({
                    "% VC": f"{p}%",
                    "Modèle": "D′",
                    "Temps limite (mm:ss)": T_str,
                    "Allure (min/km)": pace_str
                })

            if rows:
                df_pred = pd.DataFrame(rows)
                st.dataframe(df_pred, hide_index=True, use_container_width=True)
            else:
                st.info("Aucune prédiction exploitable avec le modèle D′ (paramètres invalides).")

        else:
            st.info("⚠️ Impossible : VC ou D′ non disponible pour le modèle D′.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # ====================== INDEX CINÉTIQUE ======================
    # ============================================================
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Index de Cinétique (sélection tests)")

    if len(tests_data) >= 2:
        test_names = [f"Test {t['i']}" for t in tests_data]

        colA_sel, colB_sel = st.columns(2)
        with colA_sel:
            sel_a = st.selectbox("Test court", test_names, key="ic_a")
        with colB_sel:
            sel_b = st.selectbox("Test long", test_names, key="ic_b")

        tA = tests_data[test_names.index(sel_a)]
        tB = tests_data[test_names.index(sel_b)]

        ic_val, unite, msg, _, reco = compute_index_cinetique(
            tA["drift_pct"], tB["drift_pct"], tA["drift_bpm"], tB["drift_bpm"]
        )

        if ic_val is not None:
            st.markdown(f"**IC = {ic_val*100:.1f}%** ({unite})")
            st.info(msg)
            st.markdown(f"**{reco['titre']}**")
            for s in reco["seances"]:
                st.markdown(f"• {s}")
        else:
            st.warning("Index non calculable avec ces deux tests.")
    else:
        st.info("Sélectionne au moins deux tests pour l'IC.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # ========================== EXPORT PDF ========================
    # ============================================================
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("📄 Export PDF — Rapport complet des tests")

    if st.button("Générer le rapport PDF", key="export_pdf_btn"):
        figs_export = []

        if len(tests_data) > 0:
            figG, axG = plt.subplots(figsize=(9, 5))

            show_fc_exp = True
            show_pace_exp = True
            show_power_exp = True

            for t in tests_data:
                seg = t["segment"]
                t0 = seg["time_s"].iloc[0]

                plot_multi_signals(
                    axG, seg, t0=t0, who=f"T{t['i']}",
                    show_fc=show_fc_exp,
                    show_pace=show_pace_exp and (get_speed_col(seg) is not None),
                    show_power=show_power_exp and ("power_smooth" in seg.columns)
                )

            axG.set_title("Comparaison des cinétiques — Tous les tests")
            axG.set_xlabel("Temps segment (s)")
            axG.grid(True, alpha=0.2)

            handles, labels = [], []
            for a in figG.axes:
                h, l = a.get_legend_handles_labels()
                handles += h; labels += l
            if handles:
                axG.legend(handles, labels, fontsize=7, loc="upper left")

            figs_export.append(figG)

        for t in tests_data:
            fig_i, ax_i = plt.subplots(figsize=(9, 4.8))

            seg = t["segment"]
            t0 = seg["time_s"].iloc[0]

            plot_multi_signals(
                ax_i, seg, t0=t0, who=f"T{t['i']}",
                show_fc=True,
                show_pace=(get_speed_col(seg) is not None),
                show_power=("power_smooth" in seg.columns)
            )

            ax_i.set_title(f"Test {t['i']} — {t['date']}")
            ax_i.set_xlabel("Temps segment (s)")
            ax_i.grid(True, alpha=0.2)

            figs_export.append(fig_i)

        pdf_buffer = fig_to_pdf_bytes(figs_export)

        st.download_button(
            label="📥 Télécharger le rapport PDF",
            data=pdf_buffer,
            file_name=f"rapport_tests_endurance_{date.today()}.pdf",
            mime="application/pdf"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# ONGLET 2 : ANALYSE ENTRAÎNEMENT (1 séance + intervalles + cinétiques)
# ---------------------------------------------------------------------
with tabs[1]:
    st.session_state.active_tab = "training"
    st.header("⚙️ Analyse entraînement (1 séance + intervalles + cinétiques)")

    if "training_session" not in st.session_state:
        st.session_state.training_session = None
    if "training_intervals" not in st.session_state:
        st.session_state.training_intervals = []

    # ---- IMPORT ----
    uploaded_file = st.file_uploader(
        "Importer un fichier d'entraînement (FIT, GPX, CSV, TCX)",
        type=ACCEPTED_TYPES,
        key="training_file"
    )

    if uploaded_file:
        try:
            df = load_activity(uploaded_file)
            df, window, dur, pauses = smooth_hr(df)
            st.session_state.training_session = (df, window, dur, pauses)
        except Exception as e:
            st.error(f"Erreur chargement séance : {e}")

    if st.session_state.training_session is None:
        st.info("Importe une séance pour commencer l’analyse.")
        st.stop()

    df, window, dur, pauses = st.session_state.training_session

    st.markdown(f"### 📂 Séance importée : **{uploaded_file.name}**")
    st.caption(f"Durée totale : {dur:.1f}s • Lissage : {window}s • Pauses détectées : {pauses}")

    # ---------------------------------------------------------------
    # 1) DÉFINITION DES INTERVALLES
    # ---------------------------------------------------------------
    st.markdown("## 📏 Définition des intervalles")

    for i, (start_s, end_s) in enumerate(st.session_state.training_intervals):
        c1, c2, c3 = st.columns([1, 1, 0.3])

        with c1:
            s_str = st.text_input(
                f"Début intervalle {i+1} (hh:mm:ss)",
                value=f"{int(start_s//60)}:{int(start_s%60):02d}",
                key=f"tr_int_start_{i}"
            )
        with c2:
            e_str = st.text_input(
                f"Fin intervalle {i+1}",
                value=f"{int(end_s//60)}:{int(end_s%60):02d}",
                key=f"tr_int_end_{i}"
            )
        with c3:
            if st.button("🗑️", key=f"tr_del_int_{i}"):
                st.session_state.training_intervals.pop(i)
                st.rerun()

        try:
            s_sec = parse_time_to_seconds(s_str)
            e_sec = parse_time_to_seconds(e_str)
            if e_sec > s_sec:
                st.session_state.training_intervals[i] = (s_sec, e_sec)
        except:
            st.warning(f"⛔ Format invalide intervalle {i+1}")

    if st.button("➕ Ajouter un intervalle", key="tr_add_int"):
        st.session_state.training_intervals.append((0, 300))
        st.rerun()

    # ---------------------------------------------------------------
    # 2) ANALYSE DES INTERVALLES
    # ---------------------------------------------------------------
    st.markdown("## 🔍 Analyse des intervalles")

    interval_segments = []

    for i, (s_sec, e_sec) in enumerate(st.session_state.training_intervals):
        seg = df[(df["time_s"] >= s_sec) & (df["time_s"] <= e_sec)]
        if seg.empty:
            continue

        interval_segments.append((i+1, seg, s_sec, e_sec))

        # --- FC ---
        stats, d_bpm, d_pct = analyze_heart_rate(seg)

        # --- Distance, vitesse, allure ---
        dist_m = segment_distance_m(seg)
        t_s = e_sec - s_sec
        v_kmh = 3.6 * dist_m / t_s if t_s > 0 else 0
        pace = format_pace_min_per_km(v_kmh)
        pace_str = f"{pace[0]}:{pace[1]:02d} min/km" if pace else "–"

        # --- CINÉTIQUE VITESSE ---
        d_v_kmh, d_v_pct = analyze_speed_kinetics(seg)

        # -------------------------
        # TABLEAU
        # -------------------------
        st.markdown(f"### Intervalle {i+1} ({s_sec:.0f}s → {e_sec:.0f}s)")
        st.dataframe(pd.DataFrame({
            "Métrique": [
                "FC moyenne",
                "Dérive FC (bpm/min)",
                "Dérive FC (%/min)",
                "Dérive vitesse (km/h/min)",
                "Dérive vitesse (%/min)",
                "Durée (s)",
                "Distance (m)",
                "Vitesse (km/h)",
                "Allure"
            ],
            "Valeur": [
                stats["FC moyenne (bpm)"],
                d_bpm,
                d_pct,
                d_v_kmh,
                d_v_pct,
                t_s,
                round(dist_m, 1),
                round(v_kmh, 2),
                pace_str
            ]
        }), hide_index=True, use_container_width=True)

        # -------------------------
        # 4) GRAPHIQUE SEGMENT
        # -------------------------
        fig, ax = plt.subplots(figsize=(9, 4.2))
        plot_multi_signals(
            ax, seg, t0=s_sec, who=f"Int{i+1}",
            show_fc=True,
            show_pace=("speed_smooth" in seg.columns),
            show_power=("power_smooth" in seg.columns)
        )
        ax.set_title(f"Cinétique — Intervalle {i+1}")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig)

    # ---------------------------------------------------------------
    # 5) GRAPHIQUE COMBINÉ (intervalles superposés)
    # ---------------------------------------------------------------
    if interval_segments:
        st.markdown("## 📊 Graphique combiné — tous les intervalles superposés")
        show_fc = st.checkbox("☑ FC", True, key="comb_fc_training_v2")
        show_pace = st.checkbox("☑ Allure", False, key="comb_pace_training_v2")
        show_power = st.checkbox("☑ Puissance", False, key="comb_pow_training_v2")

        figC, axC = plt.subplots(figsize=(10, 4.8))
        for idx, seg, s0, s1 in interval_segments:
            plot_multi_signals(
                axC, seg, t0=s0,
                who=f"Int{idx}",
                show_fc=show_fc,
                show_pace=show_pace and ("speed_smooth" in seg.columns),
                show_power=show_power and ("power_smooth" in seg.columns)
            )

        axC.set_title("Cinétique combinée — Intervalles superposés")
        axC.set_xlabel("Temps segment (s)")
        axC.grid(True, alpha=0.25)

        handles, labels = [], []
        for a in figC.axes:
            h, l = a.get_legend_handles_labels()
            handles += h; labels += l
        if handles:
            axC.legend(handles, labels, fontsize=8, loc="upper left")

        st.pyplot(figC)
