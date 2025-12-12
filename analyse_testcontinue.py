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
from datetime import date, datetime, timedelta
import xml.etree.ElementTree as ET
import gpxpy
import requests

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

# ========================= SIDEBAR — COEFFICIENTS (CODE 2) ==============================
st.sidebar.header("⚙️ Coefficients (comme code 2)")

use_elev_coeff = st.sidebar.checkbox("Activer coefficients montée/descente 🎢", value=True)
if use_elev_coeff:
    k_up = st.sidebar.number_input("Coefficient montée (k_up)", value=1.040, format="%.3f", step=0.001)
    k_down = st.sidebar.number_input("Coefficient descente (k_down)", value=0.996, format="%.3f", step=0.001)
else:
    k_up = 1.0
    k_down = 1.0

use_temp_coeff = st.sidebar.checkbox("Activer coefficients température 🌡️", value=True)
if use_temp_coeff:
    k_temp_hot = st.sidebar.number_input("Sensibilité chaude (k_temp_hot)", value=0.0020, format="%.4f", step=0.0005)
    k_temp_cold = st.sidebar.number_input("Sensibilité froide (k_temp_cold)", value=0.0020, format="%.4f", step=0.0005)
    opt_temp = st.sidebar.number_input("Température optimale (°C)", value=12.0, format="%.1f", step=0.5)
else:
    k_temp_hot = 0.0
    k_temp_cold = 0.0
    opt_temp = 12.0

st.sidebar.caption("👉 Température récupérée automatiquement via Open-Meteo archive si lat/lon + timestamps dispo.\n"
                   "Sinon fallback sur une température manuelle.")

manual_temp_fallback = st.sidebar.number_input("Temp fallback (°C) si météo impossible", value=float(opt_temp), step=0.5)

# ========================= MÉTÉO — OPEN-METEO ARCHIVE (COMME CODE 2) ==============================
@st.cache_data(show_spinner=False)
def get_weather_openmeteo_day(lat, lon, date_obj):
    """
    Récupère TOUTE la journée météo (24 valeurs horaires) en un seul appel.
    Retourne times(list datetime), temps(list), winds(list), hums(list)
    """
    try:
        date_str = date_obj.strftime("%Y-%m-%d")
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={date_str}&end_date={date_str}"
            "&hourly=temperature_2m,relativehumidity_2m,wind_speed_10m"
            "&timezone=UTC"
        )
        r = requests.get(url, timeout=20)
        data = r.json()
        if "hourly" not in data:
            return None

        times = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
        temps = data["hourly"]["temperature_2m"]
        winds = data["hourly"]["wind_speed_10m"]
        hums  = data["hourly"]["relativehumidity_2m"]
        return times, temps, winds, hums
    except Exception:
        return None

def _to_utc_naive(dt):
    if dt is None:
        return None
    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt, errors="coerce")
            if pd.isna(dt):
                return None
            dt = dt.to_pydatetime()
        if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
            # Convert to UTC then drop tz
            return dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)  # type: ignore
        return dt.replace(tzinfo=None)
    except Exception:
        return None

def get_avg_weather_for_period(lat, lon, start_dt, end_dt):
    """
    Même logique que ton code 2:
      - élargit si < 5 minutes
      - charge la journée
      - moyenne sur l’intervalle
      - fallback : heure la plus proche
    """
    try:
        if start_dt is None or end_dt is None:
            return None, None, None

        start_dt = _to_utc_naive(start_dt)
        end_dt = _to_utc_naive(end_dt)
        if start_dt is None or end_dt is None:
            return None, None, None

        if (end_dt - start_dt).total_seconds() < 300:
            start_dt -= timedelta(minutes=2)
            end_dt += timedelta(minutes=2)

        meteo_day = get_weather_openmeteo_day(lat, lon, start_dt.date())
        if not meteo_day:
            return None, None, None

        times, temps, winds, hums = meteo_day

        selT = [T for t, T in zip(times, temps) if start_dt <= t <= end_dt]
        selW = [W for t, W in zip(times, winds) if start_dt <= t <= end_dt]
        selH = [H for t, H in zip(times, hums)  if start_dt <= t <= end_dt]

        if not selT:
            closest_index = min(range(len(times)), key=lambda i: abs(times[i] - start_dt))
            return float(temps[closest_index]), float(winds[closest_index]), float(hums[closest_index])

        return float(np.mean(selT)), float(np.mean(selW)), float(np.mean(selH))
    except Exception:
        return None, None, None

def infer_segment_weather(df_seg):
    """
    Déduit température moyenne segment via Open-Meteo archive.
    Besoin: lat/lon + timestamps dans le segment.
    """
    try:
        if df_seg is None or df_seg.empty:
            return None, None, None

        lc = {c.lower(): c for c in df_seg.columns}
        if not ("lat" in lc and "lon" in lc and "timestamp" in lc):
            return None, None, None

        latc = pd.to_numeric(df_seg[lc["lat"]], errors="coerce")
        lonc = pd.to_numeric(df_seg[lc["lon"]], errors="coerce")
        if latc.notna().sum() < 5 or lonc.notna().sum() < 5:
            return None, None, None

        lat = float(latc.dropna().iloc[len(latc.dropna())//2])
        lon = float(lonc.dropna().iloc[len(lonc.dropna())//2])

        t0 = pd.to_datetime(df_seg[lc["timestamp"]], errors="coerce").dropna()
        if t0.empty:
            return None, None, None
        start_dt = t0.iloc[0].to_pydatetime()
        end_dt = t0.iloc[-1].to_pydatetime()

        return get_avg_weather_for_period(lat, lon, start_dt, end_dt)
    except Exception:
        return None, None, None

# ========================= MODÈLE TEMP (NON LINÉAIRE) — CODE 2 ==============================
def temp_multiplier_nonlin(temp, opt_temp=12.0, k_hot=0.002, k_cold=0.002):
    """
    Identique à code 2:
    >opt -> 1 + k_hot*(temp-opt)
    <opt -> 1 + k_cold*(opt-temp)
    """
    try:
        if temp is None:
            return 1.0
        diff = float(temp) - float(opt_temp)
        if diff > 0:
            mult = 1.0 + float(k_hot) * diff
        else:
            mult = 1.0 + float(k_cold) * (-diff)
        return max(0.1, float(mult))
    except Exception:
        return 1.0

def factor_elevation(dist_m, d_up, d_down, k_up=1.040, k_down=0.996):
    """
    Identique esprit code 2:
    factor = 1 + (k_up-1)*(D_up/dist) + (1-k_down)*(D_down/dist)
    """
    try:
        dist_m = float(dist_m) if dist_m and dist_m > 0 else 1000.0
        up_factor = (float(k_up) - 1.0) * (float(d_up) / dist_m)
        down_factor = (1.0 - float(k_down)) * (float(d_down) / dist_m)
        factor = 1.0 + up_factor + down_factor
        return max(0.01, float(factor))
    except Exception:
        return 1.0

def recalibrate_time_to_ideal(time_s_raw, dist_m, d_up, d_down, temp_real):
    """
    Recalibre un temps brut vers conditions idéales (plat + opt_temp),
    exactement dans l’esprit de recalibrate_ref_to_ideal du code 2.
    """
    if time_s_raw is None or not math.isfinite(time_s_raw) or time_s_raw <= 0:
        return None

    # 1) retirer élévation
    fe = factor_elevation(dist_m, d_up, d_down, k_up=k_up, k_down=k_down)
    t_no_elev = float(time_s_raw) / fe

    # 2) retirer température réelle
    mult_real = temp_multiplier_nonlin(temp_real, opt_temp=opt_temp, k_hot=k_temp_hot, k_cold=k_temp_cold)
    t_no_temp = t_no_elev / (mult_real if mult_real != 0 else 1.0)

    # 3) appliquer temp optimale (mult_opt = 1.0 par construction)
    mult_opt = temp_multiplier_nonlin(opt_temp, opt_temp=opt_temp, k_hot=k_temp_hot, k_cold=k_temp_cold)
    t_ideal = t_no_temp * mult_opt
    return max(0.0, float(t_ideal))

# ========================= LECTURE FICHIERS ==============================

def _fit_semi_to_deg(x):
    try:
        return float(x) * (180.0 / (2**31))
    except Exception:
        return None

def load_activity(file):
    """Charge un fichier FIT, GPX, CSV ou TCX. Harmonise: timestamp, heart_rate, lat/lon, alt, distance, speed, power."""
    name = file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(file)

    elif name.endswith(".fit"):
        data = []
        try:
            file.seek(0)
            with fitdecode.FitReader(file) as fit:
                for frame in fit:
                    if isinstance(frame, fitdecode.records.FitDataMessage) and frame.name == "record":
                        row = {f.name: f.value for f in frame.fields}
                        data.append(row)
            df = pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Erreur FIT : {e}")

        # Harmonisation lat/lon FIT (position_lat/position_long -> degrés)
        if "position_lat" in df.columns and "lat" not in df.columns:
            df["lat"] = df["position_lat"].apply(_fit_semi_to_deg)
        if "position_long" in df.columns and "lon" not in df.columns:
            df["lon"] = df["position_long"].apply(_fit_semi_to_deg)

        # Harmonisation altitude FIT
        if "enhanced_altitude" in df.columns and "alt" not in df.columns:
            df["alt"] = df["enhanced_altitude"]
        elif "altitude" in df.columns and "alt" not in df.columns:
            df["alt"] = df["altitude"]

        # Harmonisation HR FIT
        if "heart_rate" not in df.columns and "hr" in df.columns:
            df["heart_rate"] = df["hr"]

    elif name.endswith(".gpx"):
        file.seek(0)
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
            file.seek(0)
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
                spd = tp.find(".//{*}Extensions//{*}Speed")

                data.append({
                    "timestamp": t.text if t is not None else None,
                    "heart_rate": float(hr.text) if hr is not None else None,
                    "distance": float(dist.text) if dist is not None else None,
                    "alt": float(alt.text) if alt is not None else None,
                    "power": float(powv.text) if powv is not None else None,
                    "lat": float(lat.text) if lat is not None else None,
                    "lon": float(lon.text) if lon is not None else None,
                    "speed": float(spd.text) if spd is not None else None,
                })

            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError("TCX vide")
        except Exception as e:
            raise ValueError(f"Erreur TCX : {e}")

    else:
        raise ValueError("Format non supporté (.fit, .gpx, .csv, .tcx uniquement)")

    # Harmonisation timestamp
    if "timestamp" not in df.columns:
        for c in df.columns:
            if "time" in c.lower():
                df = df.rename(columns={c: "timestamp"})
                break

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Nettoyage numeric
    for c in ["heart_rate","speed","enhanced_speed","power","distance","lat","lon","alt"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "heart_rate" not in df.columns or df["heart_rate"].dropna().empty:
        raise ValueError("Pas de FC détectée")

    df = df.dropna(subset=["heart_rate"]).reset_index(drop=True)
    return df

# ========================= OUTILS CALCUL ==============================

def get_speed_col(df):
    if "speed_smooth" in df.columns:
        return "speed_smooth"
    if "enhanced_speed" in df.columns: return "enhanced_speed"
    if "speed" in df.columns: return "speed"
    return None

# ------------------------------------------------------------
# Lissage cardio/vitesse/puissance
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
    total_dur = float(df["time_s"].iloc[-1]) if not df.empty else 0.0

    if total_dur < 360:
        window_sec = 5
    elif total_dur < 900:
        window_sec = 10
    else:
        window_sec = 20

    step = np.median(np.diff(df["time_s"])) if len(df) > 3 else 1.0
    if step <= 0 or np.isnan(step):
        step = 1
    window_size = max(1, int(window_sec / step))

    df["hr_smooth"] = df[hr_col].rolling(window_size, min_periods=1).mean()

    sp = None
    if "enhanced_speed" in df.columns:
        sp = "enhanced_speed"
    elif "speed" in df.columns:
        sp = "speed"

    if sp:
        df["speed_smooth"] = df[sp].rolling(window_size, min_periods=1).mean()

    if "power" in df.columns:
        df["power_smooth"] = df["power"].rolling(window_size, min_periods=1).mean()

    pauses = int((df["delta_t"] > 2 * median_step).sum())
    return df, window_sec, total_dur, pauses

# ------------------------------------------------------------
# Analyse FC (dérive)
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
# Distance segment robuste
# ------------------------------------------------------------
def haversine_dist_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def segment_distance_m(df_seg):
    if df_seg is None or df_seg.empty or len(df_seg) < 2:
        return 0.0

    # 1) distance cumulée si dispo
    if "distance" in df_seg.columns and df_seg["distance"].dropna().shape[0] >= 2:
        d0 = float(df_seg["distance"].iloc[0])
        d1 = float(df_seg["distance"].iloc[-1])
        if np.isfinite(d0) and np.isfinite(d1):
            return max(0.0, d1 - d0)

    # 2) speed * dt si dispo
    sp = None
    if "enhanced_speed" in df_seg.columns:
        sp = "enhanced_speed"
    elif "speed" in df_seg.columns:
        sp = "speed"
    if sp is not None and "delta_t" in df_seg.columns:
        dist = float(np.nansum(df_seg[sp].fillna(0).values * df_seg["delta_t"].fillna(0).values))
        if dist > 0:
            return dist

    # 3) lat/lon
    lc = {c.lower(): c for c in df_seg.columns}
    if "lat" in lc and "lon" in lc:
        lats = df_seg[lc["lat"]].astype(float).values
        lons = df_seg[lc["lon"]].astype(float).values
        dist = 0.0
        for i in range(1, len(df_seg)):
            if all(np.isfinite([lats[i-1], lats[i], lons[i-1], lons[i]])):
                dist += haversine_dist_m(lats[i-1], lons[i-1], lats[i], lons[i])
        return float(max(0.0, dist))

    return 0.0

def compute_dplus_dminus(df_seg):
    if df_seg is None or df_seg.empty or "alt" not in df_seg.columns:
        return 0.0, 0.0

    alt = pd.to_numeric(df_seg["alt"], errors="coerce").astype(float)
    alt = alt.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
    alt = alt.rolling(5, min_periods=1, center=True).median()

    d = alt.diff().fillna(0.0)
    d_up = float(d[d > 0].sum())
    d_down = float((-d[d < 0]).sum())
    if not math.isfinite(d_up): d_up = 0.0
    if not math.isfinite(d_down): d_down = 0.0
    return max(0.0, d_up), max(0.0, d_down)

# ------------------------------------------------------------
# Allure
# ------------------------------------------------------------
def format_pace_min_per_km(v_kmh):
    if v_kmh is None or v_kmh <= 0 or not math.isfinite(v_kmh):
        return None
    min_per_km = 60.0 / v_kmh
    total_seconds = int(round(min_per_km * 60.0))
    return total_seconds // 60, total_seconds % 60, min_per_km

# ------------------------------------------------------------
# Export PDF
# ------------------------------------------------------------
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
# Cinétique vitesse
# ------------------------------------------------------------
def analyze_speed_kinetics(df):
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
    else:
        if drift_short_bpm is None or drift_long_bpm is None or drift_short_bpm == 0:
            return None, None, "Index non calculable (dérives indisponibles).", None, None
        IC = 1.0 - (drift_long_bpm / drift_short_bpm)
        unite = "bpm/min"

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

def compute_pace_series_from_speed_smooth(df):
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
    if who.startswith("T1"):
        c_fc, c_pace, c_pow = COLOR_RED_T1, COLOR_BLUE_T1, COLOR_ORANGE_T1
    elif who.startswith("T2"):
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
        pace_series = compute_pace_series_from_speed_smooth(df)
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

    if "nb_tests" not in st.session_state:
        st.session_state.nb_tests = 2

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

    tests_data = []
    VC_kmh_raw = None
    VC_kmh_ideal = None
    D_prime_raw = None
    D_prime_ideal = None

    A_raw = None
    k_log_raw = None
    A_ideal = None
    k_log_ideal = None

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
                except Exception:
                    st.error("Format temps invalide.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                if end_sec <= start_sec:
                    st.error("Fin doit être > début")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                if end_sec > df["time_s"].max():
                    st.warning(f"⚠️ Fin > fichier ({df['time_s'].max():.0f}s). Limitation auto.")
                    end_sec = float(df["time_s"].max())

                segment = df[(df["time_s"] >= start_sec) & (df["time_s"] <= end_sec)]
                if len(segment) < 10:
                    st.warning("Segment trop court pour analyse.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                # ---- ANALYSE FC ----
                stats, drift_bpm, drift_pct = analyze_heart_rate(segment)

                # ---- DIST / VITESSE / ALLURE BRUT ----
                dist_m = segment_distance_m(segment)
                t_s = float(end_sec - start_sec)
                v_kmh = 3.6 * dist_m / t_s if t_s > 0 else 0.0
                pace = format_pace_min_per_km(v_kmh)
                pace_str = f"{int(pace[0])}:{int(pace[1]):02d} min/km" if pace else "–"

                # ---- D+ / D- ----
                d_up, d_down = compute_dplus_dminus(segment)

                # ---- TEMPÉRATURE OPEN-METEO (segment) ----
                avgT, avgW, avgH = infer_segment_weather(segment)
                if avgT is None:
                    avgT = float(manual_temp_fallback)

                # ---- RECALIBRAGE CONDITIONS IDÉALES (temps et vitesse) ----
                t_ideal = recalibrate_time_to_ideal(t_s, dist_m, d_up, d_down, avgT) if (USE_RECAL := True) else None
                if t_ideal is None or t_ideal <= 0:
                    v_kmh_ideal = None
                    pace_ideal_str = "–"
                else:
                    v_kmh_ideal = 3.6 * dist_m / t_ideal
                    pace_ideal = format_pace_min_per_km(v_kmh_ideal)
                    pace_ideal_str = f"{int(pace_ideal[0])}:{int(pace_ideal[1]):02d} min/km" if pace_ideal else "–"

                # ---- CINÉTIQUE VITESSE ----
                d_v_kmh, d_v_pct = analyze_speed_kinetics(segment)

                # ---- TABLEAU ----
                df_table = pd.DataFrame({
                    "Métrique": [
                        "FC moyenne (bpm)", "FC max (bpm)",
                        "Dérive FC (bpm/min)", "Dérive FC (%/min)",
                        "Dérive vitesse (km/h/min)", "Dérive vitesse (%/min)",
                        "Durée segment BRUT (s)",
                        "Distance (m)",
                        "Vitesse BRUT (km/h)", "Allure BRUT (min/km)",
                        "D+ (m)", "D- (m)",
                        "Temp moyenne (°C)",
                        "Durée IDEAL (s)",
                        "Vitesse IDEAL (km/h)",
                        "Allure IDEAL (min/km)",
                    ],
                    "Valeur": [
                        stats["FC moyenne (bpm)"], stats["FC max (bpm)"],
                        drift_bpm, drift_pct,
                        d_v_kmh, d_v_pct,
                        round(t_s, 1),
                        round(dist_m, 1),
                        round(v_kmh, 2), pace_str,
                        round(d_up, 1), round(d_down, 1),
                        round(float(avgT), 2) if avgT is not None else None,
                        round(t_ideal, 1) if t_ideal is not None else None,
                        round(v_kmh_ideal, 2) if v_kmh_ideal is not None else None,
                        pace_ideal_str
                    ]
                })
                st.dataframe(df_table, hide_index=True, use_container_width=True)

                # ---- GRAPHIQUE ----
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
                    "t_s_raw": t_s,
                    "t_s_ideal": t_ideal,
                    "v_kmh_raw": v_kmh,
                    "v_kmh_ideal": v_kmh_ideal,
                    "pace_raw": pace_str,
                    "pace_ideal": pace_ideal_str,
                    "d_up": d_up,
                    "d_down": d_down,
                    "avg_temp": avgT,
                    "date": test_date,
                })

            st.markdown('</div>', unsafe_allow_html=True)

        if idx % 2 == 1 and idx < len(indices) - 1:
            cols = st.columns(2)

    # ============================================================
    # GRAPHIQUE COMBINÉ
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
    # VC BRUTE + VC CONDITIONS IDÉALES
    # ============================================================
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Modèle Hyperbolique — Vitesse Critique (VC)")

    valid_raw = [t for t in tests_data if t["dist_m"] > 0 and t["t_s_raw"] and t["t_s_raw"] > 0]
    valid_ideal = [t for t in tests_data if t["dist_m"] > 0 and t["t_s_ideal"] and t["t_s_ideal"] > 0]

    # --- VC BRUTE ---
    if len(valid_raw) >= 2:
        D = np.array([t["dist_m"] for t in valid_raw])
        T = np.array([t["t_s_raw"] for t in valid_raw])
        slope, intercept = np.polyfit(T, D, 1)
        VC_m_s_raw = float(slope)
        D_prime_raw = float(intercept)
        VC_kmh_raw = VC_m_s_raw * 3.6
    else:
        VC_kmh_raw = None
        D_prime_raw = None

    # --- VC IDEAL ---
    if len(valid_ideal) >= 2:
        D2 = np.array([t["dist_m"] for t in valid_ideal])
        T2 = np.array([t["t_s_ideal"] for t in valid_ideal])
        slope2, intercept2 = np.polyfit(T2, D2, 1)
        VC_m_s_ideal = float(slope2)
        D_prime_ideal = float(intercept2)
        VC_kmh_ideal = VC_m_s_ideal * 3.6
    else:
        VC_kmh_ideal = None
        D_prime_ideal = None

    # Affichage
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📌 VC BRUTE")
        if VC_kmh_raw:
            pace_min_km = 60.0 / VC_kmh_raw
            total_pace_s = int(round(pace_min_km * 60))
            pm = total_pace_s // 60
            ps = total_pace_s % 60
            st.success(
                f"**VC brute = {VC_kmh_raw:.2f} km/h**  \n"
                f"➡️ **{pm}:{ps:02d} min/km**  \n"
                f"**D′ brut = {D_prime_raw:.1f} m**"
            )
        else:
            st.info("Au moins 2 tests valides requis (BRUT).")

    with c2:
        st.markdown("### 🔧 VC CONDITIONS IDÉALES")
        if VC_kmh_ideal:
            pace_min_km = 60.0 / VC_kmh_ideal
            total_pace_s = int(round(pace_min_km * 60))
            pm = total_pace_s // 60
            ps = total_pace_s % 60
            st.success(
                f"**VC idéale = {VC_kmh_ideal:.2f} km/h**  \n"
                f"➡️ **{pm}:{ps:02d} min/km**  \n"
                f"**D′ idéal = {D_prime_ideal:.1f} m**"
            )
        else:
            st.info("Au moins 2 tests valides requis (IDÉAL).")

    st.caption("VC idéale : recalibrage (plat + 12°C) basé sur D+ / D- et température Open-Meteo, comme ton code 2.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # POWER LAW BRUT + IDEAL
    # ============================================================
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("📈 Modèle Power Law (T = A · V^{-k}) — Brut vs Idéal")

    # --- Fit brut ---
    if len(valid_raw) >= 2:
        V = np.array([t["dist_m"] / t["t_s_raw"] for t in valid_raw if t["t_s_raw"] > 0])
        TT = np.array([t["t_s_raw"] for t in valid_raw if t["t_s_raw"] > 0])
        msk = V > 0
        V, TT = V[msk], TT[msk]
        if len(V) >= 2:
            X = np.log(V)
            Y = np.log(TT)
            slope_pl, intercept_pl = np.polyfit(X, Y, 1)
            k_log_raw = float(-slope_pl)
            A_raw = float(np.exp(intercept_pl))
        else:
            k_log_raw, A_raw = None, None
    else:
        k_log_raw, A_raw = None, None

    # --- Fit idéal ---
    if len(valid_ideal) >= 2:
        V2 = np.array([t["dist_m"] / t["t_s_ideal"] for t in valid_ideal if t["t_s_ideal"] > 0])
        TT2 = np.array([t["t_s_ideal"] for t in valid_ideal if t["t_s_ideal"] > 0])
        msk2 = V2 > 0
        V2, TT2 = V2[msk2], TT2[msk2]
        if len(V2) >= 2:
            X2 = np.log(V2)
            Y2 = np.log(TT2)
            slope_pl2, intercept_pl2 = np.polyfit(X2, Y2, 1)
            k_log_ideal = float(-slope_pl2)
            A_ideal = float(np.exp(intercept_pl2))
        else:
            k_log_ideal, A_ideal = None, None
    else:
        k_log_ideal, A_ideal = None, None

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Brut")
        if k_log_raw is not None and A_raw is not None:
            st.write(f"**k = {k_log_raw:.3f}**, **A = {A_raw:.2f}**")
        else:
            st.info("Pas assez de données brut pour ajuster Power Law.")

    with c2:
        st.markdown("### Idéal (plat + 12°C)")
        if k_log_ideal is not None and A_ideal is not None:
            st.write(f"**k = {k_log_ideal:.3f}**, **A = {A_ideal:.2f}**")
        else:
            st.info("Pas assez de données idéal pour ajuster Power Law.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # INDEX CINÉTIQUE
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
    # EXPORT PDF
    # ============================================================
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("📄 Export PDF — Rapport complet des tests")

    if st.button("Générer le rapport PDF", key="export_pdf_btn"):
        figs_export = []

        if len(tests_data) > 0:
            figG, axG = plt.subplots(figsize=(9, 5))

            for t in tests_data:
                seg = t["segment"]
                t0 = seg["time_s"].iloc[0]
                plot_multi_signals(
                    axG, seg, t0=t0, who=f"T{t['i']}",
                    show_fc=True,
                    show_pace=(get_speed_col(seg) is not None),
                    show_power=("power_smooth" in seg.columns)
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

    uploaded_file = st.file_uploader(
        "Importer un fichier d'entraînement (FIT, GPX, CSV, TCX)",
        type=ACCEPTED_TYPES,
        key="training_file"
    )

    if uploaded_file:
        try:
            df = load_activity(uploaded_file)
            df, window, dur, pauses = smooth_hr(df)
            st.session_state.training_session = (df, window, dur, pauses, uploaded_file.name)
        except Exception as e:
            st.error(f"Erreur chargement séance : {e}")

    if st.session_state.training_session is None:
        st.info("Importe une séance pour commencer l’analyse.")
        st.stop()

    df, window, dur, pauses, fname = st.session_state.training_session

    st.markdown(f"### 📂 Séance importée : **{fname}**")
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
            s_sec = parse_time_to_seconds(s_str if ":" in s_str else f"0:{s_str}")
            e_sec = parse_time_to_seconds(e_str if ":" in e_str else f"0:{e_str}")
            if e_sec > s_sec:
                st.session_state.training_intervals[i] = (s_sec, e_sec)
        except Exception:
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

        # --- Distance/vitesse brut ---
        dist_m = segment_distance_m(seg)
        t_s = float(e_sec - s_sec)
        v_kmh = 3.6 * dist_m / t_s if t_s > 0 else 0.0
        pace = format_pace_min_per_km(v_kmh)
        pace_str = f"{pace[0]}:{pace[1]:02d} min/km" if pace else "–"

        # --- D+/D- ---
        d_up, d_down = compute_dplus_dminus(seg)

        # --- Temp (Open-Meteo) ---
        avgT, avgW, avgH = infer_segment_weather(seg)
        if avgT is None:
            avgT = float(manual_temp_fallback)

        # --- Recalibrage (temps/vitesse idéal) ---
        t_ideal = recalibrate_time_to_ideal(t_s, dist_m, d_up, d_down, avgT)
        if t_ideal and t_ideal > 0:
            v_kmh_ideal = 3.6 * dist_m / t_ideal
            pace_i = format_pace_min_per_km(v_kmh_ideal)
            pace_ideal_str = f"{pace_i[0]}:{pace_i[1]:02d} min/km" if pace_i else "–"
        else:
            v_kmh_ideal = None
            pace_ideal_str = "–"

        # --- Cinétique vitesse ---
        d_v_kmh, d_v_pct = analyze_speed_kinetics(seg)

        # -------------------------
        # TABLEAU (BRUT + IDEAL)
        # -------------------------
        st.markdown(f"### Intervalle {i+1} ({s_sec:.0f}s → {e_sec:.0f}s)")
        st.dataframe(pd.DataFrame({
            "Métrique": [
                "FC moyenne",
                "Dérive FC (bpm/min)",
                "Dérive FC (%/min)",
                "Dérive vitesse (km/h/min)",
                "Dérive vitesse (%/min)",
                "Durée BRUT (s)",
                "Distance (m)",
                "Vitesse BRUT (km/h)",
                "Allure BRUT",
                "D+ (m)", "D- (m)",
                "Temp moyenne (°C)",
                "Durée IDEAL (s)",
                "Vitesse IDEAL (km/h)",
                "Allure IDEAL",
            ],
            "Valeur": [
                stats["FC moyenne (bpm)"],
                d_bpm,
                d_pct,
                d_v_kmh,
                d_v_pct,
                round(t_s, 1),
                round(dist_m, 1),
                round(v_kmh, 2),
                pace_str,
                round(d_up, 1), round(d_down, 1),
                round(float(avgT), 2) if avgT is not None else None,
                round(t_ideal, 1) if t_ideal is not None else None,
                round(v_kmh_ideal, 2) if v_kmh_ideal is not None else None,
                pace_ideal_str
            ]
        }), hide_index=True, use_container_width=True)

        # -------------------------
        # GRAPHIQUE SEGMENT
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
    # GRAPHIQUE COMBINÉ (intervalles superposés)
    # ---------------------------------------------------------------
    if interval_segments:
        st.markdown("## 📊 Graphique combiné — tous les intervalles superposés")
        show_fc = st.checkbox("☑ FC", True, key="comb_fc_training_v2")
        show_pace = st.checkbox("☑ Allure", False, key="comb_pace_training_v2")
        show_power = st.checkbox("☑ Puissance", False, key="comb_pow_training_v2")

        figC, axC = plt.subplots(figsize=(10, 4.8))
        for idx2, seg, s0, s1 in interval_segments:
            plot_multi_signals(
                axC, seg, t0=s0,
                who=f"Int{idx2}",
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
