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
import re

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

# ========================= MÉTÉO (Open-Meteo archive) =========================
@st.cache_data(show_spinner=False)
def get_weather_openmeteo_day(lat, lon, date_obj):
    """
    Récupère TOUTE la journée météo (horaire) via Open-Meteo archive.
    Retourne (times, temps, winds, hums) en UTC.
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

def get_avg_weather_for_period(lat, lon, start_dt, end_dt):
    """
    Moyenne météo sur une période (robuste si période courte).
    - start_dt/end_dt attendus en naive UTC.
    """
    if start_dt is None or end_dt is None:
        return None, None, None
    try:
        # élargir si < 5 min
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
            closest = min(range(len(times)), key=lambda i: abs(times[i] - start_dt))
            return float(temps[closest]), float(winds[closest]), float(hums[closest])

        return float(np.mean(selT)), float(np.mean(selW)), float(np.mean(selH))
    except Exception:
        return None, None, None

# ========================= LECTURE FICHIERS ==============================
def load_activity(file):
    """Charge un fichier FIT, GPX, CSV ou TCX."""
    name = file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(file)

    elif name.endswith(".fit"):
        data = []
        try:
            # IMPORTANT: FitReader lit un file-like. On remet au début.
            file.seek(0)
            with fitdecode.FitReader(file) as fit:
                for frame in fit:
                    if isinstance(frame, fitdecode.records.FitDataMessage) and frame.name == "record":
                        data.append({f.name: f.value for f in frame.fields})
            df = pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Erreur FIT : {e}")

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
        if "time" in c.lower() or c.lower() == "timestamp":
            df = df.rename(columns={c: "timestamp"})
            break

    if "heart_rate" not in df.columns:
        # certains FIT ont "heart_rate" / d'autres "heart_rate" déjà ok
        # si absent, on tente les alias courants
        for cand in ["hr", "heart_rate_bpm", "heartrate", "bpm"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "heart_rate"})
                break

    if "heart_rate" not in df.columns:
        raise ValueError("Pas de FC détectée")

    # Nettoyage + types
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    # on travaille en naive UTC
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.dropna(subset=["timestamp", "heart_rate"]).reset_index(drop=True)

    for c in ["heart_rate","speed","enhanced_speed","power","distance","lat","lon","alt"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# ========================= OUTILS TEMPS (FIX "Format temps invalide") ==============================
_TIME_RE = re.compile(r"^\s*(\d+)(?::(\d{1,2}))?(?::(\d{1,2}))?\s*$")

def parse_time_to_seconds(tstr: str) -> int:
    """
    Accepte :
      - "hh:mm:ss"
      - "mm:ss"
      - "ss"
      - "12" / "12.5" / "12,5" (secondes)
    """
    if tstr is None:
        raise ValueError("Format temps invalide.")
    s = str(tstr).strip()
    if s == "":
        raise ValueError("Format temps invalide.")

    # format numérique simple (secondes)
    if ":" not in s:
        try:
            val = float(s.replace(",", "."))
            return int(round(val))
        except Exception:
            raise ValueError("Format temps invalide.")

    m = _TIME_RE.match(s)
    if not m:
        raise ValueError("Format temps invalide.")

    a = m.group(1)
    b = m.group(2)
    c = m.group(3)

    if b is None and c is None:
        # "ss"
        return int(a)

    if c is None:
        # "mm:ss" (a=mm, b=ss)
        mm = int(a)
        ss = int(b)
        return mm * 60 + ss

    # "hh:mm:ss" (a=hh, b=mm, c=ss)
    hh = int(a)
    mm = int(b)
    ss = int(c)
    return hh * 3600 + mm * 60 + ss

def seconds_to_hms(seconds: float) -> str:
    try:
        seconds = int(round(float(seconds)))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    except Exception:
        return "0:00:00"

def hms_from_seconds_for_inputs(seconds: float) -> str:
    # pour préremplir proprement les champs
    return seconds_to_hms(seconds)

# ========================= OUTILS CALCUL ==============================
def get_speed_col(df):
    if "enhanced_speed" in df.columns: return "enhanced_speed"
    if "speed" in df.columns: return "speed"
    return None

def haversine_dist_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def segment_distance_m(df_seg):
    if df_seg is None or df_seg.empty or len(df_seg) < 2:
        return 0.0

    # distance cumulée dispo ?
    if "distance" in df_seg.columns:
        d0 = float(df_seg["distance"].iloc[0])
        d1 = float(df_seg["distance"].iloc[-1])
        if np.isfinite(d0) and np.isfinite(d1) and d1 >= d0:
            return float(d1 - d0)

    # speed * dt
    sp = get_speed_col(df_seg)
    if sp is not None and "delta_t" in df_seg.columns:
        dist = float(np.nansum(df_seg[sp].fillna(0).values * df_seg["delta_t"].fillna(0).values))
        if dist > 0:
            return dist

    # gps
    if "lat" in df_seg.columns and "lon" in df_seg.columns:
        lats = df_seg["lat"].astype(float).values
        lons = df_seg["lon"].astype(float).values
        dist = 0.0
        for i in range(1, len(df_seg)):
            if all(np.isfinite([lats[i-1], lats[i], lons[i-1], lons[i]])):
                dist += haversine_dist_m(lats[i-1], lons[i-1], lats[i], lons[i])
        return float(max(0.0, dist))

    return 0.0

def segment_elevation_up_down(df_seg):
    """
    D+ / D- (m) sur le segment, robuste (lissage médian).
    """
    if df_seg is None or df_seg.empty or "alt" not in df_seg.columns:
        return 0.0, 0.0

    alt = pd.to_numeric(df_seg["alt"], errors="coerce").astype(float)
    alt = alt.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
    alt = alt.rolling(5, min_periods=1, center=True).median()

    d = alt.diff().fillna(0.0)
    dup = float(d[d > 0].sum())
    ddn = float(-d[d < 0].sum())
    if not math.isfinite(dup): dup = 0.0
    if not math.isfinite(ddn): ddn = 0.0
    return max(0.0, dup), max(0.0, ddn)

def segment_grade_percent_net(df_seg):
    """
    Pente moyenne nette (%) = 100 * (alt_fin - alt_deb) / distance
    """
    dist_m = segment_distance_m(df_seg)
    if dist_m <= 0 or "alt" not in df_seg.columns:
        return None
    alt = pd.to_numeric(df_seg["alt"], errors="coerce").astype(float)
    alt = alt.replace([np.inf, -np.inf], np.nan)
    alt = alt.dropna()
    if len(alt) < 2:
        return None
    deniv = float(alt.iloc[-1] - alt.iloc[0])
    grade = 100.0 * deniv / dist_m
    if not math.isfinite(grade):
        return None
    return float(grade)

def format_pace_min_per_km(v_kmh):
    if v_kmh is None or v_kmh <= 0 or not math.isfinite(v_kmh):
        return None
    min_per_km = 60.0 / float(v_kmh)
    total_seconds = int(round(min_per_km * 60.0))
    return total_seconds // 60, total_seconds % 60, min_per_km

# ========================= Lissage Cardio ==============================
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
    total_dur = float(df["time_s"].iloc[-1]) if len(df) else 0.0

    if total_dur < 360:
        window_sec = 5
    elif total_dur < 900:
        window_sec = 10
    else:
        window_sec = 20

    step = np.median(np.diff(df["time_s"])) if len(df) > 2 else 1
    if step <= 0 or np.isnan(step):
        step = 1
    window_size = max(1, int(window_sec / step))

    df["hr_smooth"] = pd.to_numeric(df[hr_col], errors="coerce").rolling(window_size, min_periods=1).mean()
    sp = get_speed_col(df)
    if sp:
        df["speed_smooth"] = pd.to_numeric(df[sp], errors="coerce").rolling(window_size, min_periods=1).mean()
    if "power" in df.columns:
        df["power_smooth"] = pd.to_numeric(df["power"], errors="coerce").rolling(window_size, min_periods=1).mean()

    pauses = int((df["delta_t"] > 2 * median_step).sum())
    return df, window_sec, total_dur, pauses

# ========================= Analyse FC ==============================
def analyze_heart_rate(df):
    hr = df["hr_smooth"].dropna()
    mean_hr = float(hr.mean()) if len(hr) else np.nan
    max_hr = float(hr.max()) if len(hr) else np.nan
    min_hr = float(hr.min()) if len(hr) else np.nan

    if len(df) >= 2 and df["time_s"].nunique() >= 2:
        slope, _, _, _, _ = linregress(df["time_s"], df["hr_smooth"])
    else:
        slope = 0.0
    drift_per_min = float(slope * 60)
    drift_percent = (drift_per_min / mean_hr) * 100 if mean_hr and mean_hr > 0 else np.nan

    stats = {
        "FC moyenne (bpm)": round(mean_hr, 1) if math.isfinite(mean_hr) else None,
        "FC max (bpm)": round(max_hr, 1) if math.isfinite(max_hr) else None,
        "FC min (bpm)": round(min_hr, 1) if math.isfinite(min_hr) else None,
        "Dérive (bpm/min)": round(drift_per_min, 4),
        "Dérive (%/min)": round(drift_percent, 4) if not np.isnan(drift_percent) else None,
        "Durée segment (s)": round(float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]), 1) if len(df) else 0.0,
    }
    return stats, float(drift_per_min), (None if np.isnan(drift_percent) else float(drift_percent))

# ========================= Cinétique vitesse =========================
def analyze_speed_kinetics(df):
    sp_col = get_speed_col(df)
    if sp_col is None or df[sp_col].dropna().empty or df["time_s"].nunique() < 2:
        return None, None

    slope, _, _, _, _ = linregress(df["time_s"], df[sp_col])
    drift_per_min = float(slope * 60)
    mean_speed = float(df[sp_col].mean())
    drift_percent = (drift_per_min / mean_speed * 100) if mean_speed > 0 else None

    return round(drift_per_min, 4), (round(drift_percent, 4) if drift_percent is not None else None)

# ========================= PDF =========================
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

# ========================= METEO sur un segment (Open-Meteo archive) =========================
def segment_lat_lon(df_seg):
    if df_seg is None or df_seg.empty:
        return None, None
    if "lat" not in df_seg.columns or "lon" not in df_seg.columns:
        return None, None
    sub = df_seg[["lat","lon"]].dropna()
    if sub.empty:
        return None, None
    lat = float(sub["lat"].iloc[0])
    lon = float(sub["lon"].iloc[0])
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None, None
    return lat, lon

def segment_start_end_dt(df_seg):
    if df_seg is None or df_seg.empty or "timestamp" not in df_seg.columns:
        return None, None
    t0 = df_seg["timestamp"].dropna()
    if t0.empty:
        return None, None
    start_dt = t0.iloc[0]
    end_dt = t0.iloc[-1]
    if not isinstance(start_dt, datetime) or not isinstance(end_dt, datetime):
        return None, None
    return start_dt, end_dt

def get_segment_weather(df_seg):
    """
    Retourne avg_temp, avg_wind, avg_hum pour le segment via Open-Meteo archive.
    Si pas de GPS/temps => (None,None,None)
    """
    lat, lon = segment_lat_lon(df_seg)
    start_dt, end_dt = segment_start_end_dt(df_seg)
    if lat is None or lon is None or start_dt is None or end_dt is None:
        return None, None, None
    return get_avg_weather_for_period(lat, lon, start_dt, end_dt)

# ========================= Recalibrage "code2" (non-linéaire) =========================
def temp_multiplier_nonlin(temp, opt_temp=12.0, k_hot=0.002, k_cold=0.002):
    """
    Même logique que ton code 2 :
    - si temp > opt => 1 + k_hot*(temp-opt)
    - si temp < opt => 1 + k_cold*(opt-temp)
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

def elevation_factor_from_dup_ddn(D_up, D_down, segment_length_m=1000.0, k_up=1.040, k_down=0.996):
    """
    Même esprit que ton code 2 :
      factor = 1 + (k_up-1)*(D_up/seg_len) + (1-k_down)*(D_down/seg_len)
    """
    try:
        seg_len = float(segment_length_m) if segment_length_m and segment_length_m > 0 else 1000.0
        up_factor = (float(k_up) - 1.0) * (float(D_up) / seg_len)
        down_factor = (1.0 - float(k_down)) * (float(D_down) / seg_len)
        factor = 1.0 + up_factor + down_factor
        return max(0.01, float(factor))
    except Exception:
        return 1.0

def recalibrate_time_to_ideal(time_seconds_raw, D_up, D_down, distance_m, temp_real,
                             k_up=1.040, k_down=0.996,
                             k_temp_hot=0.002, k_temp_cold=0.002,
                             opt_temp=12.0):
    """
    Convertit un temps réel -> temps recalibré (conditions idéales)
      - retire effet D+/D-
      - retire effet température réelle
    """
    try:
        t = float(time_seconds_raw)
        if not math.isfinite(t) or t <= 0:
            return None

        dist = float(distance_m) if distance_m and distance_m > 0 else 1000.0

        # 1) retirer effet dénivelé
        fact_elev = elevation_factor_from_dup_ddn(D_up, D_down, segment_length_m=dist, k_up=k_up, k_down=k_down)
        t_no_elev = t / fact_elev

        # 2) retirer effet température réelle
        mult_real = temp_multiplier_nonlin(temp_real, opt_temp=opt_temp, k_hot=k_temp_hot, k_cold=k_temp_cold)
        t_no_temp = t_no_elev / (mult_real if mult_real != 0 else 1.0)

        return max(0.0, float(t_no_temp))
    except Exception:
        return None

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

def compute_pace_series_from_speed(df):
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
        pace_series = compute_pace_series_from_speed(df)
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

# ========================= SIDEBAR (page côté) ==============================
st.sidebar.title("⚙️ Coefficients (comme ton code 2)")
st.sidebar.caption("Ces coefficients s’appliquent au recalibrage (conditions idéales).")

k_up = st.sidebar.number_input("Coefficient montée (k_up)", value=1.040, format="%.3f", step=0.001)
k_down = st.sidebar.number_input("Coefficient descente (k_down)", value=0.996, format="%.3f", step=0.001)
k_temp_hot = st.sidebar.number_input("Sensibilité chaude (k_temp_hot)", value=0.0020, format="%.4f", step=0.0005)
k_temp_cold = st.sidebar.number_input("Sensibilité froide (k_temp_cold)", value=0.0020, format="%.4f", step=0.0005)
opt_temp = st.sidebar.number_input("Température optimale (°C)", value=12.0, format="%.1f", step=0.5)

st.sidebar.markdown("---")
st.sidebar.caption("Si pas de GPS/temps, la météo auto ne peut pas être calculée. Dans ce cas tu peux forcer une température manuelle par test/intervalle.")

# ---------------------------------------------------------------------
# ONGLET 1 : TESTS D’ENDURANCE (2 à 6 tests + VC + LOG + D′)
# ---------------------------------------------------------------------
with tabs[0]:
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
    A = None
    k_log = None

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

            manual_temp = st.number_input(f"🌡️ Température manuelle Test {i} (°C) (si météo auto indispo)", value=float(0.0), step=0.5, key=f"manual_temp_{i}")

            if uploaded:
                try:
                    df = load_activity(uploaded)
                except Exception as e:
                    st.error(f"Erreur dans le fichier du Test {i} : {e}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                lag = st.slider(f"Correction du décalage capteur (s) — Test {i}", 0, 10, 0, key=f"lag_{i}")
                df["timestamp"] = df["timestamp"] - pd.to_timedelta(lag, unit="s")

                df, window, total_dur, pauses = smooth_hr(df)
                st.caption(f"Durée détectée : {total_dur:.1f}s • Lissage : {window}s • Pauses détectées : {pauses}")

                col_start, col_end = st.columns(2)
                with col_start:
                    start_str = st.text_input(f"Début (hh:mm:ss) — Test {i}", value="0:00:00", key=f"start_{i}")
                with col_end:
                    end_str = st.text_input(f"Fin (hh:mm:ss) — Test {i}", value="0:12:00", key=f"end_{i}")

                try:
                    start_sec = parse_time_to_seconds(start_str)
                    end_sec = parse_time_to_seconds(end_str)
                except Exception:
                    st.error("Format temps invalide. Exemple: 0:12:00 ou 12:30 ou 90")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                if end_sec <= start_sec:
                    st.error("Fin doit être > début")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                if end_sec > df["time_s"].max():
                    st.warning(f"⚠️ Fin > fichier ({df['time_s'].max():.0f}s). Limitation auto.")
                    end_sec = int(df["time_s"].max())

                segment = df[(df["time_s"] >= start_sec) & (df["time_s"] <= end_sec)]
                if len(segment) < 10:
                    st.warning("Segment trop court pour analyse.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                # ---- ANALYSE FC ----
                stats, drift_bpm, drift_pct = analyze_heart_rate(segment)

                # ---- DIST / TEMPS ----
                dist_m = segment_distance_m(segment)
                t_s_raw = float(end_sec - start_sec)

                v_kmh_raw = (3.6 * dist_m / t_s_raw) if (t_s_raw > 0 and dist_m > 0) else 0.0

                # ---- DENIVELE ----
                D_up, D_down = segment_elevation_up_down(segment)
                grade_pct = segment_grade_percent_net(segment)

                # ---- METEO AUTO ----
                avgT, avgW, avgH = get_segment_weather(segment)
                # fallback manuel si auto indispo
                temp_real = avgT if avgT is not None else (manual_temp if manual_temp != 0.0 else None)

                # ---- RE-CALIBRAGE (conditions idéales) sur le TEMPS ----
                t_s_ideal = recalibrate_time_to_ideal(
                    time_seconds_raw=t_s_raw,
                    D_up=D_up,
                    D_down=D_down,
                    distance_m=dist_m if dist_m > 0 else 1000.0,
                    temp_real=temp_real,
                    k_up=k_up, k_down=k_down,
                    k_temp_hot=k_temp_hot, k_temp_cold=k_temp_cold,
                    opt_temp=opt_temp
                )

                v_kmh_ideal = (3.6 * dist_m / t_s_ideal) if (t_s_ideal is not None and t_s_ideal > 0 and dist_m > 0) else None

                # ---- ALLURES ----
                pace_raw = format_pace_min_per_km(v_kmh_raw)
                pace_raw_str = f"{int(pace_raw[0])}:{int(pace_raw[1]):02d} min/km" if pace_raw else "–"

                pace_ideal = format_pace_min_per_km(v_kmh_ideal) if v_kmh_ideal else None
                pace_ideal_str = f"{int(pace_ideal[0])}:{int(pace_ideal[1]):02d} min/km" if pace_ideal else "–"

                # ---- CINÉTIQUE VITESSE ----
                d_v_kmh, d_v_pct = analyze_speed_kinetics(segment)

                # ---- TABLEAU ----
                df_table = pd.DataFrame({
                    "Métrique": [
                        "FC moyenne (bpm)", "FC max (bpm)",
                        "Dérive FC (bpm/min)", "Dérive FC (%/min)",
                        "Dérive vitesse (km/h/min)", "Dérive vitesse (%/min)",
                        "Durée segment réelle (s)",
                        "Durée segment recalibrée (s)",
                        "Distance (m)",
                        "Vitesse réelle (km/h)",
                        "Allure réelle (min/km)",
                        "Vitesse recalibrée (km/h)",
                        "Allure recalibrée (conditions idéales) (min/km)",
                        "Pente nette moyenne (%)",
                        "D+ (m)", "D- (m)",
                        "Température (°C)",
                        "Vent (m/s)",
                        "Humidité (%)",
                    ],
                    "Valeur": [
                        stats["FC moyenne (bpm)"], stats["FC max (bpm)"],
                        drift_bpm, drift_pct,
                        d_v_kmh, d_v_pct,
                        round(t_s_raw, 1),
                        (round(t_s_ideal, 1) if t_s_ideal is not None else None),
                        round(dist_m, 1),
                        round(v_kmh_raw, 2),
                        pace_raw_str,
                        (round(v_kmh_ideal, 2) if v_kmh_ideal is not None else None),
                        pace_ideal_str,
                        (round(grade_pct, 3) if grade_pct is not None else None),
                        round(D_up, 1), round(D_down, 1),
                        (round(temp_real, 2) if temp_real is not None else None),
                        (round(avgW, 2) if avgW is not None else None),
                        (round(avgH, 2) if avgH is not None else None),
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
                    "segment": segment,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "stats": stats,
                    "drift_bpm": drift_bpm,
                    "drift_pct": drift_pct,
                    "d_v_kmh": d_v_kmh,
                    "d_v_pct": d_v_pct,
                    "dist_m": dist_m,
                    "t_s_raw": t_s_raw,
                    "t_s_ideal": t_s_ideal,
                    "v_kmh_raw": v_kmh_raw,
                    "v_kmh_ideal": v_kmh_ideal,
                    "pace_raw_str": pace_raw_str,
                    "pace_ideal_str": pace_ideal_str,
                    "grade_pct": grade_pct,
                    "D_up": D_up,
                    "D_down": D_down,
                    "temp_real": temp_real,
                    "wind": avgW,
                    "hum": avgH,
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

    valid_tests_raw = [t for t in tests_data if t["dist_m"] > 0 and t["t_s_raw"] and t["t_s_raw"] > 0]
    valid_tests_ideal = [t for t in tests_data if t["dist_m"] > 0 and t["t_s_ideal"] and t["t_s_ideal"] > 0]

    # VC brute
    if len(valid_tests_raw) >= 2:
        D = np.array([t["dist_m"] for t in valid_tests_raw], dtype=float)
        T = np.array([t["t_s_raw"] for t in valid_tests_raw], dtype=float)
        slope, intercept = np.polyfit(T, D, 1)
        VC_m_s = float(slope)
        D_prime_raw = float(intercept)
        VC_kmh_raw = VC_m_s * 3.6
    else:
        VC_kmh_raw = None

    # VC recalibrée (conditions idéales)
    if len(valid_tests_ideal) >= 2:
        D2 = np.array([t["dist_m"] for t in valid_tests_ideal], dtype=float)
        T2 = np.array([t["t_s_ideal"] for t in valid_tests_ideal], dtype=float)
        slope2, intercept2 = np.polyfit(T2, D2, 1)
        VC_m_s2 = float(slope2)
        D_prime_ideal = float(intercept2)
        VC_kmh_ideal = VC_m_s2 * 3.6
    else:
        VC_kmh_ideal = None

    if VC_kmh_raw is None and VC_kmh_ideal is None:
        st.info("Il faut au moins deux tests valides pour calculer la VC (brute et/ou recalibrée).")
    else:
        colvc1, colvc2 = st.columns(2)
        with colvc1:
            st.markdown("### 📌 VC réelle (brute)")
            if VC_kmh_raw is not None and VC_kmh_raw > 0:
                pace = format_pace_min_per_km(VC_kmh_raw)
                vc_pace_str = f"{pace[0]}:{pace[1]:02d} min/km" if pace else "–"
                st.success(f"**VC réelle = {VC_kmh_raw:.2f} km/h**  \n➡️ **{vc_pace_str}**  \n**D′ = {D_prime_raw:.1f} m**")
            else:
                st.info("VC brute non calculable.")
        with colvc2:
            st.markdown("### 🔧 VC recalibrée (conditions idéales)")
            if VC_kmh_ideal is not None and VC_kmh_ideal > 0:
                pace = format_pace_min_per_km(VC_kmh_ideal)
                vc_pace_str = f"{pace[0]}:{pace[1]:02d} min/km" if pace else "–"
                st.success(f"**VC recalibrée = {VC_kmh_ideal:.2f} km/h**  \n➡️ **{vc_pace_str}**  \n**D′ = {D_prime_ideal:.1f} m**")
            else:
                st.info("VC recalibrée non calculable.")

    st.markdown('</div>', unsafe_allow_html=True)

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

    df, window, dur, pauses, filename = st.session_state.training_session

    st.markdown(f"### 📂 Séance importée : **{filename}**")
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
                value=hms_from_seconds_for_inputs(start_s),
                key=f"tr_int_start_{i}"
            )
        with c2:
            e_str = st.text_input(
                f"Fin intervalle {i+1} (hh:mm:ss)",
                value=hms_from_seconds_for_inputs(end_s),
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
        except Exception:
            st.warning(f"⛔ Format invalide intervalle {i+1} (ex: 0:05:00)")

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
        if seg.empty or len(seg) < 10:
            continue

        interval_segments.append((i+1, seg, s_sec, e_sec))

        # --- FC ---
        stats, d_bpm, d_pct = analyze_heart_rate(seg)

        # --- Distance / temps ---
        dist_m = segment_distance_m(seg)
        t_s_raw = float(e_sec - s_sec)
        v_kmh_raw = (3.6 * dist_m / t_s_raw) if (t_s_raw > 0 and dist_m > 0) else 0.0

        # --- D+ / D- / pente ---
        D_up, D_down = segment_elevation_up_down(seg)
        grade_pct = segment_grade_percent_net(seg)

        # --- METEO ---
        avgT, avgW, avgH = get_segment_weather(seg)
        manual_temp = st.number_input(
            f"🌡️ Température manuelle Intervalle {i+1} (°C) (si météo auto indispo)",
            value=float(0.0),
            step=0.5,
            key=f"manual_temp_int_{i}"
        )
        temp_real = avgT if avgT is not None else (manual_temp if manual_temp != 0.0 else None)

        # --- recalibrage temps -> idéal ---
        t_s_ideal = recalibrate_time_to_ideal(
            time_seconds_raw=t_s_raw,
            D_up=D_up,
            D_down=D_down,
            distance_m=dist_m if dist_m > 0 else 1000.0,
            temp_real=temp_real,
            k_up=k_up, k_down=k_down,
            k_temp_hot=k_temp_hot, k_temp_cold=k_temp_cold,
            opt_temp=opt_temp
        )
        v_kmh_ideal = (3.6 * dist_m / t_s_ideal) if (t_s_ideal is not None and t_s_ideal > 0 and dist_m > 0) else None

        # --- allures ---
        pace_raw = format_pace_min_per_km(v_kmh_raw)
        pace_raw_str = f"{pace_raw[0]}:{pace_raw[1]:02d} min/km" if pace_raw else "–"
        pace_ideal = format_pace_min_per_km(v_kmh_ideal) if v_kmh_ideal else None
        pace_ideal_str = f"{pace_ideal[0]}:{pace_ideal[1]:02d} min/km" if pace_ideal else "–"

        # --- dérive vitesse ---
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
                "Durée réelle (s)",
                "Durée recalibrée (s)",
                "Distance (m)",
                "Vitesse réelle (km/h)",
                "Allure réelle",
                "Vitesse recalibrée (km/h)",
                "Allure recalibrée (conditions idéales)",
                "Pente nette moyenne (%)",
                "D+ (m)",
                "D- (m)",
                "Température (°C)",
                "Vent (m/s)",
                "Humidité (%)",
            ],
            "Valeur": [
                stats["FC moyenne (bpm)"],
                d_bpm,
                d_pct,
                d_v_kmh,
                d_v_pct,
                round(t_s_raw, 1),
                (round(t_s_ideal, 1) if t_s_ideal is not None else None),
                round(dist_m, 1),
                round(v_kmh_raw, 2),
                pace_raw_str,
                (round(v_kmh_ideal, 2) if v_kmh_ideal is not None else None),
                pace_ideal_str,
                (round(grade_pct, 3) if grade_pct is not None else None),
                round(D_up, 1),
                round(D_down, 1),
                (round(temp_real, 2) if temp_real is not None else None),
                (round(avgW, 2) if avgW is not None else None),
                (round(avgH, 2) if avgH is not None else None),
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
        for idx2, seg2, s0, s1 in interval_segments:
            plot_multi_signals(
                axC, seg2, t0=s0,
                who=f"Int{idx2}",
                show_fc=show_fc,
                show_pace=show_pace and ("speed_smooth" in seg2.columns),
                show_power=show_power and ("power_smooth" in seg2.columns)
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
