# analyse_course_v3_merged.py — Suite Coach Running — v3 + Code 2 intégré
# Onglets :
#   [0] 🏃 Prédiction de course (v3)
#   [1] 🧪 Tests d'endurance + VC
#   [2] ⚙️  Analyse entraînement
#
# pip install streamlit gpxpy fitparse fitdecode pandas numpy pydeck matplotlib requests scipy

import streamlit as st
import math
import gpxpy
from fitparse import FitFile
try:
    import fitdecode
    HAS_FITDECODE = True
except ImportError:
    HAS_FITDECODE = False

from datetime import datetime, timedelta, date, time
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
import xml.etree.ElementTree as ET
import requests
import io
from io import BytesIO
import re
from scipy import stats as sp_stats
from scipy.stats import linregress

# ══════════════════════════════════════════════════════════════
# CONFIG — un seul set_page_config
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Coach Running — Suite complète",
                   layout="wide", page_icon="🏃")
TZ_NAME_DEFAULT = "Europe/Paris"

# ── Couleurs (code 2) ──
COLOR_RED_T1   = "#d21f3c";  COLOR_RED_T2   = "#8b0a1a";  COLOR_RED_SES   = "#f57c92"
COLOR_BLUE_T1  = "#0066cc";  COLOR_BLUE_T2  = "#003366";  COLOR_BLUE_SES  = "#66a3ff"
COLOR_ORANGE_T1= "#ff8c00";  COLOR_ORANGE_T2= "#cc6600";  COLOR_ORANGE_SES= "#ffb84d"
COLOR_GREY     = "#6b7280"

ACCEPTED_TYPES = ["fit","FIT","gpx","GPX","csv","CSV","tcx","TCX"]

# ══════════════════════════════════════════════════════════════
# CSS UNIFIÉ
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
/* ── Boîtes pédagogiques v3 ── */
.param-box {{
    background:#f8f9fa; border-left:4px solid #1f77b4;
    border-radius:4px; padding:8px 12px; margin-bottom:8px; font-size:0.88rem;
}}
.param-up   {{ color:#d62728; font-weight:600; }}
.param-down {{ color:#2ca02c; font-weight:600; }}
.highlight-box {{
    background:#fff3cd; border:1px solid #ffc107;
    border-radius:6px; padding:12px 16px; margin:8px 0;
}}
/* ── Cards (code 2) ── */
.report-card {{
  padding:1rem 1.2rem; border-radius:14px;
  border:1px solid rgba(0,0,0,0.08);
  background:linear-gradient(180deg,#ffffff 0%,#fafafa 100%);
  box-shadow:0 6px 18px rgba(0,0,0,0.06); margin-bottom:0.8rem;
}}
/* ── DataFrames ── */
div[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; }}
div[data-testid="stDataFrame"] td {{font-size:0.92rem; font-weight:500; padding:6px 10px;}}
div[data-testid="stDataFrame"] th {{
  background-color:#f3f4f6; color:#111827;
  font-weight:600; border-bottom:2px solid #e5e7eb;
}}
/* ── Rows stylés ── */
.tr-raw td     {{ background-color:{COLOR_RED_T2};  color:white; border-bottom:1px solid rgba(255,255,255,0.25); }}
.tr-ideal td   {{ background-color:{COLOR_BLUE_T2}; color:white; border-bottom:1px solid rgba(255,255,255,0.25); }}
.tr-general td {{ background-color:{COLOR_GREY};    color:white; border-bottom:1px solid rgba(255,255,255,0.25); }}
/* ── Sidebar ── */
.sidebar-label {{
    background:#e8f4fd; border-radius:4px; padding:6px 10px;
    font-size:0.80rem; color:#1f77b4; margin-bottom:10px;
}}
.block-container {{ padding-top:1.4rem; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPERS UI
# ══════════════════════════════════════════════════════════════
def param_help(text_up, text_down, note=""):
    note_html = f"<br><em>{note}</em>" if note else ""
    st.markdown(f'<div class="param-box">'
                f'<span class="param-up">⬆️ Augmenter</span> : {text_up}<br>'
                f'<span class="param-down">⬇️ Diminuer</span> : {text_down}'
                f'{note_html}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# UTILITAIRES PARTAGÉS
# ══════════════════════════════════════════════════════════════
def safe_float(val, default=0.0):
    try:
        if val is None: return float(default)
        if isinstance(val, str):
            s = val.strip()
            if s in ("","nan","none"): return float(default)
            return float(s.replace(",","."))
        if isinstance(val,(float,int,np.number)):
            if np.isnan(val) or np.isinf(val): return float(default)
            return float(val)
        return float(val)
    except Exception: return float(default)


def hms_to_seconds(hms):
    if hms is None: return 0
    try:
        parts=[int(p) for p in str(hms).strip().split(":")]
        if len(parts)==3:   h,m,s=parts
        elif len(parts)==2: h,m,s=0,parts[0],parts[1]
        elif len(parts)==1: h,m,s=0,0,parts[0]
        else: return 0
        if not(0<=m<=59 and 0<=s<=59): return 0
        return max(0,h*3600+m*60+s)
    except Exception: return 0


def seconds_to_hms(s):
    try:
        s=int(round(float(s)))
        return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"
    except Exception: return "0:00:00"


def hms_to_timedelta(hms): return timedelta(seconds=hms_to_seconds(hms))


def pace_str(secs_per_km):
    if secs_per_km is None or secs_per_km<=0 or not math.isfinite(secs_per_km):
        return "0:00"
    t=int(round(float(secs_per_km)))
    return f"{t//60}:{t%60:02d}"


def haversine_m(lat1,lon1,lat2,lon2):
    R=6371000.0
    p1,p2=math.radians(lat1),math.radians(lat2)
    dp=math.radians(lat2-lat1); dl=math.radians(lon2-lon1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))


def bearing_deg(lat1,lon1,lat2,lon2):
    p1,p2=math.radians(lat1),math.radians(lat2); dl=math.radians(lon2-lon1)
    y=math.sin(dl)*math.cos(p2)
    x=math.cos(p1)*math.sin(p2)-math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(y,x))+360.0)%360.0


def compute_dplus_dminus(elevs):
    arr=np.array([safe_float(e,np.nan) for e in elevs],dtype=float)
    arr=arr[~np.isnan(arr)]
    if arr.size<2: return 0.0,0.0
    diffs=np.diff(arr)
    return float(np.sum(np.clip(diffs,0,None))),float(-np.sum(np.clip(diffs,None,0)))


# ── Utilitaires code 2 ──
_TIME_RE=re.compile(r"^\s*(\d+)(?::(\d{1,2}))?(?::(\d{1,2}))?\s*$")

def parse_time_to_seconds(tstr):
    if tstr is None: raise ValueError("Format temps invalide.")
    s=str(tstr).strip()
    if s=="": raise ValueError("Format temps invalide.")
    if ":" not in s:
        try: return int(round(float(s.replace(",","."))))
        except Exception: raise ValueError("Format temps invalide.")
    m=_TIME_RE.match(s)
    if not m: raise ValueError("Format temps invalide.")
    a,b,c=m.group(1),m.group(2),m.group(3)
    if b is None and c is None: return int(a)
    if c is None: return int(a)*60+int(b)
    return int(a)*3600+int(b)*60+int(c)


def hms_from_seconds_for_inputs(seconds): return seconds_to_hms(seconds)


def get_speed_col(df):
    if "enhanced_speed" in df.columns: return "enhanced_speed"
    if "speed"          in df.columns: return "speed"
    return None


def haversine_2d_m(lat1,lon1,lat2,lon2):
    R=6371008.8; phi1=np.radians(lat1); phi2=np.radians(lat2)
    dphi=np.radians(lat2-lat1); dlambda=np.radians(lon2-lon1)
    a=np.sin(dphi/2)**2+np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def distance_3d_m(lat1,lon1,alt1,lat2,lon2,alt2):
    try:
        d2=haversine_2d_m(lat1,lon1,lat2,lon2)
        if alt1 is None or alt2 is None or not math.isfinite(float(alt1)) or not math.isfinite(float(alt2)):
            return float(d2)
        dz=float(alt2)-float(alt1)
        return float(math.sqrt(d2*d2+dz*dz))
    except Exception: return 0.0


def segment_distance_m(df_seg):
    if df_seg is None or df_seg.empty or len(df_seg)<2: return 0.0
    if "distance" in df_seg.columns:
        d0,d1=df_seg["distance"].iloc[0],df_seg["distance"].iloc[-1]
        if pd.notna(d0) and pd.notna(d1):
            try:
                d0,d1=float(d0),float(d1)
                if math.isfinite(d0) and math.isfinite(d1) and d1>=d0 and (d1-d0)<1e7:
                    return float(d1-d0)
            except Exception: pass
    if "lat" in df_seg.columns and "lon" in df_seg.columns:
        lat=pd.to_numeric(df_seg["lat"],errors="coerce").values
        lon=pd.to_numeric(df_seg["lon"],errors="coerce").values
        alt=pd.to_numeric(df_seg["alt"],errors="coerce").values if "alt" in df_seg.columns else None
        dist=0.0
        for i in range(1,len(df_seg)):
            if np.isfinite(lat[i-1]) and np.isfinite(lat[i]) and np.isfinite(lon[i-1]) and np.isfinite(lon[i]):
                if alt is not None and np.isfinite(alt[i-1]) and np.isfinite(alt[i]):
                    dist+=distance_3d_m(lat[i-1],lon[i-1],alt[i-1],lat[i],lon[i],alt[i])
                else:
                    dist+=haversine_2d_m(lat[i-1],lon[i-1],lat[i],lon[i])
        if dist>0: return float(dist)
    sp=get_speed_col(df_seg)
    if sp is not None and "delta_t" in df_seg.columns:
        try: return float(np.nansum(df_seg[sp].fillna(0).values*df_seg["delta_t"].fillna(0).values))
        except Exception: return 0.0
    return 0.0


def segment_elevation_up_down(df_seg):
    if df_seg is None or df_seg.empty or "alt" not in df_seg.columns: return 0.0,0.0
    alt=pd.to_numeric(df_seg["alt"],errors="coerce").astype(float)
    alt=alt.replace([np.inf,-np.inf],np.nan).interpolate(limit_direction="both")
    alt=alt.rolling(7,min_periods=1,center=True).median()
    d=alt.diff().fillna(0.0)
    return max(0.0,float(d[d>0].sum())), max(0.0,float(-d[d<0].sum()))


def segment_grade_percent_net(df_seg):
    if df_seg is None or df_seg.empty or "alt" not in df_seg.columns: return None
    dist_m=segment_distance_m(df_seg)
    if dist_m<=0: return None
    alt=pd.to_numeric(df_seg["alt"],errors="coerce").dropna()
    if len(alt)<2: return None
    grade=100.0*float(alt.iloc[-1]-alt.iloc[0])/dist_m
    return grade if math.isfinite(grade) else None


def format_pace_min_per_km(v_kmh):
    if v_kmh is None or v_kmh<=0 or not math.isfinite(v_kmh): return None
    min_per_km=60.0/float(v_kmh)
    total_seconds=int(round(min_per_km*60.0))
    return total_seconds//60, total_seconds%60, min_per_km


def _pace_str_from_kmh(v_kmh):
    try:
        p=format_pace_min_per_km(v_kmh)
        if not p: return "–"
        return f"{int(p[0])}:{int(p[1]):02d} min/km"
    except Exception: return "–"



# ══════════════════════════════════════════════════════════════
# MODÈLES PHYSIQUES (v3)
# ══════════════════════════════════════════════════════════════

def wbgt_simplified(T_c, RH):
    """WBGT approximé (Stull 2011)."""
    try:
        RH_c = max(0.0, min(100.0, float(RH)))
        T = float(T_c)
        Tw = (T * math.atan(0.151977 * (RH_c + 8.313659) ** 0.5)
              + math.atan(T + RH_c)
              - math.atan(RH_c - 1.676331)
              + 0.00391838 * RH_c ** 1.5 * math.atan(0.023101 * RH_c)
              - 4.686035)
        Tg = T + 2.0
        return 0.7 * Tw + 0.2 * Tg + 0.1 * T
    except Exception:
        return float(T_c)

def effective_temp(T_c, RH, use_wbgt):
    return wbgt_simplified(T_c, RH) if use_wbgt else float(T_c)

def altitude_vo2_multiplier(altitude_m, altitude_ref_m=0.0):
    alt = max(0.0, float(altitude_m))
    alt_ref = max(0.0, float(altitude_ref_m))
    effective_alt = max(0.0, alt - max(1500.0, alt_ref))
    penalty = min(0.25, 0.01 * (effective_alt / 100.0))
    return 1.0 + penalty

def minetti_cost(grade_fraction):
    g = max(-0.45, min(0.45, float(grade_fraction)))
    c = (155.4*g**5 - 30.4*g**4 - 43.3*g**3 + 46.3*g**2 + 19.5*g + 3.6)
    return max(0.1, float(c))

def minetti_multiplier(grade_pct):
    flat = minetti_cost(0.0)
    ratio = minetti_cost(float(grade_pct) / 100.0) / flat
    return float(max(0.92, min(1.35, ratio)))

def grade_multiplier_heuristic(grade_pct, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down):
    try:
        g = float(grade_pct) / 100.0
        g0u = max(1e-6, float(g0_up) / 100.0)
        g0d = max(1e-6, float(g0_down) / 100.0)
        if g >= 0:
            g_eff = math.tanh(g / g0u) * g0u
            mult = 1.0 + float(k_up) * g_eff
        else:
            g_eff = math.tanh((-g) / g0d) * g0d
            bonus = min(float(k_down) * g_eff, abs(float(down_cap)))
            mult = 1.0 - bonus
        mult = min(mult, 1.0 + float(max_up))
        mult = max(mult, 1.0 + float(max_down))
        return max(0.01, float(mult))
    except Exception:
        return 1.0

def combined_grade_multiplier(grade_pct, use_minetti, minetti_weight,
                               k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down):
    if not use_minetti:
        return grade_multiplier_heuristic(grade_pct, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down)
    m_min = minetti_multiplier(grade_pct)
    m_heu = grade_multiplier_heuristic(grade_pct, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down)
    w = max(0.0, min(1.0, float(minetti_weight)))
    return w * m_min + (1.0 - w) * m_heu

def temp_multiplier_v3(temp_eff, opt_temp, cold_quad, hot_quad, max_penalty):
    if temp_eff is None:
        return 1.0
    d = float(temp_eff) - float(opt_temp)
    pen = hot_quad * d**2 if d >= 0 else cold_quad * (-d)**2
    return 1.0 + min(float(max_penalty), float(pen))

def wind_components(wind_speed_ms, wind_dir_from_deg, course_bearing_deg):
    if wind_speed_ms is None or wind_dir_from_deg is None:
        return 0.0, 0.0
    ws = float(wind_speed_ms)
    if ws <= 0:
        return 0.0, 0.0
    wind_to = (float(wind_dir_from_deg) + 180.0) % 360.0
    delta = math.radians((wind_to - course_bearing_deg + 540.0) % 360.0 - 180.0)
    along = ws * math.cos(delta)
    return float(max(0.0, -along)), float(max(0.0, along))

def wind_multiplier(head_ms, tail_ms, pace_s_per_km, drag_coeff, tail_credit, cap_head, cap_tail):
    pace = max(150.0, float(pace_s_per_km))
    v_run = 1000.0 / pace
    w_along = float(head_ms) - float(tail_ms)
    v_rel = max(0.0, v_run + w_along)
    base = max(1e-9, v_run ** 2)
    extra = (v_rel**2 - v_run**2) / base
    if extra < 0:
        extra = float(tail_credit) * extra
    mult = 1.0 + float(drag_coeff) * extra
    return float(max(1.0 + cap_tail, min(1.0 + cap_head, mult)))

def wind_gate(grade_pct, g1=2.0, g2=8.0, min_gate=0.25):
    g = max(0.0, float(grade_pct))
    if g <= g1:
        return 1.0
    if g >= g2:
        return float(min_gate)
    return float(1.0 - (g - g1) / (g2 - g1) * (1.0 - min_gate))

def cap_combined(mult_total, grade_pct, base_cap, extra_per_pct, max_cap):
    g = max(0.0, float(grade_pct))
    cap = min(float(max_cap), float(base_cap) + float(extra_per_pct) * g)
    return min(float(mult_total), 1.0 + cap)

def fatigue_multiplier(d_plus_cum, dist_cum, d_plus_total, dist_total, rate_pct, mode):
    if rate_pct <= 0:
        return 1.0
    rate = rate_pct / 100.0
    prog_dist  = min(1.0, dist_cum  / max(1.0, dist_total))
    prog_dplus = min(1.0, d_plus_cum / max(1.0, d_plus_total))
    dplus_ratio = d_plus_total / max(1.0, dist_total)
    w_dplus = min(0.8, dplus_ratio * 10.0)
    if mode == "distance":
        prog = prog_dist
    elif mode == "d_plus":
        prog = prog_dplus
    else:
        prog = w_dplus * prog_dplus + (1.0 - w_dplus) * prog_dist
    k = 2.0
    factor = (math.exp(k * prog) - 1.0) / (math.exp(k) - 1.0)
    return 1.0 + rate * factor


# ══════════════════════════════════════════════════════════════
# MÉTÉO CODE 2 (fitdecode-based tabs)
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_weather_openmeteo_day(lat, lon, date_obj):
    try:
        if lat is None or lon is None or date_obj is None:
            return None
        date_str = date_obj.strftime("%Y-%m-%d")
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={float(lat)}&longitude={float(lon)}"
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
    if lat is None or lon is None or start_dt is None or end_dt is None:
        return None, None, None
    try:
        if (end_dt - start_dt).total_seconds() < 300:
            start_dt -= timedelta(minutes=2)
            end_dt   += timedelta(minutes=2)
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

def get_segment_weather(segment_df):
    try:
        if segment_df is None or segment_df.empty:
            return None, None, None
        if "lat" not in segment_df.columns or "lon" not in segment_df.columns:
            return None, None, None
        lat = segment_df["lat"].dropna().iloc[0]
        lon = segment_df["lon"].dropna().iloc[0]
        if not np.isfinite(lat) or not np.isfinite(lon):
            return None, None, None
        start_dt = segment_df["timestamp"].iloc[0]
        end_dt   = segment_df["timestamp"].iloc[-1]
        return get_avg_weather_for_period(lat, lon, start_dt, end_dt)
    except Exception:
        return None, None, None


# ══════════════════════════════════════════════════════════════
# LECTURE FICHIERS — fitdecode (onglets 1 et 2)
# ══════════════════════════════════════════════════════════════

def _fit_semicircles_to_deg(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x) * (180.0 / (2**31))
    except Exception:
        return None

def load_activity(file):
    """Charge un fichier FIT, GPX, CSV ou TCX -> DataFrame harmonisé (FC obligatoire)."""
    name = file.name.lower()
    if name.endswith(".csv"):
        file.seek(0)
        df = pd.read_csv(file)
    elif name.endswith(".fit"):
        data = []
        try:
            file.seek(0)
            if not HAS_FITDECODE:
                raise ValueError("fitdecode non disponible — réinstallez les dépendances.")
            with fitdecode.FitReader(file) as fit:
                for frame in fit:
                    if isinstance(frame, fitdecode.records.FitDataMessage) and frame.name == "record":
                        row = {f.name: f.value for f in frame.fields}
                        data.append(row)
            df = pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Erreur FIT : {e}")
        if "position_lat" in df.columns and "lat" not in df.columns:
            df["lat"] = df["position_lat"].apply(_fit_semicircles_to_deg)
        if "position_long" in df.columns and "lon" not in df.columns:
            df["lon"] = df["position_long"].apply(_fit_semicircles_to_deg)
        if "enhanced_altitude" in df.columns and "alt" not in df.columns:
            df["alt"] = pd.to_numeric(df["enhanced_altitude"], errors="coerce")
        elif "altitude" in df.columns and "alt" not in df.columns:
            df["alt"] = pd.to_numeric(df["altitude"], errors="coerce")
        if "distance" in df.columns:
            df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
        if "enhanced_speed" in df.columns:
            df["enhanced_speed"] = pd.to_numeric(df["enhanced_speed"], errors="coerce")
        if "speed" in df.columns:
            df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
        if "heart_rate" in df.columns:
            df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    elif name.endswith(".gpx"):
        file.seek(0)
        gpx = gpxpy.parse(file)
        data = []
        for trk in gpx.tracks:
            for seg in trk.segments:
                for pt in seg.points:
                    data.append({"timestamp": pt.time, "lat": pt.latitude,
                                 "lon": pt.longitude, "alt": pt.elevation})
        df = pd.DataFrame(data)
    elif name.endswith(".tcx"):
        try:
            file.seek(0)
            content = file.read().decode("utf-8", errors="ignore")
            root = ET.fromstring(content)
            data = []
            for tp in root.findall(".//{*}Trackpoint"):
                t  = tp.find(".//{*}Time")
                hr = tp.find(".//{*}HeartRateBpm/{*}Value")
                dist = tp.find(".//{*}DistanceMeters")
                alt  = tp.find(".//{*}AltitudeMeters")
                lat  = tp.find(".//{*}Position/{*}LatitudeDegrees")
                lon  = tp.find(".//{*}Position/{*}LongitudeDegrees")
                powv = tp.find(".//{*}Watts")
                data.append({
                    "timestamp":  t.text if t is not None else None,
                    "heart_rate": float(hr.text) if hr is not None else None,
                    "distance":   float(dist.text) if dist is not None else None,
                    "alt":        float(alt.text) if alt is not None else None,
                    "power":      float(powv.text) if powv is not None else None,
                    "lat":        float(lat.text) if lat is not None else None,
                    "lon":        float(lon.text) if lon is not None else None,
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
    # Harmonisation FC
    if "heart_rate" not in df.columns:
        for cand in ["hr", "heart_rate_bpm", "heartrate", "bpm"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "heart_rate"})
                break
    if "heart_rate" not in df.columns:
        raise ValueError("Pas de FC détectée dans ce fichier.")
    # Timestamp -> UTC naive
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    for c in ["heart_rate", "speed", "enhanced_speed", "power", "distance", "lat", "lon", "alt"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "heart_rate"]).reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════
# LISSAGE CARDIO + ANALYSES (Code 2)
# ══════════════════════════════════════════════════════════════

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
    window_size = max(1, int(window_sec / max(step, 0.001)))
    df["hr_smooth"] = pd.to_numeric(df[hr_col], errors="coerce").rolling(window_size, min_periods=1).mean()
    sp = get_speed_col(df)
    if sp:
        df["speed_smooth"] = pd.to_numeric(df[sp], errors="coerce").rolling(window_size, min_periods=1).mean()
    if "power" in df.columns:
        df["power_smooth"] = pd.to_numeric(df["power"], errors="coerce").rolling(window_size, min_periods=1).mean()
    pauses = int((df["delta_t"] > 2 * median_step).sum())
    return df, window_sec, total_dur, pauses

def analyze_heart_rate(df):
    hr = df["hr_smooth"].dropna()
    mean_hr = float(hr.mean()) if len(hr) else np.nan
    max_hr  = float(hr.max())  if len(hr) else np.nan
    min_hr  = float(hr.min())  if len(hr) else np.nan
    if len(df) >= 2 and df["time_s"].nunique() >= 2:
        slope, _, _, _, _ = linregress(df["time_s"], df["hr_smooth"])
    else:
        slope = 0.0
    drift_per_min = float(slope * 60)
    drift_percent = (drift_per_min / mean_hr) * 100 if (mean_hr and mean_hr > 0) else np.nan
    stats = {
        "FC moyenne (bpm)":   round(mean_hr, 1) if math.isfinite(mean_hr) else None,
        "FC max (bpm)":       round(max_hr, 1)  if math.isfinite(max_hr)  else None,
        "FC min (bpm)":       round(min_hr, 1)  if math.isfinite(min_hr)  else None,
        "Dérive (bpm/min)":   round(drift_per_min, 4),
        "Dérive (%/min)":     round(drift_percent, 4) if not np.isnan(drift_percent) else None,
        "Durée segment (s)":  round(float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]), 1) if len(df) else 0.0,
    }
    return stats, float(drift_per_min), (None if np.isnan(drift_percent) else float(drift_percent))

def analyze_speed_kinetics(df):
    sp_col = get_speed_col(df)
    if sp_col is None or df[sp_col].dropna().empty or df["time_s"].nunique() < 2:
        return None, None
    slope, _, _, _, _ = linregress(df["time_s"], df[sp_col])
    drift_per_min = float(slope * 60)
    mean_speed = float(df[sp_col].mean())
    drift_percent = (drift_per_min / mean_speed * 100) if mean_speed > 0 else None
    return round(drift_per_min, 4), (round(drift_percent, 4) if drift_percent is not None else None)


# ══════════════════════════════════════════════════════════════
# GRAPHIQUES (Code 2)
# ══════════════════════════════════════════════════════════════

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

def pace_formatter(v, pos):
    if v is None or not math.isfinite(v) or v <= 0:
        return ""
    m = int(v)
    s = int(round((v - m) * 60))
    if s == 60:
        m += 1; s = 0
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
    ax_pace = None; ax_pow = None
    tt = df["time_s"].values - t0
    if show_fc and "hr_smooth" in df.columns:
        ax.plot(tt, df["hr_smooth"], color=c_fc, linewidth=linewidth, label=f"{who} • FC (bpm)")
        ax.set_ylabel("FC (bpm)")
    if show_pace and "speed_smooth" in df.columns:
        pace_series = compute_pace_series_from_speed(df)
        if pace_series is not None:
            ax_pace = add_pace_axis(ax)
            ax_pace.plot(tt, pace_series, color=c_pace, linewidth=linewidth,
                         label=f"{who} • Allure (min/km)")
    if show_power and "power_smooth" in df.columns:
        ax_pow = add_power_axis(ax, offset=60)
        ax_pow.plot(tt, df["power_smooth"], color=c_pow, linewidth=linewidth,
                    label=f"{who} • Puissance (W)")
    return ax, ax_pace, ax_pow


# ══════════════════════════════════════════════════════════════
# RECALIBRAGE + INDEX CINÉTIQUE (Code 2)
# ══════════════════════════════════════════════════════════════

def temp_multiplier_nonlin(temp, opt_temp=12.0, k_hot=0.002, k_cold=0.002):
    try:
        if temp is None:
            return 1.0
        diff = float(temp) - float(opt_temp)
        mult = (1.0 + float(k_hot) * diff) if diff > 0 else (1.0 + float(k_cold) * (-diff))
        return max(0.1, float(mult))
    except Exception:
        return 1.0

def elevation_factor_from_dup_ddn(D_up, D_down, segment_length_m=1000.0, k_up=1.040, k_down=0.996):
    try:
        seg_len = float(segment_length_m) if segment_length_m and segment_length_m > 0 else 1000.0
        up_factor   = (float(k_up) - 1.0) * (float(D_up) / seg_len)
        down_factor = (1.0 - float(k_down)) * (float(D_down) / seg_len)
        factor = 1.0 + up_factor + down_factor
        return max(0.01, float(factor))
    except Exception:
        return 1.0

def recalibrate_time_to_ideal_c2(time_seconds_raw, D_up, D_down, distance_m,
                                   temp_real,
                                   k_up=1.040, k_down=0.996,
                                   k_temp_hot=0.002, k_temp_cold=0.002,
                                   opt_temp=12.0):
    try:
        t = float(time_seconds_raw)
        if not math.isfinite(t) or t <= 0:
            return None
        dist = float(distance_m) if distance_m and distance_m > 0 else 1000.0
        fact_elev = elevation_factor_from_dup_ddn(D_up, D_down,
                                                   segment_length_m=dist,
                                                   k_up=k_up, k_down=k_down)
        t_no_elev = t / fact_elev
        mult_real = temp_multiplier_nonlin(temp_real, opt_temp=opt_temp,
                                           k_hot=k_temp_hot, k_cold=k_temp_cold)
        t_no_temp = t_no_elev / (mult_real if mult_real != 0 else 1.0)
        return max(0.0, float(t_no_temp))
    except Exception:
        return None

def compute_index_cinetique(drift_short_pct, drift_long_pct,
                             drift_short_bpm, drift_long_bpm):
    use_pct = (drift_short_pct is not None
               and drift_long_pct is not None
               and drift_short_pct != 0)
    if use_pct:
        IC = 1.0 - (drift_long_pct / drift_short_pct)
        unite = "%/min"
    else:
        if drift_short_bpm is None or drift_long_bpm is None or drift_short_bpm == 0:
            return None, None, "Index non calculable.", None, None
        IC = 1.0 - (drift_long_bpm / drift_short_bpm)
        unite = "bpm/min"
    if IC >= 0.70:
        titre = "Très bonne stabilité"; msg = "Excellente endurance."
        seances = ["Tempo long", "Blocs soutenus", "Z2 volumineux"]
    elif IC >= 0.40:
        titre = "Bonne stabilité"; msg = "Bonne base aérobie."
        seances = ["Intervalles moyens", "Tempo"]
    elif IC >= 0.15:
        titre = "Stabilité moyenne"; msg = "À renforcer."
        seances = ["Progressifs", "Z2 + tempo"]
    else:
        titre = "Stabilité faible"; msg = "Reconstruction aérobie."
        seances = ["Z2 majoritaire", "Côtes faciles"]
    return float(IC), unite, msg, None, {"titre": titre, "seances": seances}

def style_metrics_table(df_table):
    def row_class(row):
        m = str(row.get("Métrique", "")).lower()
        if "réelle" in m or "brute" in m:
            return "tr-raw"
        if "recalibrée" in m or "idéale" in m:
            return "tr-ideal"
        return "tr-general"
    classes = pd.DataFrame(
        [[row_class(row)] * len(df_table.columns) for _, row in df_table.iterrows()],
        index=df_table.index, columns=df_table.columns,
    )
    return df_table.style.set_td_classes(classes)

def fit_power_law_from_tests(valid_tests, use_ideal=False):
    try:
        if len(valid_tests) < 2:
            return None, None
        if use_ideal:
            V = np.array([t["dist_m"] / t["t_s_ideal"]
                          for t in valid_tests if t.get("t_s_ideal") and t["t_s_ideal"] > 0])
            T = np.array([t["t_s_ideal"]
                          for t in valid_tests if t.get("t_s_ideal") and t["t_s_ideal"] > 0])
        else:
            V = np.array([t["dist_m"] / t["t_s_raw"]
                          for t in valid_tests if t.get("t_s_raw") and t["t_s_raw"] > 0])
            T = np.array([t["t_s_raw"]
                          for t in valid_tests if t.get("t_s_raw") and t["t_s_raw"] > 0])
        mask = np.isfinite(V) & np.isfinite(T) & (V > 0) & (T > 0)
        if mask.sum() < 2:
            return None, None
        X = np.log(V[mask]); Y = np.log(T[mask])
        slope, intercept = np.polyfit(X, Y, 1)
        k = -float(slope); A = float(np.exp(intercept))
        if A > 0 and k > 0:
            return A, k
        return None, None
    except Exception:
        return None, None

def build_hybrid_holding_table(VC_kmh, D_prime_m, A_pl, k_pl,
                                pct_low=80, pct_high=130, step=2):
    if VC_kmh is None or VC_kmh <= 0:
        return None
    VC_ms = VC_kmh / 3.6
    rows = []
    if A_pl and k_pl:
        for p in range(pct_low, 100, step):
            v = VC_kmh * p / 100
            Tlim = A_pl * (v / 3.6) ** (-k_pl)
            rows.append({"% VC": f"{p}%", "Modèle": "Power Law",
                         "Vitesse (km/h)": round(v, 2), "Temps limite": seconds_to_hms(Tlim)})
    if D_prime_m and D_prime_m > 0:
        for p in range(102, pct_high + 1, step):
            v_ms = VC_kmh * p / 100 / 3.6
            if v_ms > VC_ms:
                Tlim = D_prime_m / (v_ms - VC_ms)
                rows.append({"% VC": f"{p}%", "Modèle": "D\u2032",
                             "Vitesse (km/h)": round(v_ms * 3.6, 2), "Temps limite": seconds_to_hms(Tlim)})
    return pd.DataFrame(rows) if rows else None



# ══════════════════════════════════════════════════════════════
# MÉTÉO v3 (prédiction GPX)
# ══════════════════════════════════════════════════════════════

class SimplePoint:
    def __init__(self, lat, lon, elev=0.0, time=None):
        self.latitude  = float(lat)
        self.longitude = float(lon)
        self.elevation = float(elev) if elev is not None else 0.0
        self.time = time

    def distance_3d(self, other):
        h = haversine_m(self.latitude, self.longitude, other.latitude, other.longitude)
        v = self.elevation - other.elevation
        return math.sqrt(h*h + v*v)


@st.cache_data(show_spinner=False)
def get_weather_minutely(lat, lon, dt_local_naive, tz_name=TZ_NAME_DEFAULT):
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               "&hourly=temperature_2m,relativehumidity_2m,wind_speed_10m,wind_direction_10m"
               f"&timezone={tz_name}")
        data = requests.get(url, timeout=20).json()
        if "hourly" not in data:
            return None
        times = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
        temps = data["hourly"]["temperature_2m"]
        winds = data["hourly"]["wind_speed_10m"]
        hums  = data["hourly"]["relativehumidity_2m"]
        wdirs = data["hourly"]["wind_direction_10m"]
        dt = dt_local_naive
        for i in range(len(times) - 1):
            if times[i] <= dt <= times[i+1]:
                r = (dt - times[i]).total_seconds() / max(1.0, (times[i+1]-times[i]).total_seconds())
                a1, a2 = float(wdirs[i]) % 360, float(wdirs[i+1]) % 360
                da = (a2 - a1 + 540.0) % 360.0 - 180.0
                return {
                    "temp": temps[i] + r*(temps[i+1]-temps[i]),
                    "wind": winds[i] + r*(winds[i+1]-winds[i]),
                    "humidity": hums[i] + r*(hums[i+1]-hums[i]),
                    "wind_dir": (a1 + r*da) % 360.0,
                }
        idx = min(range(len(times)), key=lambda i: abs(times[i]-dt))
        return {"temp": float(temps[idx]), "wind": float(winds[idx]),
                "humidity": float(hums[idx]), "wind_dir": float(wdirs[idx])}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_weather_archive_day(lat, lon, date_obj, tz_name=TZ_NAME_DEFAULT):
    try:
        ds = date_obj.strftime("%Y-%m-%d")
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               f"&start_date={ds}&end_date={ds}"
               "&hourly=temperature_2m,relativehumidity_2m,wind_speed_10m,wind_direction_10m"
               f"&timezone={tz_name}")
        data = requests.get(url, timeout=20).json()
        if "hourly" not in data:
            return None
        return ([datetime.fromisoformat(t) for t in data["hourly"]["time"]],
                data["hourly"]["temperature_2m"],
                data["hourly"]["wind_speed_10m"],
                data["hourly"]["relativehumidity_2m"],
                data["hourly"]["wind_direction_10m"])
    except Exception:
        return None


def get_avg_weather_v3(lat, lon, start_dt, end_dt, tz_name=TZ_NAME_DEFAULT):
    if start_dt is None or end_dt is None:
        return None, None, None
    if (end_dt - start_dt).total_seconds() < 300:
        start_dt -= timedelta(minutes=2)
        end_dt   += timedelta(minutes=2)
    res = get_weather_archive_day(lat, lon, start_dt.date(), tz_name=tz_name)
    if not res:
        return None, None, None
    times, temps, winds, hums, _ = res
    selT = [T for t, T in zip(times, temps) if start_dt <= t <= end_dt]
    selW = [W for t, W in zip(times, winds) if start_dt <= t <= end_dt]
    selH = [H for t, H in zip(times, hums)  if start_dt <= t <= end_dt]
    if not selT:
        idx = min(range(len(times)), key=lambda i: abs(times[i]-start_dt))
        return float(temps[idx]), float(winds[idx]), float(hums[idx])
    return float(np.mean(selT)), float(np.mean(selW)), float(np.mean(selH))


# ══════════════════════════════════════════════════════════════
# DEM
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Correction altimétrique DEM...")
def fetch_dem_elevations(lats, lons, dataset="srtm30m"):
    try:
        locs = "|".join(f"{la},{lo}" for la, lo in zip(lats, lons))
        data = requests.get(f"https://api.opentopodata.org/v1/{dataset}?locations={locs}", timeout=30).json()
        if data.get("status") != "OK":
            return [None] * len(lats)
        return [r.get("elevation") for r in data["results"]]
    except Exception:
        return [None] * len(lats)


def correct_elevations_dem(points, max_points=100, dataset="srtm30m"):
    n = len(points)
    if n < 2:
        return np.array([getattr(p, "elevation", 0.0) or 0.0 for p in points])
    step = max(1, n // max_points)
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    lats = tuple(points[i].latitude for i in indices)
    lons = tuple(points[i].longitude for i in indices)
    dem = fetch_dem_elevations(lats, lons, dataset=dataset)
    cum_all = [0.0]
    for i in range(1, n):
        cum_all.append(cum_all[-1] + haversine_m(
            points[i-1].latitude, points[i-1].longitude,
            points[i].latitude, points[i].longitude))
    cum_sub = [cum_all[i] for i in indices]
    valid = [(d, e) for d, e in zip(cum_sub, dem) if e is not None]
    if len(valid) < 2:
        return np.array([getattr(p, "elevation", 0.0) or 0.0 for p in points])
    return np.interp(cum_all, [v[0] for v in valid], [v[1] for v in valid])


# ══════════════════════════════════════════════════════════════
# PARSING v3 (fitparse pour références Tab 0)
# ══════════════════════════════════════════════════════════════

def analyze_hr_simple(hr_records):
    hrs = [h for h in hr_records if h is not None and 50 <= h <= 220]
    if len(hrs) < 10:
        return {"hr_max": None, "hr_avg": None, "hr_drift": None, "reliability": "inconnue"}
    arr = np.array(hrs, dtype=float)
    n = len(arr)
    hr_max = float(np.percentile(arr, 95))
    hr_avg = float(np.mean(arr))
    q1, q3 = int(n*0.25), int(n*0.75)
    drift = float(np.mean(arr[q3:])) - float(np.mean(arr[:q1]))
    reliability = "haute" if drift < 5 else ("moyenne" if drift < 12 else "basse (dérive cardiaque forte)")
    return {"hr_max": round(hr_max), "hr_avg": round(hr_avg),
            "hr_drift": round(drift, 1), "hr_threshold_est": round(hr_max * 0.88),
            "reliability": reliability}


def parse_gpx_points(file):
    try:
        file.seek(0)
        gpx = gpxpy.parse(file)
        pts = [p for track in gpx.tracks for seg in track.segments for p in seg.points]
        return gpx, pts
    except Exception as e:
        st.error(f"Erreur GPX : {e}")
        return None, []


def parse_fit_v3(file, tz_name=TZ_NAME_DEFAULT):
    """Parsing FIT via fitparse pour les références v3 (Tab 0)."""
    try:
        file.seek(0)
        fit = FitFile(file)
        fit.parse()
        records, times_pts, hr_records = [], [], []
        start_global = elapsed_global = None
        for msg in fit.get_messages("session"):
            vals = {d.name: d.value for d in msg}
            if isinstance(vals.get("start_time"), datetime):
                start_global = vals["start_time"].replace(tzinfo=None)
            if isinstance(vals.get("total_elapsed_time"), (int, float)):
                elapsed_global = float(vals["total_elapsed_time"])
        for msg in fit.get_messages("record"):
            vals = {d.name: d.value for d in msg}
            lat_r = vals.get("position_lat")
            lon_r = vals.get("position_long")
            if lat_r is None or lon_r is None:
                continue
            lat = lat_r * (180 / 2**31)
            lon = lon_r * (180 / 2**31)
            ts = vals.get("timestamp")
            dt = ts.replace(tzinfo=None) if isinstance(ts, datetime) else None
            alt = (vals.get("enhanced_altitude") or vals.get("altitude") or
                   vals.get("baro_altitude") or vals.get("gps_altitude") or 0.0)
            dist = float(vals.get("distance") or 0.0)
            hr = vals.get("heart_rate")
            hr_records.append(int(hr) if hr is not None else None)
            records.append((lat, lon, float(alt), dist))
            times_pts.append(dt)
        if not records:
            return None
        df = pd.DataFrame(records, columns=["lat", "lon", "elev", "dist"])
        valid_t = [t for t in times_pts if t is not None]
        if len(valid_t) >= 2:
            start_dt, end_dt = min(valid_t), max(valid_t)
        elif start_global and elapsed_global:
            start_dt = start_global
            end_dt = start_global + timedelta(seconds=elapsed_global)
        else:
            start_dt = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=1)
            end_dt = start_dt + timedelta(minutes=5)
        avgT, avgW, avgH = get_avg_weather_v3(records[0][0], records[0][1], start_dt, end_dt, tz_name)
        elev_arr = df["elev"].values
        dup = float(np.sum(np.clip(np.diff(elev_arr), 0, None))) if elev_arr.size >= 2 else 0.0
        ddn = float(-np.sum(np.clip(np.diff(elev_arr), None, 0))) if elev_arr.size >= 2 else 0.0
        return {
            "points": [{"lat": r[0], "lon": r[1], "elev": r[2], "dist": r[3], "time": t}
                       for r, t in zip(records, times_pts)],
            "distance": float(df["dist"].max()),
            "D_up": dup, "D_down": ddn,
            "duration_hms": seconds_to_hms((end_dt - start_dt).total_seconds()),
            "avg_temp": avgT, "avg_wind": avgW, "avg_humidity": avgH,
            "hr_analysis": analyze_hr_simple(hr_records),
        }
    except Exception as e:
        st.error(f"Erreur FIT : {e}")
        return None


def parse_tcx_v3(file, tz_name=TZ_NAME_DEFAULT):
    try:
        file.seek(0)
        root = ET.parse(file).getroot()
    except Exception:
        return None
    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    pts, times, elevs = [], [], []
    for tp in root.findall(".//tcx:Trackpoint", ns):
        lat = tp.find("tcx:Position/tcx:LatitudeDegrees", ns)
        lon = tp.find("tcx:Position/tcx:LongitudeDegrees", ns)
        if lat is None or lon is None:
            continue
        ele = tp.find("tcx:AltitudeMeters", ns)
        tim = tp.find("tcx:Time", ns)
        elev = float(ele.text) if ele is not None else 0.0
        try:
            t = datetime.fromisoformat(tim.text.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            t = None
        pts.append(SimplePoint(float(lat.text), float(lon.text), elev, t))
        times.append(t); elevs.append(elev)
    if len(pts) < 2:
        return None
    vt = [t for t in times if t is not None]
    start_dt = vt[0] if vt else datetime.now() - timedelta(days=1)
    end_dt = vt[-1] if len(vt) > 1 else start_dt + timedelta(minutes=5)
    avgT, avgW, avgH = get_avg_weather_v3(pts[0].latitude, pts[0].longitude, start_dt, end_dt, tz_name)
    total = sum(pts[i].distance_3d(pts[i-1]) for i in range(1, len(pts)))
    dup, ddn = compute_dplus_dminus(elevs)
    return {"points": pts, "distance": round(total),
            "D_up": round(dup, 1), "D_down": round(ddn, 1),
            "duration_hms": seconds_to_hms((end_dt - start_dt).total_seconds()),
            "avg_temp": avgT, "avg_wind": avgW, "avg_humidity": avgH, "hr_analysis": None}


def extract_segment(points, start_td, end_td):
    def get_t(p):
        return p.get("time") if isinstance(p, dict) else getattr(p, "time", None)
    ts = [get_t(p) for p in points if get_t(p) is not None]
    if len(ts) < 2:
        return points
    t0 = min(ts)
    seg = [p for p in points if get_t(p) is not None
           and t0 + start_td <= get_t(p) <= t0 + end_td + timedelta(seconds=1)]
    return seg if len(seg) >= 2 else points


# ══════════════════════════════════════════════════════════════
# MODÈLE LOG-LOG (Riegel) — v3
# ══════════════════════════════════════════════════════════════

def fit_loglog(refs):
    X, Y = [], []
    for r in refs:
        d_m = safe_float(r.get("distance", 0))
        t = r.get("temps")
        secs = float(t) if isinstance(t, (int, float, np.number)) else hms_to_seconds(str(t))
        if d_m <= 0 or secs <= 0:
            continue
        X.append(math.log(d_m / 1000.0)); Y.append(math.log(secs))
    if len(X) >= 2:
        K, loga = np.polyfit(X, Y, 1)
        K = float(max(0.85, min(1.25, K)))
        a = math.exp(float(loga))
        return (a if 0 < a < 1e7 else 240.0), K
    elif len(X) == 1:
        return math.exp(Y[0]) / (math.exp(X[0])), 1.0
    return 240.0, 1.0

def predict_flat(dist_m, a, K):
    return float(a) * ((dist_m / 1000.0) ** float(K))

def crossval_loo(refs_prepared):
    n = len(refs_prepared)
    if n < 3:
        return None
    rows = []
    for i in range(n):
        train = [r for j, r in enumerate(refs_prepared) if j != i]
        test = refs_prepared[i]
        a_cv, K_cv = fit_loglog(train)
        pred_s = predict_flat(test["distance"], a_cv, K_cv)
        actual_s = float(test["temps"])
        rows.append({
            "Réf": i+1,
            "Distance (km)": round(test["distance"]/1000.0, 2),
            "Temps réel": seconds_to_hms(actual_s),
            "Temps prédit": seconds_to_hms(pred_s),
            "Erreur (s)": round(pred_s - actual_s, 0),
            "Erreur (%)": round((pred_s - actual_s) / actual_s * 100.0, 2) if actual_s > 0 else 0,
        })
    df_cv = pd.DataFrame(rows)
    mae  = float(np.mean(np.abs(df_cv["Erreur (s)"].values)))
    mape = float(np.mean(np.abs(df_cv["Erreur (%)"].values)))
    return df_cv, mae, mape

def elev_factor_global(D_up_m, D_down_m, dist_m,
                        k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down):
    dist = max(1e-6, float(dist_m))
    g_up = float(D_up_m) / dist; g_dn = float(D_down_m) / dist
    g0u = max(1e-6, float(g0_up) / 100.0); g0d = max(1e-6, float(g0_down) / 100.0)
    up_term    = float(k_up)   * math.tanh(g_up / g0u) * g0u
    down_bonus = min(float(k_down) * math.tanh(g_dn / g0d) * g0d, abs(float(down_cap)))
    mult = 1.0 + up_term - down_bonus
    mult = min(mult, 1.0 + float(max_up)); mult = max(mult, 1.0 + float(max_down))
    return max(0.01, float(mult))

def recalibrate_ref_to_ideal(ref, opt_temp, use_wbgt, cold_quad, hot_quad, temp_max_penalty,
                              k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down,
                              elev_ref_power, temp_ref_power):
    secs = hms_to_seconds(ref.get("temps")) if ref.get("temps") is not None else 0
    D_up  = safe_float(ref.get("D_up", 0.0))
    D_down = safe_float(ref.get("D_down", 0.0))
    dist  = max(1.0, safe_float(ref.get("distance", 1000.0)))
    f_elev = elev_factor_global(D_up, D_down, dist, k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down)
    secs_no_elev = secs / (f_elev ** float(elev_ref_power))
    temp_real = ref.get("avg_temp")
    hum_real  = safe_float(ref.get("avg_humidity", 50.0), 50.0)
    if temp_real is not None:
        temp_eff = effective_temp(temp_real, hum_real, use_wbgt)
        f_temp = temp_multiplier_v3(temp_eff, opt_temp, cold_quad, hot_quad, temp_max_penalty)
        secs_no_temp = secs_no_elev / (max(0.01, f_temp) ** float(temp_ref_power))
    else:
        secs_no_temp = secs_no_elev
    return max(0.0, float(secs_no_temp))

def prepare_refs(refs_input, use_recalibrated, opt_temp, use_wbgt,
                 cold_quad, hot_quad, temp_max_penalty,
                 k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down,
                 elev_ref_power, temp_ref_power):
    out = []
    for r in refs_input:
        d = safe_float(r.get("distance", 0.0))
        raw_t = r.get("duration_hms_file") or r.get("temps", "0:00:00")
        if use_recalibrated:
            secs = recalibrate_ref_to_ideal(
                ref={**r, "temps": raw_t},
                opt_temp=opt_temp, use_wbgt=use_wbgt,
                cold_quad=cold_quad, hot_quad=hot_quad, temp_max_penalty=temp_max_penalty,
                k_up=k_up, k_down=k_down, down_cap=down_cap,
                g0_up=g0_up, g0_down=g0_down, max_up=max_up, max_down=max_down,
                elev_ref_power=elev_ref_power, temp_ref_power=temp_ref_power,
            )
        else:
            secs = float(hms_to_seconds(raw_t))
        out.append({"distance": float(d), "temps": float(secs)})
    return out

def apply_ultra_pacing(t_raw, d_end_m, seg_len_m, total_corr_m, amp_pct):
    if len(t_raw) == 0 or amp_pct <= 0:
        return t_raw
    total_corr_m = max(1e-9, float(total_corr_m))
    d_mid = np.asarray(d_end_m) - 0.5 * np.asarray(seg_len_m)
    prog = np.clip(d_mid / total_corr_m, 0.0, 1.0)
    A = amp_pct / 100.0
    mult = 1.0 + A * (2.0 * prog - 1.0)
    t_adj = np.asarray(t_raw) * mult
    s_raw = np.sum(t_raw); s_adj = np.sum(t_adj)
    if s_raw > 0 and s_adj > 0:
        t_adj *= s_raw / s_adj
    return t_adj



# ══════════════════════════════════════════════════════════════
# PRÉDICTION PRINCIPALE (v3)
# ══════════════════════════════════════════════════════════════

def run_prediction(
    distance_cible_km, refs_input, points, date_course, heure_course,
    use_recalibrated, opt_temp, use_wbgt,
    cold_quad, hot_quad, temp_max_penalty, temp_power,
    elev_ref_power, temp_ref_power,
    apply_grade, use_minetti, minetti_weight,
    k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down,
    elev_smooth_window, grade_power,
    apply_altitude, altitude_ref_m,
    apply_wind, wind_mode, wind_smooth_km,
    drag_coeff, tail_credit, wind_cap_head, wind_cap_tail, wind_power,
    wind_gate_g1, wind_gate_g2, wind_gate_min,
    base_cap, extra_per_pct, max_cap,
    apply_fatigue, fatigue_rate, fatigue_mode,
    apply_ultra, ultra_amp,
    objective_hms,
    show_smooth_pace, smooth_window_km,
    dem_elevations,
    tz_name=TZ_NAME_DEFAULT,
):
    if not points or len(points) < 2:
        raise ValueError("GPX invalide ou trop court.")

    if dem_elevations is not None and len(dem_elevations) == len(points):
        elev_arr = np.array([e if e is not None else 0.0 for e in dem_elevations], dtype=float)
    else:
        elev_arr = np.array([getattr(p, "elevation", 0.0) or 0.0 for p in points], dtype=float)

    total_m = 0.0; cum = [0.0]
    for i in range(1, len(points)):
        total_m += haversine_m(points[i-1].latitude, points[i-1].longitude,
                               points[i].latitude, points[i].longitude)
        cum.append(total_m)
    dist_gpx_km = total_m / 1000.0
    if not distance_cible_km:
        distance_cible_km = dist_gpx_km
    fac = distance_cible_km / max(dist_gpx_km, 1e-9)
    total_corr = total_m * fac
    dists_corr = np.array(cum, dtype=float) * fac

    if elev_arr.size != dists_corr.size:
        xs = np.linspace(0, total_m, elev_arr.size)
        elev_arr = np.interp(np.linspace(0, total_m, dists_corr.size), xs, elev_arr)

    w = int(elev_smooth_window)
    if w % 2 == 0: w += 1
    if w >= 3 and elev_arr.size >= w:
        elev_s = np.convolve(elev_arr, np.ones(w)/w, mode="same")
    else:
        elev_s = elev_arr

    diffs_el = np.diff(elev_s)
    d_plus_total = float(np.sum(np.clip(diffs_el, 0, None)))
    avg_alt = float(np.mean(elev_s))

    refs_fit = prepare_refs(
        refs_input=refs_input, use_recalibrated=use_recalibrated,
        opt_temp=opt_temp, use_wbgt=use_wbgt,
        cold_quad=cold_quad, hot_quad=hot_quad, temp_max_penalty=temp_max_penalty,
        k_up=k_up, k_down=k_down, down_cap=down_cap,
        g0_up=g0_up, g0_down=g0_down, max_up=max_up, max_down=max_down,
        elev_ref_power=elev_ref_power, temp_ref_power=temp_ref_power,
    )
    a, K = fit_loglog(refs_fit)
    if objective_hms:
        obj_s = hms_to_seconds(objective_hms)
        d_km = distance_cible_km
        a = obj_s / (d_km ** K) if d_km > 0 else a
    base_total_s = predict_flat(int(distance_cible_km * 1000), a, K)
    base_s_per_km = base_total_s / max(distance_cible_km, 1e-9)

    alt_mult = altitude_vo2_multiplier(avg_alt, altitude_ref_m) if apply_altitude else 1.0

    km_marks = [i * 1000 for i in range(1, int(total_corr // 1000) + 1)]
    last = total_corr - int(total_corr // 1000) * 1000
    if last > 1e-6:
        km_marks.append(total_corr)

    lats_arr = np.array([p.latitude for p in points], dtype=float)
    lons_arr = np.array([p.longitude for p in points], dtype=float)
    dt_dep = datetime.combine(date_course, heure_course)

    pre = []
    cum_t = 0.0; cum_dp = 0.0; cum_dist = 0.0

    for i, d in enumerate(km_marks):
        seg_len = 1000.0
        if i == len(km_marks) - 1 and last > 1e-6:
            seg_len = d - (km_marks[-2] if len(km_marks) >= 2 else 0)
        e_cur = float(np.interp(d, dists_corr, elev_s))
        e_prv = float(np.interp(max(d - seg_len, 0), dists_corr, elev_s)) if i > 0 else e_cur
        grade = (e_cur - e_prv) / max(1e-6, seg_len) * 100.0
        seg_dp = max(0.0, e_cur - e_prv)
        cum_dp += seg_dp; cum_dist += seg_len
        t_flat = base_s_per_km * (seg_len / 1000.0)
        if apply_grade:
            gm = combined_grade_multiplier(grade, use_minetti, minetti_weight,
                                           k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down)
            t1 = t_flat * (gm ** grade_power)
        else:
            gm = 1.0; t1 = t_flat
        t2 = t1 * alt_mult
        if apply_fatigue and fatigue_rate > 0:
            fm = fatigue_multiplier(cum_dp, cum_dist, d_plus_total, total_corr, fatigue_rate, fatigue_mode)
        else:
            fm = 1.0
        t3 = t2 * fm
        passage_dt = dt_dep + timedelta(seconds=cum_t + t3/2.0)
        lat_s = float(np.interp(d, dists_corr, lats_arr))
        lon_s = float(np.interp(d, dists_corr, lons_arr))
        lat0  = float(np.interp(max(d-seg_len,0), dists_corr, lats_arr))
        lon0  = float(np.interp(max(d-seg_len,0), dists_corr, lons_arr))
        cap   = bearing_deg(lat0, lon0, lat_s, lon_s)
        meteo = get_weather_minutely(lat_s, lon_s, passage_dt, tz_name)
        temp_raw = meteo["temp"] if meteo else None
        wind_raw = meteo["wind"] if meteo else None
        hum_raw  = meteo["humidity"] if meteo else None
        wdir_raw = meteo.get("wind_dir") if meteo else None
        temp_eff_val = None
        if temp_raw is not None and hum_raw is not None:
            temp_eff_val = effective_temp(temp_raw, hum_raw, use_wbgt)
        if temp_eff_val is not None:
            tm = temp_multiplier_v3(temp_eff_val, opt_temp, cold_quad, hot_quad, temp_max_penalty)
            t4 = t3 * (tm ** temp_power)
        else:
            tm = 1.0; t4 = t3
        pace_local = (t4 / seg_len) * 1000.0 if seg_len > 0 else t4
        head, tail = wind_components(wind_raw, wdir_raw, cap)
        pre.append({
            "idx": i, "d": d, "seg_len": seg_len, "grade": grade, "grade_mult": gm,
            "seg_dp": seg_dp, "cum_dp": cum_dp, "fat_mult": fm, "alt_mult": alt_mult,
            "temp_raw": temp_raw, "temp_eff": temp_eff_val, "hum": hum_raw,
            "wind": wind_raw, "wdir": wdir_raw, "cap": cap,
            "head": head, "tail": tail, "temp_mult": tm,
            "t_flat": t_flat, "t_no_wind": t4, "pace_no_wind": pace_local,
        })
        cum_t += t4

    df_pre = pd.DataFrame(pre)

    if apply_wind and not df_pre.empty:
        if wind_mode == "Global":
            hg = float(np.median(df_pre["head"]))
            tg = float(np.median(df_pre["tail"]))
            pg = float(np.median(df_pre["pace_no_wind"]))
            wm_raw = wind_multiplier(hg, tg, pg, drag_coeff, tail_credit, wind_cap_head, wind_cap_tail)
            df_pre["wind_mult_raw"] = wm_raw
        else:
            w_s = int(max(1, wind_smooth_km)); w_s += (1 if w_s%2==0 else 0)
            hs = pd.Series(df_pre["head"]).rolling(w_s, center=True, min_periods=1).median()
            ts_ = pd.Series(df_pre["tail"]).rolling(w_s, center=True, min_periods=1).median()
            wms = [wind_multiplier(h, t, p, drag_coeff, tail_credit, wind_cap_head, wind_cap_tail)
                   for h, t, p in zip(hs, ts_, df_pre["pace_no_wind"])]
            df_pre["wind_mult_raw"] = wms
            df_pre["head_s"] = hs.values; df_pre["tail_s"] = ts_.values
    else:
        df_pre["wind_mult_raw"] = 1.0

    t_raw_arr = []
    wm_adj_list = []
    for _, row in df_pre.iterrows():
        wm = float(row["wind_mult_raw"])
        g  = float(row["grade"])
        gate = wind_gate(g, wind_gate_g1, wind_gate_g2, wind_gate_min)
        wm_gated = 1.0 + gate * (wm - 1.0)
        t_w = float(row["t_no_wind"]) * (wm_gated ** wind_power)
        mt = t_w / max(1e-9, float(row["t_flat"]))
        mt = cap_combined(mt, g, base_cap, extra_per_pct, max_cap)
        t_raw_arr.append(float(row["t_flat"]) * mt)
        wm_adj_list.append(wm_gated)
    df_pre["wind_mult_adj"] = wm_adj_list
    t_raw_arr = np.array(t_raw_arr, dtype=float)

    if apply_ultra and ultra_amp > 0:
        t_raw_arr = apply_ultra_pacing(t_raw_arr, df_pre["d"].values, df_pre["seg_len"].values, total_corr, ultra_amp)

    if objective_hms:
        s_obj = hms_to_seconds(objective_hms)
        s_sum = float(np.sum(t_raw_arr))
        t_raw_arr = t_raw_arr * (s_obj / s_sum) if s_sum > 0 else t_raw_arr

    rows = []
    cum_t2 = 0.0
    for i in range(len(df_pre)):
        seg = df_pre.iloc[i]
        ts = float(t_raw_arr[i])
        cum_t2 += ts
        pace_val = (ts / float(seg["seg_len"])) * 1000.0 if seg["seg_len"] > 0 else ts
        rows.append({
            "Km": (int(seg["idx"])+1) if seg["seg_len"] >= 999 else f"{int(seg['idx'])+1} ({seg['seg_len']:.0f}m)",
            "Pente (%)": round(float(seg["grade"]), 2),
            "Mult Pente": round(float(seg["grade_mult"]), 4),
            "D+ seg (m)": round(float(seg["seg_dp"]), 1),
            "D+ cum (m)": round(float(seg["cum_dp"]), 1),
            "Mult Fatigue": round(float(seg["fat_mult"]), 4),
            "Mult Altitude": round(float(seg["alt_mult"]), 4),
            "Temp GPS (°C)": round(float(seg["temp_raw"]), 1) if seg["temp_raw"] is not None else None,
            "Temp eff/WBGT (°C)": round(float(seg["temp_eff"]), 1) if seg["temp_eff"] is not None else None,
            "Mult Temp": round(float(seg["temp_mult"]), 4),
            "Vent (m/s)": round(float(seg["wind"]), 1) if seg["wind"] is not None else None,
            "Headwind (m/s)": round(float(seg.get("head_s", seg["head"])), 2),
            "Tailwind (m/s)": round(float(seg.get("tail_s", seg["tail"])), 2),
            "Mult Vent": round(float(seg["wind_mult_adj"]), 4),
            "Humidité (%)": round(float(seg["hum"]), 1) if seg["hum"] is not None else None,
            "Temps seg (s)": round(ts, 1),
            "Allure (min/km)": pace_str(pace_val),
            "Temps cumulé": seconds_to_hms(cum_t2),
        })

    df_out = pd.DataFrame(rows)
    if show_smooth_pace and not df_out.empty:
        w_p = int(max(1, smooth_window_km)); w_p += (1 if w_p%2==0 else 0)
        s_p = pd.Series(df_out["Temps seg (s)"].astype(float)).rolling(w_p, center=True, min_periods=1).median()
        df_out["Allure lissée (min/km)"] = s_p.apply(pace_str)

    total_s = float(np.sum(t_raw_arr))
    ci_low  = total_s * 0.95
    ci_high = total_s * 1.05

    return {
        "df": df_out, "total_s": total_s, "total_human": seconds_to_hms(total_s),
        "ci_low": seconds_to_hms(ci_low), "ci_high": seconds_to_hms(ci_high),
        "dist_gpx_km": dist_gpx_km, "K": K, "avg_alt": avg_alt, "d_plus_total": d_plus_total,
        "refs_fit": refs_fit, "pre_df": df_pre,
    }



# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# UI PRINCIPALE
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════

# ── SIDEBAR ──
st.sidebar.title("⚙️ Paramètres globaux")

# Mode interface v3
mode = st.sidebar.radio("Mode interface",
                         ["🟢 Simple (recommandé)", "🔵 Expert (tous les curseurs)"],
                         horizontal=False)
EXPERT = "Expert" in mode

st.sidebar.markdown("---")
st.sidebar.subheader("Coefficients recalibrage (Onglets 1 & 2)")
st.sidebar.caption("Utilisés pour le recalibrage des tests d'endurance et des séances d'entraînement.")

c2_k_up       = st.sidebar.number_input("Coeff montée (k_up)",   value=1.040, format="%.3f", step=0.001)
c2_k_down     = st.sidebar.number_input("Coeff descente (k_down)", value=0.996, format="%.3f", step=0.001)
c2_k_temp_hot = st.sidebar.number_input("Sensib. chaleur (k_temp_hot)",  value=0.0020, format="%.4f", step=0.0005)
c2_k_temp_cold= st.sidebar.number_input("Sensib. froid (k_temp_cold)",   value=0.0020, format="%.4f", step=0.0005)
c2_opt_temp   = st.sidebar.number_input("Température optimale (°C)", value=12.0, format="%.1f", step=0.5)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Si pas de GPS/temps, la météo auto ne peut pas être calculée.\n"
    "Dans ce cas, tu peux forcer une température manuelle par test/intervalle."
)

# ── ONGLETS PRINCIPAUX ──
main_tabs = st.tabs(["🏃 Prédiction de course (v3)", "🧪 Tests d'endurance + VC", "⚙️ Analyse entraînement"])


# ════════════════════════════════════════════════════════════════
# ONGLET 0 — PRÉDICTION v3
# ════════════════════════════════════════════════════════════════
with main_tabs[0]:
    st.title("🏃 Prédiction de course — Coach & Athlète")
    st.caption("v3 — WBGT · Minetti · DEM · Recalibration des références · Interface pédagogique")

    # ─── GPX ───
    st.markdown("---")
    st.header("1️⃣  Parcours GPX")
    gpx_file = st.file_uploader("📂 Importer le fichier GPX de la course cible", type=["gpx"])
    points = None
    dem_elevations = None

    if gpx_file:
        _gpx, points = parse_gpx_points(gpx_file)
        if points:
            tot_tmp = sum(haversine_m(points[i-1].latitude, points[i-1].longitude,
                                       points[i].latitude, points[i].longitude)
                          for i in range(1, len(points)))
            dup_tmp, ddn_tmp = compute_dplus_dminus([getattr(p, "elevation", 0.0) or 0.0 for p in points])
            avg_alt_tmp = np.mean([getattr(p, "elevation", 0.0) or 0.0 for p in points])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Distance GPX", f"{tot_tmp/1000:.2f} km")
            c2.metric("D+ GPS", f"{dup_tmp:.0f} m")
            c3.metric("D- GPS", f"{ddn_tmp:.0f} m")
            c4.metric("Alt. moy.", f"{avg_alt_tmp:.0f} m")

    with st.expander("🏔️ Correction altimétrique DEM (optionnel — recommandé en montagne)"):
        st.info(
            "Le GPS vertical a une précision de ±5-15 m, ce qui peut *inventer* 200-400 m de D+ sur un marathon. "
            "Le DEM (modèle numérique de terrain) donne l'altitude réelle à ±1 m."
        )
        use_dem = st.checkbox("Activer la correction DEM", value=False)
        dem_dataset = "srtm30m"
        if use_dem:
            dem_dataset = st.selectbox(
                "Dataset", ["srtm30m (global, 30m)", "eudem25m (Europe, 25m — plus précis)", "mapzen (global fusion)"],
                index=0).split()[0]
            if gpx_file and points and st.button("🔄 Télécharger et corriger l'altitude"):
                with st.spinner("Correction DEM en cours..."):
                    dem_elevations = list(correct_elevations_dem(points, max_points=100, dataset=dem_dataset))
                    st.session_state["dem_elevations"] = dem_elevations
                    dup_dem, ddn_dem = compute_dplus_dminus([e or 0.0 for e in dem_elevations])
                    st.success(f"DEM OK — D+ DEM: **{dup_dem:.0f} m** (vs GPS: {dup_tmp:.0f} m) | D- DEM: **{ddn_dem:.0f} m**")
        if "dem_elevations" in st.session_state:
            dem_elevations = st.session_state["dem_elevations"]

    # ─── RÉFÉRENCES ───
    st.markdown("---")
    st.header("2️⃣  Courses de référence")
    st.info(
        "Les références servent à **calibrer le modèle** sur l'athlète. "
        "Plus les distances sont variées (5 km, semi, marathon...), meilleure est la précision. "
        "Minimum conseillé : **3 références** sur les 12 derniers mois."
    )

    if "n_refs" not in st.session_state:
        st.session_state.n_refs = 3
    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("➕ Ajouter une référence") and st.session_state.n_refs < 6:
            st.session_state.n_refs += 1
    with cc2:
        if st.button("➖ Retirer") and st.session_state.n_refs > 1:
            st.session_state.n_refs -= 1

    refs_raw = []
    for i in range(1, st.session_state.n_refs + 1):
        with st.expander(f"📌 Référence {i}", expanded=(i <= 2)):
            use_file = st.checkbox(f"Importer depuis fichier FIT/TCX (Garmin, Polar...)", key=f"use_file_{i}")
            c1, c2, c3, c4 = st.columns(4)
            dist = c1.number_input("Distance (m)", value=float(st.session_state.get(f"dist_{i}", 5000*i)), key=f"dist_{i}")
            temps = c2.text_input("Temps (h:mm:ss)", value=str(st.session_state.get(f"temps_{i}", "0:40:00")), key=f"temps_{i}")
            dup   = c3.number_input("D+ (m)", value=float(st.session_state.get(f"dup_{i}", 0.0)), key=f"dup_{i}")
            ddn   = c4.number_input("D- (m)", value=float(st.session_state.get(f"ddn_{i}", 0.0)), key=f"ddn_{i}")
            file_in = st.file_uploader(f"Fichier FIT/TCX", type=["fit", "tcx"], key=f"fileref_{i}") if use_file else None

            dur_hms_file = avg_temp_ref = avg_wind_ref = avg_hum_ref = hr_ref = None
            fname = file_in.name.lower() if file_in else ""
            fit_data = tcx_data = None

            if file_in:
                if fname.endswith(".fit"):
                    fit_data = parse_fit_v3(file_in)
                    if fit_data:
                        dist, dup, ddn = fit_data["distance"], fit_data["D_up"], fit_data["D_down"]
                        dur_hms_file = fit_data["duration_hms"]
                        avg_temp_ref, avg_wind_ref, avg_hum_ref = fit_data["avg_temp"], fit_data["avg_wind"], fit_data["avg_humidity"]
                        hr_ref = fit_data.get("hr_analysis")
                elif fname.endswith(".tcx"):
                    tcx_data = parse_tcx_v3(file_in)
                    if tcx_data:
                        dist, dup, ddn = tcx_data["distance"], tcx_data["D_up"], tcx_data["D_down"]
                        dur_hms_file = tcx_data["duration_hms"]
                        avg_temp_ref, avg_wind_ref, avg_hum_ref = tcx_data["avg_temp"], tcx_data["avg_wind"], tcx_data["avg_humidity"]

                cs, ce = st.columns(2)
                sh = cs.text_input("Début segment (hh:mm:ss)", "00:00:00", key=f"start_{i}")
                eh = ce.text_input("Fin segment (hh:mm:ss)",   "23:59:59", key=f"end_{i}")
                start_td, end_td = hms_to_timedelta(sh), hms_to_timedelta(eh)
                if start_td.total_seconds() > 0 or end_td.total_seconds() < 86399:
                    pts_src = None
                    if fit_data and "points" in fit_data: pts_src = fit_data["points"]
                    elif tcx_data and "points" in tcx_data: pts_src = tcx_data["points"]
                    if pts_src:
                        seg = extract_segment(pts_src, start_td, end_td)
                        seg_dist = 0.0; seg_elevs = []; seg_times = []
                        for j in range(1, len(seg)):
                            p1, p2 = seg[j-1], seg[j]
                            la1, lo1 = (p1["lat"], p1["lon"]) if isinstance(p1, dict) else (p1.latitude, p1.longitude)
                            la2, lo2 = (p2["lat"], p2["lon"]) if isinstance(p2, dict) else (p2.latitude, p2.longitude)
                            e2 = p2.get("elev", 0) if isinstance(p2, dict) else p2.elevation
                            t2 = p2.get("time") if isinstance(p2, dict) else p2.time
                            seg_dist += haversine_m(la1, lo1, la2, lo2)
                            seg_elevs.append(e2)
                            if t2: seg_times.append(t2)
                        dup, ddn = compute_dplus_dminus(seg_elevs)
                        if len(seg_times) >= 2:
                            dur_hms_file = seconds_to_hms((seg_times[-1]-seg_times[0]).total_seconds())
                        dist = round(seg_dist)
            else:
                if EXPERT:
                    cs2, ce2 = st.columns(2)
                    avg_temp_ref = cs2.number_input(f"Temp moy. course (°C)", value=15.0, key=f"avgT_{i}")
                    avg_hum_ref  = ce2.number_input(f"Humidité moy. (%)", value=60.0, key=f"avgH_{i}")
                else:
                    avg_temp_ref = None; avg_hum_ref = None

            temps_eff = dur_hms_file if dur_hms_file else temps
            secs_brut = hms_to_seconds(temps_eff)
            dist_km = safe_float(dist, 1.0) / 1000.0
            if secs_brut > 0 and dist_km > 0:
                pace_val = pace_str(secs_brut / dist_km)
                st.caption(f"📍 {dist:.0f} m · {temps_eff} · **{pace_val}/km**"
                           + (f" · D+ {dup:.0f}m" if dup > 0 else "")
                           + (f" · Temp GPS: {avg_temp_ref:.0f}°C" if avg_temp_ref else "")
                           + (f" · FC fiabilité: {hr_ref.get('reliability')}" if hr_ref else ""))
            if hr_ref and hr_ref.get("hr_max"):
                st.caption(f"💓 FC max {hr_ref['hr_max']} bpm · dérive {hr_ref['hr_drift']} bpm · seuil estimé ~{hr_ref['hr_threshold_est']} bpm")

            refs_raw.append({
                "distance": float(dist), "temps": str(temps_eff),
                "D_up": float(dup), "D_down": float(ddn),
                "duration_hms_file": dur_hms_file,
                "avg_temp": avg_temp_ref, "avg_humidity": avg_hum_ref, "avg_wind": avg_wind_ref,
                "hr_analysis": hr_ref,
            })

    # ─── RECALIBRATION ───
    st.markdown("---")
    st.header("3️⃣  Recalibration des références vers les conditions idéales")

    st.markdown('''
<div class="highlight-box">
<strong>Pourquoi recalibrer ?</strong><br>
Une course réalisée par 30°C et 80% d'humidité vaut <em>physiologiquement mieux</em>
qu'un temps identique par 12°C et temps sec. Sans correction, le modèle croit que
l'athlète est plus lent qu'il ne l'est vraiment.<br><br>
La recalibration <em>restitue</em> chaque référence à ce qu'aurait été le résultat
dans des conditions parfaites (plat, température optimale, humidité neutre),
avant de construire le modèle de performance.
</div>
''', unsafe_allow_html=True)

    use_recalibrated = st.checkbox(
        "✅ Recalibrer les références vers les conditions idéales (fortement recommandé)", value=True)

    if use_recalibrated:
        st.success("Les références seront normalisées avant le fit.")
    else:
        st.warning("Les références brutes sont utilisées.")

    opt_temp = 12.0
    use_wbgt = True
    elev_ref_power = 0.60
    temp_ref_power = 0.85

    with st.expander("⚙️ Paramètres de recalibration"):
        opt_temp = st.slider("Température optimale de course (°C)", 5.0, 20.0, 12.0, 0.5)
        param_help("L'athlète est avantagé par des températures plus basses",
                   "L'athlète est optimal à des températures plus élevées",
                   "12°C est une bonne valeur par défaut")
        use_wbgt = st.checkbox("Utiliser le WBGT (chaleur+humidité) — recommandé", value=True)
        col_ep1, col_ep2 = st.columns(2)
        with col_ep1:
            elev_ref_power = st.slider("Force correction pente des références", 0.0, 1.0, 0.60, 0.05)
        with col_ep2:
            temp_ref_power = st.slider("Force correction température des références", 0.0, 1.0, 0.85, 0.05)

    # Valeurs par défaut pour la recalibration
    _k_up_prev   = st.session_state.get("k_up_val", 12.0)
    _k_down_prev = st.session_state.get("k_down_val", 5.0)
    _g0u_prev    = st.session_state.get("g0_up_val", 3.0)
    _g0d_prev    = st.session_state.get("g0_down_val", 2.5)
    cold_quad_prev = 0.0012; hot_quad_prev = 0.0016; temp_max_pen_prev = 0.10

    calib_rows = []
    for r in refs_raw:
        t_brut = hms_to_seconds(r.get("duration_hms_file") or r.get("temps", ""))
        dist_km_r = safe_float(r.get("distance", 1.0)) / 1000.0
        avg_t = r.get("avg_temp"); avg_h = safe_float(r.get("avg_humidity", 50.0), 50.0)
        wbgt_val = wbgt_simplified(avg_t, avg_h) if avg_t is not None and use_wbgt else None
        t_ideal = recalibrate_ref_to_ideal(
            ref={**r, "temps": r.get("duration_hms_file") or r.get("temps", "0:00:00")},
            opt_temp=opt_temp, use_wbgt=use_wbgt,
            cold_quad=cold_quad_prev, hot_quad=hot_quad_prev, temp_max_penalty=temp_max_pen_prev,
            k_up=_k_up_prev, k_down=_k_down_prev, down_cap=-0.08,
            g0_up=_g0u_prev, g0_down=_g0d_prev, max_up=0.30, max_down=-0.06,
            elev_ref_power=elev_ref_power, temp_ref_power=temp_ref_power,
        ) if use_recalibrated else float(t_brut)
        gain_s = t_brut - t_ideal
        calib_rows.append({
            "Distance": f"{safe_float(r['distance'])/1000:.1f} km",
            "Temps brut": seconds_to_hms(t_brut),
            "Allure brute": pace_str(t_brut / dist_km_r) if dist_km_r > 0 else "-",
            "D+": f"{r['D_up']:.0f} m",
            "Temp GPS": f"{avg_t:.0f}°C" if avg_t is not None else "?",
            "WBGT": f"{wbgt_val:.1f}°C" if wbgt_val is not None else "-",
            "Temps recalibré": seconds_to_hms(t_ideal) if use_recalibrated else "—",
            "Allure recalibrée": pace_str(t_ideal / dist_km_r) if (use_recalibrated and dist_km_r > 0) else "—",
            "Gain correction": f"-{seconds_to_hms(gain_s)}" if gain_s > 0 else (f"+{seconds_to_hms(-gain_s)}" if gain_s < 0 else "0"),
        })
    st.subheader("📋 Résumé de la recalibration")
    st.dataframe(pd.DataFrame(calib_rows), use_container_width=True)

    # ─── PARAMÈTRES DU MODÈLE ───
    st.markdown("---")
    st.header("4️⃣  Paramètres du modèle")

    cold_quad = 0.0012; hot_quad = 0.0016; temp_max_penalty = 0.10; temp_power = 1.0
    with st.expander("🌡️ Température & Humidité"):
        if EXPERT:
            c1, c2 = st.columns(2)
            with c1:
                cold_quad = st.number_input("Sensibilité au froid (coeff quad.)", value=0.0012, step=0.0002, format="%.4f")
            with c2:
                hot_quad = st.number_input("Sensibilité à la chaleur (coeff quad.)", value=0.0016, step=0.0002, format="%.4f")
            temp_max_penalty = st.slider("Pénalité maximale température (%)", 0.00, 0.20, 0.10, 0.01)
            temp_power = st.slider("Damping température (puissance)", 0.2, 1.2, 1.0, 0.05)
        if use_wbgt:
            st.markdown("**Aperçu : impact WBGT sur l'allure**")
            ex_t = st.slider("Température exemple (°C)", -5, 40, 20, 1, key="demo_temp")
            ex_h = st.slider("Humidité exemple (%)", 10, 100, 60, 5, key="demo_hum")
            ex_wbgt = wbgt_simplified(ex_t, ex_h)
            ex_mult = temp_multiplier_v3(ex_wbgt, opt_temp, cold_quad, hot_quad, temp_max_penalty)
            pen_pct = (ex_mult - 1.0) * 100.0
            cd1, cd2, cd3 = st.columns(3)
            cd1.metric("WBGT", f"{ex_wbgt:.1f}°C")
            cd2.metric("Multiplicateur", f"{ex_mult:.3f}")
            cd3.metric("Pénalité", f"+{pen_pct:.1f}%" if pen_pct > 0 else f"{pen_pct:.1f}%")

    apply_altitude = True; altitude_ref_m = 0.0
    with st.expander("🏔️ Altitude physiologique (hypoxie)"):
        apply_altitude = st.checkbox("Appliquer la pénalité d'altitude (VO2 réduite au-dessus de 1500 m)", value=True)
        if apply_altitude:
            altitude_ref_m = st.number_input("Altitude habituelle d'entraînement (m)", value=0.0, step=100.0)

    apply_grade = True; use_minetti = True; minetti_weight = 0.6
    elev_smooth_window = 11; grade_power = 0.85
    k_up = 12.0; k_down = 5.0; down_cap = -0.08
    g0_up = 3.0; g0_down = 2.5; max_up = 0.30; max_down = -0.06

    with st.expander("🎢 Modèle de pente"):
        apply_grade = st.checkbox("Prendre en compte la pente", value=True)
        use_minetti = st.checkbox("Modèle Minetti (base physiologique — Minetti et al. 2002)", value=True)
        if use_minetti:
            minetti_weight = st.slider("Part de Minetti dans le calcul (vs heuristique)", 0.0, 1.0, 0.6, 0.1)
        if EXPERT:
            elev_smooth_window = st.slider("Lissage altitude (fenêtre en points GPS)", 1, 51, 11, 2)
            grade_power = st.slider("Amortissement de l'effet pente (puissance)", 0.2, 1.0, 0.85, 0.05)
            c1, c2, c3 = st.columns(3)
            with c1:
                k_up = st.number_input("Sensibilité montée (k_up)", value=12.0, step=0.5)
            with c2:
                k_down = st.number_input("Sensibilité descente (k_down)", value=5.0, step=0.5)
            with c3:
                down_cap = st.number_input("Cap bonus descente", value=-0.08, step=0.01, format="%.2f")
            st.session_state["k_up_val"] = k_up; st.session_state["k_down_val"] = k_down
            st.session_state["g0_up_val"] = g0_up; st.session_state["g0_down_val"] = g0_down

    apply_wind = True; wind_mode = "Lissé"; wind_smooth_km = 5
    drag_coeff = 0.012; tail_credit = 0.35; wind_cap_head = 0.10; wind_cap_tail = -0.04
    wind_power = 1.0; wind_gate_g1 = 2.0; wind_gate_g2 = 8.0; wind_gate_min = 0.25

    with st.expander("💨 Vent"):
        apply_wind = st.checkbox("Appliquer l'effet du vent", value=True)
        if apply_wind and EXPERT:
            wind_smooth_km = st.slider("Lissage vent sur N km", 1, 11, 5, 2)
            c1, c2 = st.columns(2)
            drag_coeff  = c1.number_input("Coefficient aérodynamique", value=0.012, step=0.002, format="%.3f")
            tail_credit = c2.slider("Crédit vent arrière", 0.0, 0.8, 0.35, 0.05)

    base_cap = 0.08; extra_per_pct = 0.004; max_cap = 0.18
    if EXPERT:
        with st.expander("🧱 Plafond anti-accumulation"):
            c1, c2, c3 = st.columns(3)
            base_cap      = c1.slider("Plafond de base (%)", 0.02, 0.20, 0.08, 0.01)
            extra_per_pct = c2.slider("Extra par % de pente", 0.000, 0.020, 0.004, 0.001)
            max_cap       = c3.slider("Plafond absolu (%)", 0.05, 0.40, 0.18, 0.01)

    apply_fatigue = False; fatigue_rate = 0.0; fatigue_mode = "mixte"
    with st.expander("🔋 Fatigue en course"):
        apply_fatigue = st.checkbox("Activer la fatigue", value=False)
        if apply_fatigue:
            fatigue_rate = st.slider("Ralentissement total en fin de course (%)", 0.0, 30.0, 8.0, 0.5)
            fatigue_mode = st.selectbox("Type de fatigue",
                ["mixte (recommandé)", "distance (plat)", "d_plus (montagne)"]).split()[0]

    apply_ultra = False; ultra_amp = 0.0
    with st.expander("⚡ Stratégie de pacing Ultra"):
        apply_ultra = st.checkbox("Activer le pacing ultra (positive split)", value=False)
        if apply_ultra:
            ultra_amp = st.slider("Amplitude (%)", 0.0, 40.0, 10.0, 0.5)

    show_smooth_pace = True; smooth_window_km = 3
    with st.expander("📉 Options d'affichage"):
        show_smooth_pace = st.checkbox("Afficher l'allure lissée", value=True)
        smooth_window_km = st.slider("Fenêtre de lissage (km)", 1, 9, 3, 2) if show_smooth_pace else 3

    # ─── COURSE CIBLE ───
    st.markdown("---")
    st.header("5️⃣  Paramètres de la course cible")
    c1, c2 = st.columns(2)
    date_course  = c1.date_input("📅 Date de course", value=date.today())
    heure_course = c2.time_input("⏰ Heure de départ", value=time(9, 0))
    colf1, colf2 = st.columns(2)
    with colf1:
        force_dist = st.checkbox("Forcer la distance (si GPX != distance officielle)", value=False)
        dist_forcee = st.number_input("Distance (km)", value=42.195, format="%.3f") if force_dist else None
    with colf2:
        force_temps = st.checkbox("Travailler à partir d'un objectif de temps", value=False)
        temps_objectif = st.text_input("Temps objectif (h:mm:ss)", value="3:30:00") if force_temps else None
    st.markdown("---")

    # ─── CROSS-VALIDATION ───
    with st.expander("🔬 Cross-validation (fiabilité du modèle)"):
        st.info("La cross-validation Leave-One-Out teste la précision du modèle. MAPE < 3% = excellent | < 7% = correct")
        if st.button("Lancer la cross-validation"):
            refs_cv = prepare_refs(refs_raw, use_recalibrated, opt_temp, use_wbgt,
                                   cold_quad, hot_quad, temp_max_penalty,
                                   k_up, k_down, down_cap, g0_up, g0_down, max_up, max_down,
                                   elev_ref_power, temp_ref_power)
            cv = crossval_loo(refs_cv)
            if cv is None:
                st.warning("Au moins 3 références nécessaires pour la cross-validation.")
            else:
                df_cv, mae, mape = cv
                st.dataframe(df_cv, use_container_width=True)
                c1, c2 = st.columns(2)
                c1.metric("Erreur absolue moyenne", f"{seconds_to_hms(mae)} ({mae:.0f}s)")
                c2.metric("Erreur relative moyenne (MAPE)", f"{mape:.2f} %")
                if mape < 3: st.success("✅ Modèle bien calibré.")
                elif mape < 7: st.warning("⚠️ Calibration acceptable.")
                else: st.error("❌ Calibration faible — vérifier les références.")

    # ─── CALCUL & RÉSULTATS ───
    st.header("6️⃣  Calcul & Résultats")

    if st.button("▶️ Calculer la prédiction", type="primary"):
        if not gpx_file or points is None:
            st.error("⚠️ Importe un fichier GPX d'abord (section 1).")
        elif not any(safe_float(r.get("distance", 0)) > 0 and hms_to_seconds(r.get("temps", "0")) > 0 for r in refs_raw):
            st.error("⚠️ Renseigne au moins une référence valide (distance + temps).")
        else:
            with st.spinner("Calcul en cours (récupération météo + prédiction)..."):
                try:
                    res = run_prediction(
                        distance_cible_km=dist_forcee if force_dist else None,
                        refs_input=refs_raw, points=points,
                        date_course=date_course, heure_course=heure_course,
                        use_recalibrated=use_recalibrated, opt_temp=opt_temp, use_wbgt=use_wbgt,
                        cold_quad=cold_quad, hot_quad=hot_quad,
                        temp_max_penalty=temp_max_penalty, temp_power=temp_power,
                        elev_ref_power=elev_ref_power, temp_ref_power=temp_ref_power,
                        apply_grade=apply_grade, use_minetti=use_minetti, minetti_weight=minetti_weight,
                        k_up=k_up, k_down=k_down, down_cap=down_cap,
                        g0_up=g0_up, g0_down=g0_down, max_up=max_up, max_down=max_down,
                        elev_smooth_window=elev_smooth_window, grade_power=grade_power,
                        apply_altitude=apply_altitude, altitude_ref_m=altitude_ref_m,
                        apply_wind=apply_wind, wind_mode="Lissé", wind_smooth_km=wind_smooth_km,
                        drag_coeff=drag_coeff, tail_credit=tail_credit,
                        wind_cap_head=wind_cap_head, wind_cap_tail=wind_cap_tail, wind_power=wind_power,
                        wind_gate_g1=wind_gate_g1, wind_gate_g2=wind_gate_g2, wind_gate_min=wind_gate_min,
                        base_cap=base_cap, extra_per_pct=extra_per_pct, max_cap=max_cap,
                        apply_fatigue=apply_fatigue, fatigue_rate=fatigue_rate, fatigue_mode=fatigue_mode,
                        apply_ultra=apply_ultra, ultra_amp=ultra_amp,
                        objective_hms=temps_objectif if force_temps else None,
                        show_smooth_pace=show_smooth_pace, smooth_window_km=smooth_window_km,
                        dem_elevations=dem_elevations,
                    )
                    st.session_state["res"] = res
                except Exception as e:
                    import traceback
                    st.error(f"Erreur : {e}")
                    st.code(traceback.format_exc())

    if "res" in st.session_state:
        res = st.session_state["res"]
        st.markdown("---")
        st.subheader("🎯 Prédiction")
        avg_pace_s = res["total_s"] / max(res["dist_gpx_km"], 1e-6)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("⏱ Temps prédit", res["total_human"])
        c2.metric("📊 Allure moy.", pace_str(avg_pace_s) + "/km")
        c3.metric("Fourchette basse (−5%)", res["ci_low"])
        c4.metric("Fourchette haute (+5%)", res["ci_high"])
        c5.metric("K Riegel", f"{res['K']:.3f}")
        st.caption(f"Distance GPX : {res['dist_gpx_km']:.3f} km | D+ total : {res['d_plus_total']:.0f} m | Alt. moy. : {res['avg_alt']:.0f} m")

        df_out = res["df"]
        if not df_out.empty:
            t1, t2, t3 = st.tabs(["📈 Allure par km", "🔎 Facteurs", "📋 Tableau détaillé"])
            with t1:
                fig, ax = plt.subplots(figsize=(12, 4))
                pv = []
                for v in df_out["Allure (min/km)"].values:
                    try:
                        parts = str(v).split(":")
                        pv.append(int(parts[0]) + int(parts[1])/60.0)
                    except Exception:
                        pv.append(float("nan"))
                x = list(range(1, len(pv)+1))
                ax.plot(x, pv, lw=1.5, alpha=0.35, color="steelblue", label="Allure brute")
                if "Allure lissée (min/km)" in df_out.columns:
                    ps = []
                    for v in df_out["Allure lissée (min/km)"].values:
                        try:
                            parts = str(v).split(":")
                            ps.append(int(parts[0]) + int(parts[1])/60.0)
                        except Exception:
                            ps.append(float("nan"))
                    ax.plot(x, ps, lw=2.5, color="firebrick", label="Allure lissée")
                ax.invert_yaxis()
                ax.set_xlabel("Kilomètre"); ax.set_ylabel("Allure (min/km)")
                ax.set_title("Allure prévisionnelle km par km")
                ax.legend(); ax.grid(alpha=0.3)
                st.pyplot(fig); plt.close(fig)
            with t2:
                fig2, ax2 = plt.subplots(figsize=(12, 4))
                x = list(range(1, len(df_out)+1))
                ax2.plot(x, df_out["Mult Pente"].values, label="Pente (Minetti+heu)", lw=2)
                if "Mult Temp" in df_out.columns:
                    ax2.plot(x, df_out["Mult Temp"].values, label="Température/WBGT", lw=2)
                if "Mult Vent" in df_out.columns:
                    ax2.plot(x, df_out["Mult Vent"].values, label="Vent", lw=2)
                if "Mult Fatigue" in df_out.columns:
                    ax2.plot(x, df_out["Mult Fatigue"].values, label="Fatigue", lw=2, ls=":")
                ax2.axhline(1.0, color="gray", lw=0.8)
                ax2.set_xlabel("Kilomètre"); ax2.set_ylabel("Multiplicateur (1.0 = neutre)")
                ax2.set_title("Décomposition des facteurs de ralentissement")
                ax2.legend(); ax2.grid(alpha=0.3)
                st.pyplot(fig2); plt.close(fig2)
            with t3:
                st.dataframe(df_out, use_container_width=True)

    # Carte & Profil
    if gpx_file and points:
        with st.expander("🗺️ Carte & Profil d'altitude", expanded=False):
            try:
                lats_m = [p.latitude for p in points]
                lons_m = [p.longitude for p in points]
                view = pdk.ViewState(latitude=float(np.mean(lats_m)), longitude=float(np.mean(lons_m)), zoom=13, pitch=0)
                deck = pdk.Deck(
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    initial_view_state=view,
                    layers=[pdk.Layer("PathLayer",
                                      data=[{"path": [[lon, lat] for lat, lon in zip(lats_m, lons_m)]}],
                                      get_path="path", get_color=[220, 50, 50], width_min_pixels=4)],
                )
                st.pydeck_chart(deck, use_container_width=True)
                cum_d = [0.0]
                for i in range(1, len(points)):
                    cum_d.append(cum_d[-1] + haversine_m(
                        points[i-1].latitude, points[i-1].longitude,
                        points[i].latitude, points[i].longitude))
                x_km = np.array(cum_d) / 1000.0
                y_gps = np.array([getattr(p, "elevation", 0.0) or 0.0 for p in points])
                w = int(elev_smooth_window); w += (1 if w%2==0 else 0)
                fig3, ax3 = plt.subplots(figsize=(10, 3))
                if w >= 3 and y_gps.size >= w:
                    y_s = np.convolve(y_gps, np.ones(w)/w, mode="same")
                    ax3.plot(x_km, y_s, lw=2, label="GPS lissé", color="steelblue")
                    ax3.plot(x_km, y_gps, lw=1, alpha=0.2, color="gray", label="GPS brut")
                else:
                    ax3.plot(x_km, y_gps, lw=2, label="GPS", color="steelblue")
                if dem_elevations is not None and len(dem_elevations) == len(points):
                    y_dem = np.array([e if e is not None else 0.0 for e in dem_elevations])
                    ax3.plot(x_km, y_dem, lw=2, ls="--", label="DEM corrigé", color="forestgreen")
                ax3.set_xlabel("Distance (km)"); ax3.set_ylabel("Altitude (m)")
                ax3.set_title("Profil d'altitude"); ax3.legend(); ax3.grid(alpha=0.3)
                st.pyplot(fig3); plt.close(fig3)
            except Exception as e:
                st.error(f"Impossible d'afficher la carte : {e}")



# ════════════════════════════════════════════════════════════════
# ONGLET 1 — TESTS D'ENDURANCE + VC (Code 2)
# ════════════════════════════════════════════════════════════════
with main_tabs[1]:
    st.header("🧪 Tests d'endurance (2 à 6 tests)")

    if "nb_tests" not in st.session_state:
        st.session_state.nb_tests = 2
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("➕ Ajouter un test", use_container_width=True, key="add_test_btn"):
            if st.session_state.nb_tests < 6:
                st.session_state.nb_tests += 1
    with colB:
        if st.button("➖ Supprimer un test", use_container_width=True, key="del_test_btn"):
            if st.session_state.nb_tests > 2:
                st.session_state.nb_tests -= 1
    st.markdown(f"### Nombre de tests sélectionnés : **{st.session_state.nb_tests}**")

    tests_data = []
    n = st.session_state.nb_tests
    indices = list(range(1, n + 1))
    cols = st.columns(2)

    for idx, i in enumerate(indices):
        col = cols[idx % 2]
        with col:
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.subheader(f"📌 Test {i}")
            uploaded = st.file_uploader(f"Fichier Test {i} (FIT, GPX, CSV, TCX)",
                                         type=ACCEPTED_TYPES, key=f"c2_file_{i}")
            test_date  = st.date_input(f"📅 Date du test {i}", value=date.today(), key=f"c2_date_{i}")
            show_fc    = st.checkbox(f"☑️ FC (Test {i})",      value=True,  key=f"c2_fc_{i}")
            show_pace  = st.checkbox(f"☑️ Allure (Test {i})",  value=False, key=f"c2_pace_{i}")
            show_power = st.checkbox(f"☑️ Puissance (Test {i})", value=False, key=f"c2_power_{i}")
            manual_temp = st.number_input(
                f"🌡️ Température manuelle Test {i} (°C) (si météo auto indispo)",
                value=float(0.0), step=0.5, key=f"c2_manual_temp_{i}")

            if uploaded:
                try:
                    df = load_activity(uploaded)
                except Exception as e:
                    st.error(f"Erreur dans le fichier du Test {i} : {e}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    if idx % 2 == 1 and idx < len(indices) - 1:
                        cols = st.columns(2)
                    continue

                lag = st.slider(f"Correction du décalage capteur (s) — Test {i}", 0, 10, 0, key=f"c2_lag_{i}")
                df["timestamp"] = df["timestamp"] - pd.to_timedelta(lag, unit="s")
                df, window, total_dur, pauses = smooth_hr(df)
                st.caption(f"Durée : {total_dur:.1f}s · Lissage : {window}s · Pauses : {pauses}")

                dur_int = max(2, int(total_dur))
                default_end = min(720, dur_int - 1)
                start_sec = st.slider(
                    f"▶️ Début du test {i} (s)", 0, dur_int - 1, 0,
                    key=f"c2_start_{i}",
                    help="Glisse pour choisir le début du segment analysé"
                )
                end_sec = st.slider(
                    f"⏹️ Fin du test {i} (s)", 1, dur_int,
                    min(default_end, dur_int),
                    key=f"c2_end_{i}",
                    help="Glisse pour choisir la fin du segment analysé"
                )
                st.caption(f"Segment sélectionné : **{seconds_to_hms(start_sec)}** → **{seconds_to_hms(end_sec)}**  ({end_sec - start_sec}s)")

                if end_sec <= start_sec:
                    st.error("La fin doit être > au début.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    if idx % 2 == 1 and idx < len(indices) - 1:
                        cols = st.columns(2)
                    continue

                segment = df[(df["time_s"] >= start_sec) & (df["time_s"] <= end_sec)]
                if len(segment) < 10:
                    st.warning("Segment trop court pour analyse.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    if idx % 2 == 1 and idx < len(indices) - 1:
                        cols = st.columns(2)
                    continue

                stats, drift_bpm, drift_pct = analyze_heart_rate(segment)
                dist_m   = segment_distance_m(segment)
                t_s_raw  = float(end_sec - start_sec)
                v_kmh_raw = (3.6 * dist_m / t_s_raw) if (t_s_raw > 0 and dist_m > 0) else 0.0
                D_up, D_down = segment_elevation_up_down(segment)
                grade_pct = segment_grade_percent_net(segment)
                avgT, avgW, avgH = get_segment_weather(segment)
                temp_real = avgT if avgT is not None else (manual_temp if manual_temp != 0.0 else None)

                t_s_ideal = recalibrate_time_to_ideal_c2(
                    time_seconds_raw=t_s_raw, D_up=D_up, D_down=D_down,
                    distance_m=(dist_m if dist_m > 0 else 1000.0),
                    temp_real=temp_real,
                    k_up=c2_k_up, k_down=c2_k_down,
                    k_temp_hot=c2_k_temp_hot, k_temp_cold=c2_k_temp_cold,
                    opt_temp=c2_opt_temp,
                )
                v_kmh_ideal = (
                    (3.6 * dist_m / t_s_ideal)
                    if (t_s_ideal is not None and t_s_ideal > 0 and dist_m > 0)
                    else None
                )
                pace_raw_str   = _pace_str_from_kmh(v_kmh_raw)
                pace_ideal_str = _pace_str_from_kmh(v_kmh_ideal) if v_kmh_ideal else "–"
                d_v_kmh, d_v_pct = analyze_speed_kinetics(segment)

                df_table = pd.DataFrame({
                    "Métrique": [
                        "FC moyenne (bpm)", "FC max (bpm)", "FC min (bpm)",
                        "Dérive FC (bpm/min)", "Dérive FC (%/min)",
                        "Dérive vitesse (km/h/min)", "Dérive vitesse (%/min)",
                        "Durée segment réelle (s)", "Durée segment recalibrée (s)",
                        "Distance (m)", "Vitesse réelle (km/h)", "Allure réelle (min/km)",
                        "Vitesse recalibrée (km/h)", "Allure recalibrée (conditions idéales) (min/km)",
                        "Pente nette moyenne (%)", "D+ (m)", "D- (m)",
                        "Température (°C)", "Vent (m/s)", "Humidité (%)",
                    ],
                    "Valeur": [
                        stats.get("FC moyenne (bpm)"), stats.get("FC max (bpm)"), stats.get("FC min (bpm)"),
                        drift_bpm, drift_pct, d_v_kmh, d_v_pct,
                        round(t_s_raw, 1),
                        (round(t_s_ideal, 1) if t_s_ideal is not None else None),
                        round(dist_m, 1),
                        round(v_kmh_raw, 2), pace_raw_str,
                        (round(v_kmh_ideal, 2) if v_kmh_ideal is not None else None),
                        pace_ideal_str,
                        (round(grade_pct, 3) if grade_pct is not None else None),
                        round(D_up, 1), round(D_down, 1),
                        (round(temp_real, 2) if temp_real is not None else None),
                        (round(avgW, 2) if avgW is not None else None),
                        (round(avgH, 2) if avgH is not None else None),
                    ],
                })
                st.dataframe(style_metrics_table(df_table), hide_index=True, use_container_width=True)

                fig, ax = plt.subplots(figsize=(9, 4.6))
                plot_multi_signals(ax, segment, t0=start_sec, who=f"T{i}",
                                   show_fc=show_fc,
                                   show_pace=show_pace and (get_speed_col(segment) is not None),
                                   show_power=show_power and ("power_smooth" in segment.columns),
                                   linewidth=1.9)
                ax.set_title(f"Cinétique — Test {i} ({test_date})")
                ax.set_xlabel("Temps segment (s)"); ax.grid(True, alpha=0.2)
                handles, labels = [], []
                for a in fig.axes:
                    h, l = a.get_legend_handles_labels()
                    handles += h; labels += l
                if handles:
                    ax.legend(handles, labels, fontsize=8, loc="upper left", frameon=False)
                st.pyplot(fig)

                tests_data.append({
                    "i": i, "segment": segment,
                    "start_sec": start_sec, "end_sec": end_sec,
                    "stats": stats, "drift_bpm": drift_bpm, "drift_pct": drift_pct,
                    "d_v_kmh": d_v_kmh, "d_v_pct": d_v_pct,
                    "dist_m": dist_m, "t_s_raw": t_s_raw, "t_s_ideal": t_s_ideal,
                    "v_kmh_raw": v_kmh_raw, "v_kmh_ideal": v_kmh_ideal,
                    "pace_raw_str": pace_raw_str, "pace_ideal_str": pace_ideal_str,
                    "grade_pct": grade_pct, "D_up": D_up, "D_down": D_down,
                    "temp_real": temp_real, "wind": avgW, "hum": avgH, "date": test_date,
                })
            st.markdown("</div>", unsafe_allow_html=True)
        if idx % 2 == 1 and idx < len(indices) - 1:
            cols = st.columns(2)

    # ── GRAPHIQUE COMBINÉ ──
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("📊 Graphique combiné — FC / Allure / Puissance")
    show_c_fc    = st.checkbox("☑️ FC",       True,  key="comb_fc_t1")
    show_c_pace  = st.checkbox("☑️ Allure",   False, key="comb_pace_t1")
    show_c_power = st.checkbox("☑️ Puissance",False, key="comb_power_t1")
    if len(tests_data) > 0:
        figC, axC = plt.subplots(figsize=(10, 5))
        for t in tests_data:
            seg = t["segment"]
            t0 = seg["time_s"].iloc[0]
            plot_multi_signals(axC, seg, t0=t0, who=f"T{t['i']}",
                               show_fc=show_c_fc,
                               show_pace=show_c_pace and (get_speed_col(seg) is not None),
                               show_power=show_c_power and ("power_smooth" in seg.columns))
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
    st.markdown("</div>", unsafe_allow_html=True)

    # ── VITESSE CRITIQUE ──
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Modèle Hyperbolique — Vitesse Critique (VC)")
    valid_tests_raw   = [t for t in tests_data if t.get("dist_m", 0) > 0 and t.get("t_s_raw") and t["t_s_raw"] > 0]
    valid_tests_ideal = [t for t in tests_data if t.get("dist_m", 0) > 0 and t.get("t_s_ideal") and t["t_s_ideal"] > 0]

    VC_kmh_raw = VC_kmh_ideal = D_prime_raw = D_prime_ideal = None
    if len(valid_tests_raw) >= 2:
        D_arr = np.array([t["dist_m"] for t in valid_tests_raw], dtype=float)
        T_arr = np.array([t["t_s_raw"] for t in valid_tests_raw], dtype=float)
        slope_r, intercept_r = np.polyfit(T_arr, D_arr, 1)
        VC_kmh_raw = float(slope_r) * 3.6; D_prime_raw = float(intercept_r)
    if len(valid_tests_ideal) >= 2:
        D2 = np.array([t["dist_m"] for t in valid_tests_ideal], dtype=float)
        T2 = np.array([t["t_s_ideal"] for t in valid_tests_ideal], dtype=float)
        slope2, intercept2 = np.polyfit(T2, D2, 1)
        VC_kmh_ideal = float(slope2) * 3.6; D_prime_ideal = float(intercept2)

    if VC_kmh_raw is None and VC_kmh_ideal is None:
        st.info("Il faut au moins deux tests valides pour calculer la VC.")
    else:
        colvc1, colvc2 = st.columns(2)
        with colvc1:
            st.markdown("### 📌 VC réelle (brute)")
            if VC_kmh_raw is not None and VC_kmh_raw > 0:
                st.success(f"**VC réelle = {VC_kmh_raw:.2f} km/h**  \n"
                           f"➡️ **{_pace_str_from_kmh(VC_kmh_raw)}**  \n"
                           f"**D′ = {D_prime_raw:.1f} m**")
            else:
                st.info("VC brute non calculable.")
        with colvc2:
            st.markdown("### 🔧 VC recalibrée (conditions idéales)")
            if VC_kmh_ideal is not None and VC_kmh_ideal > 0:
                st.success(f"**VC recalibrée = {VC_kmh_ideal:.2f} km/h**  \n"
                           f"➡️ **{_pace_str_from_kmh(VC_kmh_ideal)}**  \n"
                           f"**D′ = {D_prime_ideal:.1f} m**")
            else:
                st.info("VC recalibrée non calculable.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── TABLEAU TEMPS DE MAINTIEN ──
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("📊 Temps de maintien par %VC — Hybride (Power Law <100% / D′ >100%)")

    A_raw = k_raw = A_ideal = k_ideal = None
    if len(valid_tests_raw) >= 2:
        A_raw, k_raw = fit_power_law_from_tests(valid_tests_raw, use_ideal=False)
    if len(valid_tests_ideal) >= 2:
        A_ideal, k_ideal = fit_power_law_from_tests(valid_tests_ideal, use_ideal=True)

    colT1, colT2 = st.columns(2)
    with colT1:
        st.markdown("### 🟦 VC brute")
        if VC_kmh_raw is None or VC_kmh_raw <= 0:
            st.info("VC brute indisponible.")
        else:
            df_hold_raw = build_hybrid_holding_table(
                VC_kmh=VC_kmh_raw, D_prime_m=D_prime_raw,
                A_pl=A_raw, k_pl=k_raw, pct_low=80, pct_high=130, step=2)
            if df_hold_raw is None or df_hold_raw.empty:
                st.info("Aucune ligne exploitable.")
            else:
                if "Vitesse (km/h)" in df_hold_raw.columns:
                    df_hold_raw["Allure"] = df_hold_raw["Vitesse (km/h)"].apply(_pace_str_from_kmh)
                st.dataframe(df_hold_raw, hide_index=True, use_container_width=True)
            if A_raw is not None and k_raw is not None:
                st.caption(f"Power Law brut : A={A_raw:.2f}, k={k_raw:.3f}")
            if D_prime_raw is not None:
                st.caption(f"D′ brut : {D_prime_raw:.1f} m")
    with colT2:
        st.markdown("### 🟧 VC recalibrée (conditions idéales)")
        if VC_kmh_ideal is None or VC_kmh_ideal <= 0:
            st.info("VC recalibrée indisponible.")
        else:
            df_hold_ideal = build_hybrid_holding_table(
                VC_kmh=VC_kmh_ideal, D_prime_m=D_prime_ideal,
                A_pl=A_ideal, k_pl=k_ideal, pct_low=80, pct_high=130, step=2)
            if df_hold_ideal is None or df_hold_ideal.empty:
                st.info("Aucune ligne exploitable.")
            else:
                if "Vitesse (km/h)" in df_hold_ideal.columns:
                    df_hold_ideal["Allure"] = df_hold_ideal["Vitesse (km/h)"].apply(_pace_str_from_kmh)
                st.dataframe(df_hold_ideal, hide_index=True, use_container_width=True)
            if A_ideal is not None and k_ideal is not None:
                st.caption(f"Power Law recalibré : A={A_ideal:.2f}, k={k_ideal:.3f}")
            if D_prime_ideal is not None:
                st.caption(f"D′ recalibré : {D_prime_ideal:.1f} m")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── INDEX CINÉTIQUE ──
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Index de Cinétique (sélection tests)")
    if len(tests_data) >= 2:
        test_names = [f"Test {t['i']}" for t in tests_data]
        colA_sel, colB_sel = st.columns(2)
        with colA_sel:
            sel_a = st.selectbox("Test court", test_names, key="ic_a")
        with colB_sel:
            sel_b = st.selectbox("Test long",  test_names, key="ic_b")
        tA = tests_data[test_names.index(sel_a)]
        tB = tests_data[test_names.index(sel_b)]
        ic_val, unite, msg, _, reco = compute_index_cinetique(
            tA.get("drift_pct"), tB.get("drift_pct"),
            tA.get("drift_bpm"), tB.get("drift_bpm"))
        if ic_val is not None and reco is not None:
            st.markdown(f"**IC = {ic_val*100:.1f}%** ({unite})")
            st.info(msg)
            st.markdown(f"**{reco['titre']}**")
            for s in reco["seances"]:
                st.markdown(f"• {s}")
        else:
            st.warning("Index non calculable avec ces deux tests.")
    else:
        st.info("Sélectionne au moins deux tests pour l'IC.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── EXPORT PDF ──
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("📄 Export PDF — Rapport complet des tests")
    if st.button("Générer le rapport PDF", key="export_pdf_btn"):
        figs_export = []
        if len(tests_data) > 0:
            figG, axG = plt.subplots(figsize=(9, 5))
            for t in tests_data:
                seg = t["segment"]; t0 = seg["time_s"].iloc[0]
                plot_multi_signals(axG, seg, t0=t0, who=f"T{t['i']}",
                                   show_fc=True,
                                   show_pace=(get_speed_col(seg) is not None),
                                   show_power=("power_smooth" in seg.columns))
            axG.set_title("Comparaison des cinétiques — Tous les tests")
            axG.set_xlabel("Temps segment (s)"); axG.grid(True, alpha=0.2)
            figs_export.append(figG)
        for t in tests_data:
            fig_i, ax_i = plt.subplots(figsize=(9, 4.8))
            seg = t["segment"]; t0 = seg["time_s"].iloc[0]
            plot_multi_signals(ax_i, seg, t0=t0, who=f"T{t['i']}",
                               show_fc=True,
                               show_pace=(get_speed_col(seg) is not None),
                               show_power=("power_smooth" in seg.columns))
            ax_i.set_title(f"Test {t['i']} — {t['date']}")
            ax_i.set_xlabel("Temps segment (s)"); ax_i.grid(True, alpha=0.2)
            figs_export.append(fig_i)
        pdf_buffer = fig_to_pdf_bytes(figs_export)
        st.download_button(
            label="📥 Télécharger le rapport PDF",
            data=pdf_buffer,
            file_name=f"rapport_tests_endurance_{date.today()}.pdf",
            mime="application/pdf",
        )
    st.markdown("</div>", unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════════
# ONGLET 2 — ANALYSE ENTRAÎNEMENT (Code 2)
# ════════════════════════════════════════════════════════════════
with main_tabs[2]:
    st.header("⚙️ Analyse entraînement (1 séance + intervalles + cinétiques)")

    if "training_session" not in st.session_state:
        st.session_state.training_session = None
    if "training_intervals" not in st.session_state:
        st.session_state.training_intervals = []

    uploaded_file = st.file_uploader(
        "Importer un fichier d'entraînement (FIT, GPX, CSV, TCX)",
        type=ACCEPTED_TYPES,
        key="training_file",
    )
    if uploaded_file:
        try:
            df_tr = load_activity(uploaded_file)
            df_tr, window, dur, pauses = smooth_hr(df_tr)
            st.session_state.training_session = (df_tr, window, dur, pauses, uploaded_file.name)
        except Exception as e:
            st.error(f"Erreur chargement séance : {e}")

    if st.session_state.training_session is None:
        st.info("Importe une séance pour commencer l'analyse.")
    else:
        df_tr, window, dur, pauses, filename = st.session_state.training_session

        st.markdown(f"### 📂 Séance importée : **{filename}**")
        st.caption(f"Durée totale : {dur:.1f}s · Lissage : {window}s · Pauses détectées : {pauses}")

        dur_int_tr = max(2, int(dur))
        st.markdown("## 📏 Définition des intervalles")
        for i, (start_s, end_s) in enumerate(st.session_state.training_intervals):
            st.markdown(f"**Intervalle {i+1}**")
            c1, c2, c3 = st.columns([5, 5, 1])
            with c1:
                s_sec = st.slider(
                    f"▶️ Début intervalle {i+1} (s)",
                    0, dur_int_tr - 1,
                    min(int(start_s), dur_int_tr - 1),
                    key=f"tr_int_start_{i}",
                )
            with c2:
                e_sec = st.slider(
                    f"⏹️ Fin intervalle {i+1} (s)",
                    1, dur_int_tr,
                    max(min(int(end_s), dur_int_tr), s_sec + 1),
                    key=f"tr_int_end_{i}",
                )
            with c3:
                st.write("")
                if st.button("🗑️", key=f"tr_del_int_{i}"):
                    st.session_state.training_intervals.pop(i)
                    st.rerun()
            st.caption(f"Segment : **{seconds_to_hms(s_sec)}** → **{seconds_to_hms(e_sec)}**  ({e_sec - s_sec}s)")
            if e_sec > s_sec:
                st.session_state.training_intervals[i] = (s_sec, e_sec)

        if st.button("➕ Ajouter un intervalle", key="tr_add_int"):
            st.session_state.training_intervals.append((0, 300))
            st.rerun()

        st.markdown("## 🔍 Analyse des intervalles")
        interval_segments = []

        for i, (s_sec, e_sec) in enumerate(st.session_state.training_intervals):
            seg = df_tr[(df_tr["time_s"] >= s_sec) & (df_tr["time_s"] <= e_sec)]
            if seg.empty or len(seg) < 10:
                continue
            interval_segments.append((i + 1, seg, s_sec, e_sec))

            stats, d_bpm, d_pct = analyze_heart_rate(seg)
            dist_m  = segment_distance_m(seg)
            t_s_raw = float(e_sec - s_sec)
            v_kmh_raw = (3.6 * dist_m / t_s_raw) if (t_s_raw > 0 and dist_m > 0) else 0.0
            D_up, D_down = segment_elevation_up_down(seg)
            grade_pct    = segment_grade_percent_net(seg)
            avgT, avgW, avgH = get_segment_weather(seg)

            manual_temp_int = st.number_input(
                f"🌡️ Température manuelle Intervalle {i+1} (°C) (si météo auto indispo)",
                value=float(0.0), step=0.5, key=f"manual_temp_int_{i}")
            temp_real = avgT if avgT is not None else (manual_temp_int if manual_temp_int != 0.0 else None)

            t_s_ideal = recalibrate_time_to_ideal_c2(
                time_seconds_raw=t_s_raw, D_up=D_up, D_down=D_down,
                distance_m=(dist_m if dist_m > 0 else 1000.0),
                temp_real=temp_real,
                k_up=c2_k_up, k_down=c2_k_down,
                k_temp_hot=c2_k_temp_hot, k_temp_cold=c2_k_temp_cold,
                opt_temp=c2_opt_temp,
            )
            v_kmh_ideal = (3.6 * dist_m / t_s_ideal) if (t_s_ideal and t_s_ideal > 0 and dist_m > 0) else None
            pace_raw_str   = _pace_str_from_kmh(v_kmh_raw)
            pace_ideal_str = _pace_str_from_kmh(v_kmh_ideal) if v_kmh_ideal else "–"
            d_v_kmh, d_v_pct = analyze_speed_kinetics(seg)

            st.markdown(f"### Intervalle {i+1} ({s_sec:.0f}s → {e_sec:.0f}s)")
            df_table = pd.DataFrame({
                "Métrique": [
                    "FC moyenne (bpm)", "FC max (bpm)", "FC min (bpm)",
                    "Dérive FC (bpm/min)", "Dérive FC (%/min)",
                    "Dérive vitesse (km/h/min)", "Dérive vitesse (%/min)",
                    "Durée réelle (s)", "Durée recalibrée (s)",
                    "Distance (m)", "Vitesse réelle (km/h)", "Allure réelle",
                    "Vitesse recalibrée (km/h)", "Allure recalibrée (conditions idéales)",
                    "Pente nette moyenne (%)", "D+ (m)", "D- (m)",
                    "Température (°C)", "Vent (m/s)", "Humidité (%)",
                ],
                "Valeur": [
                    stats.get("FC moyenne (bpm)"), stats.get("FC max (bpm)"), stats.get("FC min (bpm)"),
                    d_bpm, d_pct, d_v_kmh, d_v_pct,
                    round(t_s_raw, 1),
                    (round(t_s_ideal, 1) if t_s_ideal is not None else None),
                    round(dist_m, 1),
                    round(v_kmh_raw, 2), pace_raw_str,
                    (round(v_kmh_ideal, 2) if v_kmh_ideal is not None else None),
                    pace_ideal_str,
                    (round(grade_pct, 3) if grade_pct is not None else None),
                    round(D_up, 1), round(D_down, 1),
                    (round(temp_real, 2) if temp_real is not None else None),
                    (round(avgW, 2) if avgW is not None else None),
                    (round(avgH, 2) if avgH is not None else None),
                ],
            })
            st.dataframe(style_metrics_table(df_table), hide_index=True, use_container_width=True)

            fig, ax = plt.subplots(figsize=(9, 4.2))
            plot_multi_signals(ax, seg, t0=s_sec, who=f"Int{i+1}",
                               show_fc=True,
                               show_pace=("speed_smooth" in seg.columns),
                               show_power=("power_smooth" in seg.columns))
            ax.set_title(f"Cinétique — Intervalle {i+1}")
            ax.grid(True, alpha=0.25)
            st.pyplot(fig)

        if interval_segments:
            st.markdown("## 📊 Graphique combiné — tous les intervalles superposés")
            show_fc_tr    = st.checkbox("☑ FC",       True,  key="comb_fc_training_v2")
            show_pace_tr  = st.checkbox("☑ Allure",   False, key="comb_pace_training_v2")
            show_power_tr = st.checkbox("☑ Puissance",False, key="comb_pow_training_v2")
            figC, axC = plt.subplots(figsize=(10, 4.8))
            for idx2, seg2, s0, s1 in interval_segments:
                plot_multi_signals(axC, seg2, t0=s0, who=f"Int{idx2}",
                                   show_fc=show_fc_tr,
                                   show_pace=show_pace_tr and ("speed_smooth" in seg2.columns),
                                   show_power=show_power_tr and ("power_smooth" in seg2.columns))
            axC.set_title("Cinétique combinée — Intervalles superposés")
            axC.set_xlabel("Temps segment (s)"); axC.grid(True, alpha=0.25)
            handles, labels = [], []
            for a in figC.axes:
                h, l = a.get_legend_handles_labels()
                handles += h; labels += l
            if handles:
                axC.legend(handles, labels, fontsize=8, loc="upper left")
            st.pyplot(figC)
