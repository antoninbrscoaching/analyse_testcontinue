import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fitdecode
from scipy.stats import linregress

# ============================================
# 🔧 Fonctions utilitaires
# ============================================

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
                        record_data = {}
                        for field in frame.fields:
                            record_data[field.name] = field.value
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

    # Harmoniser le nom de la colonne de temps
    for c in df.columns:
        if "time" in c.lower():
            df.rename(columns={c: "timestamp"}, inplace=True)
            break

    # Validation minimale
    if "heart_rate" not in df.columns:
        raise ValueError("Le fichier ne contient pas de données de fréquence cardiaque ('heart_rate').")

    df = df.dropna(subset=["heart_rate"]).reset_index(drop=True)
    return df


def smooth_hr(df, time_col="timestamp", hr_col="heart_rate"):
    """Lisse la fréquence cardiaque avec un temps continu (ignore les pauses montre)."""
    df = df.copy()
    df = df.sort_values(by=time_col).reset_index(drop=True)

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

    pauses = (df["delta_t"] > 2 * median_step).sum()

    return df, window_sec, total_dur, pauses


def analyze_heart_rate(df):
    """Analyse la FC moyenne, max, min et calcule la dérive via régression."""
    hr = df["hr_smooth"].dropna()
    mean_hr = hr.mean()
    max_hr = hr.max()
    min_hr = hr.min()

    slope, _, _, _, _ = linregress(df["time_s"], df["hr_smooth"])
    drift_per_min = slope * 60
    drift_percent = (drift_per_min / mean_hr) * 100

    return {
        "FC moyenne (bpm)": round(mean_hr, 1),
        "FC max (bpm)": round(max_hr, 1),
        "FC min (bpm)": round(min_hr, 1),
        "Dérive (bpm/min)": round(drift_per_min, 2),
        "Dérive (%/min)": round(drift_percent, 3),
        "Durée segment (s)": round(df["time_s"].iloc[-1] - df["time_s"].iloc[0], 1),
    }


def parse_time_to_seconds(tstr):
    """Convertit un texte hh:mm:ss ou mm:ss en secondes."""
    parts = [int(p) for p in tstr.strip().split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = 0, 0, parts[0]
    return h * 3600 + m * 60 + s


# ============================================
# 🖥️ Application Streamlit principale
# ============================================

st.set_page_config(page_title="Analyseur Endurance Nolio", layout="wide")

st.title("🏃‍♂️ Analyseur de Tests d'Endurance Nolio")

# ===================================================
# 🧩 Bloc 1 – Analyse du premier fichier
# ===================================================

st.header("🧪 Test 1")
uploaded_file1 = st.file_uploader("Importe ton premier test (FIT, GPX ou CSV)", type=["fit", "gpx", "csv"], key="file1")

if uploaded_file1:
    try:
        df1 = load_activity(uploaded_file1)
    except Exception as e:
        st.error(f"Erreur fichier 1 : {e}")
        st.stop()

    df1["timestamp"] = pd.to_datetime(df1["timestamp"], errors="coerce")
    df1 = df1.dropna(subset=["timestamp"])

    lag1 = st.slider("Correction du décalage capteur (s) - Test 1", 0, 10, 0, key="lag1")
    df1["timestamp"] = df1["timestamp"] - pd.to_timedelta(lag1, unit="s")

    df1, window_sec1, total_dur1, pauses1 = smooth_hr(df1)
    st.info(f"[Test 1] Durée détectée : {total_dur1:.1f}s — Lissage : {window_sec1}s — Pauses : {pauses1}")

    st.subheader("🎯 Sélection du segment Test 1 (format hh:mm:ss)")
    col1, col2 = st.columns(2)
    with col1:
        start_str1 = st.text_input("Début du segment Test 1", value="0:00:00", key="start1")
    with col2:
        end_str1 = st.text_input("Fin du segment Test 1", value="0:12:00", key="end1")

    try:
        start_sec1 = parse_time_to_seconds(start_str1)
        end_sec1 = parse_time_to_seconds(end_str1)
    except:
        st.error("Format temps invalide (hh:mm:ss).")
        st.stop()

    if end_sec1 > df1["time_s"].max():
        st.warning("⚠️ Fin au-delà des données disponibles (Test 1).")
        end_sec1 = df1["time_s"].max()

    interval_df1 = df1[(df1["time_s"] >= start_sec1) & (df1["time_s"] <= end_sec1)]

    if len(interval_df1) > 10:
        st.subheader(f"📊 Résultats Test 1 ({start_str1} → {end_str1})")
        stats1 = analyze_heart_rate(interval_df1)
        st.write(stats1)

        fig1, ax1 = plt.subplots()
        ax1.plot(interval_df1["time_s"], interval_df1["hr_smooth"], color="crimson", label="FC Test 1")
        ax1.set_xlabel("Temps (s)")
        ax1.set_ylabel("Fréquence cardiaque (bpm)")
        ax1.set_title("Cinétique cardiaque - Test 1")
        ax1.legend()
        st.pyplot(fig1)

# ===================================================
# 🧩 Bloc 2 – Analyse du second fichier
# ===================================================

st.divider()
st.header("🧪 Test 2")
uploaded_file2 = st.file_uploader("Importe ton second test (FIT, GPX ou CSV)", type=["fit", "gpx", "csv"], key="file2")

if uploaded_file2:
    try:
        df2 = load_activity(uploaded_file2)
    except Exception as e:
        st.error(f"Erreur fichier 2 : {e}")
        st.stop()

    df2["timestamp"] = pd.to_datetime(df2["timestamp"], errors="coerce")
    df2 = df2.dropna(subset=["timestamp"])

    lag2 = st.slider("Correction du décalage capteur (s) - Test 2", 0, 10, 0, key="lag2")
    df2["timestamp"] = df2["timestamp"] - pd.to_timedelta(lag2, unit="s")

    df2, window_sec2, total_dur2, pauses2 = smooth_hr(df2)
    st.info(f"[Test 2] Durée détectée : {total_dur2:.1f}s — Lissage : {window_sec2}s — Pauses : {pauses2}")

    st.subheader("🎯 Sélection du segment Test 2 (format hh:mm:ss)")
    col3, col4 = st.columns(2)
    with col3:
        start_str2 = st.text_input("Début du segment Test 2", value="0:00:00", key="start2")
    with col4:
        end_str2 = st.text_input("Fin du segment Test 2", value="0:12:00", key="end2")

    try:
        start_sec2 = parse_time_to_seconds(start_str2)
        end_sec2 = parse_time_to_seconds(end_str2)
    except:
        st.error("Format temps invalide (hh:mm:ss).")
        st.stop()

    if end_sec2 > df2["time_s"].max():
        st.warning("⚠️ Fin au-delà des données disponibles (Test 2).")
        end_sec2 = df2["time_s"].max()

    interval_df2 = df2[(df2["time_s"] >= start_sec2) & (df2["time_s"] <= end_sec2)]

    if len(interval_df2) > 10:
        st.subheader(f"📊 Résultats Test 2 ({start_str2} → {end_str2})")
        stats2 = analyze_heart_rate(interval_df2)
        st.write(stats2)

        fig2, ax2 = plt.subplots()
        ax2.plot(interval_df2["time_s"], interval_df2["hr_smooth"], color="royalblue", label="FC Test 2")
        ax2.set_xlabel("Temps (s)")
        ax2.set_ylabel("Fréquence cardiaque (bpm)")
        ax2.set_title("Cinétique cardiaque - Test 2")
        ax2.legend()
        st.pyplot(fig2)
