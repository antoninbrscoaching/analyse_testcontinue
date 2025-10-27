import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fitparse import FitFile
from scipy.stats import linregress

# ============================================
# 🔧 Fonctions utilitaires
# ============================================

def load_activity(file):
    """Charge un fichier FIT, GPX ou CSV."""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".fit"):
        fitfile = FitFile(file)
        data = []
        for record in fitfile.get_messages("record"):
            record_data = {}
            for field in record:
                record_data[field.name] = field.value
            data.append(record_data)
        df = pd.DataFrame(data)
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

    # Nettoyage et validation
    if "heart_rate" not in df.columns:
        raise ValueError("Le fichier ne contient pas de données de fréquence cardiaque ('heart_rate').")

    df = df.dropna(subset=["heart_rate"]).reset_index(drop=True)
    return df


def smooth_hr(df, time_col="timestamp", hr_col="heart_rate"):
    """Lisse la fréquence cardiaque avec une fenêtre adaptée à la durée du test."""
    df = df.copy()
    df["time_s"] = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds()
    total_dur = df["time_s"].iloc[-1] - df["time_s"].iloc[0]

    # Fenêtre adaptative selon la durée
    if total_dur < 360:      # < 6 min
        window_sec = 5
    elif total_dur < 900:    # < 15 min
        window_sec = 10
    else:
        window_sec = 20

    df = df.set_index("time_s")
    df["hr_smooth"] = df[hr_col].rolling(f"{window_sec}s", min_periods=1).mean()
    df = df.reset_index()

    return df, window_sec


def get_interval(df, interval_number=4):
    """Extrait automatiquement l’intervalle n°4 s’il existe, sinon 4e quintile du fichier."""
    if "lap" in df.columns:
        return df[df["lap"] == interval_number]
    elif "interval" in df.columns:
        return df[df["interval"] == interval_number]
    else:
        total_len = len(df)
        parts = np.array_split(df, 5)
        if len(parts) >= interval_number:
            return parts[interval_number - 1]
        else:
            st.warning("Impossible de détecter l’intervalle 4, tout le fichier est utilisé.")
            return df


def analyze_heart_rate(df):
    """Analyse la FC moyenne, max, min et calcule la dérive via régression."""
    hr = df["hr_smooth"].dropna()
    mean_hr = hr.mean()
    max_hr = hr.max()
    min_hr = hr.min()

    # dérive linéaire FC(t)
    slope, _, _, _, _ = linregress(df["time_s"], df["hr_smooth"])
    drift_per_min = slope * 60      # bpm/min
    drift_percent = (drift_per_min / mean_hr) * 100

    return {
        "FC moyenne (bpm)": round(mean_hr, 1),
        "FC max (bpm)": round(max_hr, 1),
        "FC min (bpm)": round(min_hr, 1),
        "Dérive (bpm/min)": round(drift_per_min, 2),
        "Dérive (%/min)": round(drift_percent, 3),
        "Durée (s)": round(df["time_s"].iloc[-1] - df["time_s"].iloc[0], 1),
    }


def compute_critical_speed(tests_text):
    """Calcule la vitesse critique et D′ à partir de tests multiples."""
    pairs = []
    for item in tests_text.split(";"):
        if not item.strip():
            continue
        d, t = item.strip().split(",")
        distance = float(d)
        mm, ss = map(int, t.strip().split(":"))
        total_s = mm * 60 + ss
        pairs.append((distance, total_s))
    D = np.array([p[0] for p in pairs])
    T = np.array([p[1] for p in pairs])
    slope, intercept = np.polyfit(T, D, 1)
    cv = slope * 3.6   # m/s -> km/h
    d_prime = intercept
    return cv, d_prime


# ============================================
# 🖥️ Application Streamlit principale
# ============================================

st.set_page_config(page_title="Analyseur Endurance Nolio", layout="wide")

st.title("🏃‍♂️ Analyseur de Tests d'Endurance Nolio")

uploaded_file = st.file_uploader("Importe ton activité (FIT, GPX ou CSV)", type=["fit", "gpx", "csv"])

if uploaded_file:
    try:
        df = load_activity(uploaded_file)
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        st.stop()

    if "timestamp" not in df.columns:
        st.error("Ce fichier ne contient pas de colonne 'timestamp' valide.")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Slider correction du lag capteur
        lag = st.slider("Correction du décalage capteur (s)", 0, 10, 0)
        df["timestamp"] = df["timestamp"] - pd.to_timedelta(lag, unit="s")

        # Lissage adaptatif
        df, window_sec = smooth_hr(df)
        st.info(f"Lissage automatique sur {window_sec} secondes (selon la durée de l’effort).")

        # Intervalle 4
        interval_df = get_interval(df, interval_number=4)

        if len(interval_df) < 10:
            st.warning("Intervalle 4 non trouvé ou trop court.")
        else:
            st.subheader("📊 Analyse de l’intervalle 4")

            hr_stats = analyze_heart_rate(interval_df)
            st.write(hr_stats)

            # Graphique FC
            fig, ax = plt.subplots()
            ax.plot(interval_df["time_s"], interval_df["hr_smooth"], color="crimson", label="FC lissée")
            ax.set_xlabel("Temps (s)")
            ax.set_ylabel("Fréquence cardiaque (bpm)")
            ax.set_title("Cinétique cardiaque")
            ax.legend()
            st.pyplot(fig)

# ============================================
# ⚡ Calculateur de vitesse critique
# ============================================

st.divider()
st.subheader("⚡ Calculateur de vitesse critique (Critical Speed)")

tests_data = st.text_area("Entre les données de tests (ex: '4000, 12:35; 5000, 15:45')")

if tests_data:
    cv, d_prime = compute_critical_speed(tests_data)
    st.write(f"**Vitesse critique : {cv:.2f} km/h**")
    st.write(f"**D′ : {d_prime:.2f} m**")

