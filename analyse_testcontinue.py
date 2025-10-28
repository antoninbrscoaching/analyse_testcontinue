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

    # Conversion du temps
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # Création d’un temps continu (ignore les pauses)
    df["delta_t"] = df[time_col].diff().dt.total_seconds().fillna(0)

    # Écarter les valeurs aberrantes dues à la pause montre
    median_step = np.median(df["delta_t"][df["delta_t"] > 0])
    if np.isnan(median_step) or median_step == 0:
        median_step = 1
    df.loc[df["delta_t"] > 2 * median_step, "delta_t"] = median_step

    # Temps cumulatif d'effort continu
    df["time_s"] = df["delta_t"].cumsum()
    total_dur = df["time_s"].iloc[-1]

    # Fenêtre adaptative selon la durée d'effort
    if total_dur < 360:
        window_sec = 5
    elif total_dur < 900:
        window_sec = 10
    else:
        window_sec = 20

    # Calcul du pas moyen
    step = np.median(np.diff(df["time_s"]))
    if step <= 0 or np.isnan(step):
        step = 1
    window_size = max(1, int(window_sec / step))

    # Lissage de la fréquence cardiaque
    df["hr_smooth"] = df[hr_col].rolling(window_size, min_periods=1).mean()

    # Compter les pauses détectées (écarts > 2× médiane)
    pauses = (df["delta_t"] > 2 * median_step).sum()

    return df, window_sec, total_dur, pauses


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
        "Durée segment (s)": round(df["time_s"].iloc[-1] - df["time_s"].iloc[0], 1),
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

    # Vérification colonne temps
    if "timestamp" not in df.columns:
        st.error("Impossible de détecter une colonne de temps ('timestamp' ou 'time').")
        st.stop()

    # Conversion du temps et correction du décalage capteur
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    lag = st.slider("Correction du décalage capteur (s)", 0, 10, 0)
    df["timestamp"] = df["timestamp"] - pd.to_timedelta(lag, unit="s")

    # Lissage adaptatif avec gestion des pauses
    df, window_sec, total_dur, pauses = smooth_hr(df)
    st.info(f"Durée totale détectée : {total_dur:.1f} s — Lissage sur {window_sec} s — Pauses détectées : {pauses}")

    # ============================
    # 🎯 Sélection manuelle du segment à analyser (libre)
    # ============================
    st.subheader("🎯 Sélection du segment à analyser")

    max_minutes_detected = round(total_dur / 60, 1)
    st.caption(f"Durée détectée dans le fichier : {max_minutes_detected:.1f} minutes (tu peux choisir au-delà).")

    # Autoriser librement de 0 à 180 minutes
    start_min = st.number_input("Début du segment (en minutes)", min_value=0.0, max_value=180.0, value=0.0, step=0.5)
    end_min = st.number_input("Fin du segment (en minutes)", min_value=start_min, max_value=180.0, value=12.0, step=0.5)

    # Conversion en secondes
    start_sec = start_min * 60
    end_sec = end_min * 60

    # ⚠️ Si l'utilisateur dépasse la durée réelle, on limite
    if end_sec > df["time_s"].max():
        st.warning("⚠️ La fin du segment dépasse la durée réelle du fichier FIT. L'analyse sera limitée aux données disponibles.")
        end_sec = df["time_s"].max()

    # Extraction de l'intervalle
    interval_df = df[(df["time_s"] >= start_sec) & (df["time_s"] <= end_sec)]

    if len(interval_df) < 10:
        st.warning("Segment trop court ou inexistant.")
    else:
        st.subheader(f"📊 Analyse du segment de {start_min:.1f} à {end_min:.1f} min")

        hr_stats = analyze_heart_rate(interval_df)
        st.write(hr_stats)

        # Graphique FC
        fig, ax = plt.subplots()
        ax.plot(interval_df["time_s"], interval_df["hr_smooth"], color="crimson", label="FC lissée")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Fréquence cardiaque (bpm)")
        ax.set_title(f"Cinétique cardiaque ({start_min:.1f} à {end_min:.1f} min)")
        ax.legend()
        st.pyplot(fig)

# ============================================
# ⚡ Calculateur de vitesse critique
# ============================================

st.divider()
st.subheader("⚡ Calculateur de vitesse critique (Critical Speed)")

tests_data = st.text_area("Entre les données de tests (ex : '4000, 12:35; 5000, 15:45')")

if tests_data:
    cv, d_prime = compute_critical_speed(tests_data)
    st.write(f"**Vitesse critique : {cv:.2f} km/h**")
    st.write(f"**D′ : {d_prime:.2f} m**")
