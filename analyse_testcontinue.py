import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fitdecode
from scipy.stats import linregress
import math

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

    # Compter les pauses détectées
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
    drift_percent = (drift_per_min / mean_hr) * 100 if mean_hr > 0 else np.nan

    return {
        "FC moyenne (bpm)": round(mean_hr, 1),
        "FC max (bpm)": round(max_hr, 1),
        "FC min (bpm)": round(min_hr, 1),
        "Dérive (bpm/min)": round(drift_per_min, 3),
        "Dérive (%/min)": round(drift_percent, 3) if not np.isnan(drift_percent) else None,
        "Durée segment (s)": round(df["time_s"].iloc[-1] - df["time_s"].iloc[0], 1),
    }, drift_per_min


def parse_time_to_seconds(tstr):
    """Convertit un texte hh:mm:ss ou mm:ss ou ss en secondes (robuste)."""
    tstr = tstr.strip()
    parts = tstr.split(":")
    try:
        parts = [int(p) for p in parts]
    except:
        # Essayer les virgules/points -> secondes en float
        val = float(tstr.replace(",", "."))
        return int(round(val))
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 1:
        h, m, s = 0, 0, parts[0]
    else:
        raise ValueError("Format temps invalide")
    return int(h * 3600 + m * 60 + s)


def haversine_dist_m(lat1, lon1, lat2, lon2):
    """Distance Haversine en mètres entre 2 points."""
    # rayon moyen terre
    R = 6371008.8
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def segment_distance_m(df_seg):
    """
    Calcule la distance (m) sur le segment prioritairement via :
    1) colonne 'distance' cumulative (m)
    2) intégration de 'speed' (m/s) * delta_t
    3) Haversine sur lat/lon
    """
    if df_seg.empty or len(df_seg) < 2:
        return 0.0

    # 1) 'distance' cumulative en mètres
    for cname in df_seg.columns:
        if cname.lower() == "distance":
            d0 = float(df_seg[cname].iloc[0])
            d1 = float(df_seg[cname].iloc[-1])
            if np.isfinite(d0) and np.isfinite(d1):
                return max(0.0, d1 - d0)

    # 2) 'speed' * delta_t
    speed_col = None
    for cname in df_seg.columns:
        if cname.lower() == "speed":
            speed_col = cname
            break
    if speed_col is not None and "delta_t" in df_seg.columns:
        # intégrer m/s * s = m
        dist = float(np.nansum(df_seg[speed_col].fillna(0).values * df_seg["delta_t"].fillna(0).values))
        if dist > 0:
            return dist

    # 3) Haversine sur lat/lon
    if set(["lat","lon"]).issubset({c.lower() for c in df_seg.columns}):
        # retrouver vrai noms colonnes lat/lon
        lat_name = [c for c in df_seg.columns if c.lower()=="lat"][0]
        lon_name = [c for c in df_seg.columns if c.lower()=="lon"][0]
        lats = df_seg[lat_name].astype(float).values
        lons = df_seg[lon_name].astype(float).values
        dist = 0.0
        for i in range(1, len(df_seg)):
            if np.isfinite(lats[i-1]) and np.isfinite(lats[i]) and np.isfinite(lons[i-1]) and np.isfinite(lons[i]):
                dist += haversine_dist_m(lats[i-1], lons[i-1], lats[i], lons[i])
        if dist > 0:
            return dist

    return 0.0


def format_pace_min_per_km(v_kmh):
    """Retourne (minutes, secondes, min/km float) pour une vitesse km/h."""
    if v_kmh <= 0 or not math.isfinite(v_kmh):
        return None
    min_per_km = 60.0 / v_kmh
    total_seconds = int(round(min_per_km * 60.0))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return minutes, seconds, min_per_km


# ============================================
# 🖥️ Application Streamlit principale
# ============================================

st.set_page_config(page_title="Analyseur Endurance Nolio + VC", layout="wide")
st.title("🏃‍♂️ Analyseur de Tests d'Endurance + Vitesse Critique")

# ===================================================
# 🧩 Bloc 1 – Analyse du premier fichier
# ===================================================

st.header("🧪 Test 1")
uploaded_file1 = st.file_uploader("Importe ton premier test (FIT, GPX ou CSV)", type=["fit", "gpx", "csv"], key="file1")

interval_df1 = None
stats1 = None
drift1 = None
dist1_m = None
t1_s = None

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
    c11, c12 = st.columns(2)
    with c11:
        start_str1 = st.text_input("Début (Test 1)", value="0:00:00", key="start1")
    with c12:
        end_str1 = st.text_input("Fin (Test 1)", value="0:12:00", key="end1")

    try:
        start_sec1 = parse_time_to_seconds(start_str1)
        end_sec1 = parse_time_to_seconds(end_str1)
    except:
        st.error("Format temps invalide (hh:mm:ss).")
        st.stop()

    if end_sec1 <= start_sec1:
        st.error("La fin doit être supérieure au début (Test 1).")
        st.stop()

    if end_sec1 > df1["time_s"].max():
        st.warning("⚠️ Fin > données disponibles (Test 1). Limitation automatique à la fin du fichier.")
        end_sec1 = df1["time_s"].max()

    interval_df1 = df1[(df1["time_s"] >= start_sec1) & (df1["time_s"] <= end_sec1)]

    if len(interval_df1) > 10:
        st.subheader(f"📊 Résultats Test 1 ({start_str1} → {end_str1})")
        stats1, drift1 = analyze_heart_rate(interval_df1)
        st.write(stats1)

        # ➕ Distance sur le segment Test 1
        dist1_m = segment_distance_m(interval_df1)
        t1_s = float(end_sec1 - start_sec1)
        st.write(f"**Distance segment (Test 1)** : {dist1_m:.1f} m — **Temps** : {t1_s:.1f} s — **Vitesse moy.** : {3.6 * (dist1_m / t1_s) if t1_s>0 else 0:.2f} km/h")

        fig1, ax1 = plt.subplots()
        ax1.plot(interval_df1["time_s"], interval_df1["hr_smooth"], label="FC Test 1")
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

interval_df2 = None
stats2 = None
drift2 = None
dist2_m = None
t2_s = None

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
    c21, c22 = st.columns(2)
    with c21:
        start_str2 = st.text_input("Début (Test 2)", value="0:00:00", key="start2")
    with c22:
        end_str2 = st.text_input("Fin (Test 2)", value="0:12:00", key="end2")

    try:
        start_sec2 = parse_time_to_seconds(start_str2)
        end_sec2 = parse_time_to_seconds(end_str2)
    except:
        st.error("Format temps invalide (hh:mm:ss).")
        st.stop()

    if end_sec2 <= start_sec2:
        st.error("La fin doit être supérieure au début (Test 2).")
        st.stop()

    if end_sec2 > df2["time_s"].max():
        st.warning("⚠️ Fin > données disponibles (Test 2). Limitation automatique à la fin du fichier.")
        end_sec2 = df2["time_s"].max()

    interval_df2 = df2[(df2["time_s"] >= start_sec2) & (df2["time_s"] <= end_sec2)]

    if len(interval_df2) > 10:
        st.subheader(f"📊 Résultats Test 2 ({start_str2} → {end_str2})")
        stats2, drift2 = analyze_heart_rate(interval_df2)
        st.write(stats2)

        # ➕ Distance sur le segment Test 2
        dist2_m = segment_distance_m(interval_df2)
        t2_s = float(end_sec2 - start_sec2)
        st.write(f"**Distance segment (Test 2)** : {dist2_m:.1f} m — **Temps** : {t2_s:.1f} s — **Vitesse moy.** : {3.6 * (dist2_m / t2_s) if t2_s>0 else 0:.2f} km/h")

        fig2, ax2 = plt.subplots()
        ax2.plot(interval_df2["time_s"], interval_df2["hr_smooth"], label="FC Test 2", color="tab:orange")
        ax2.set_xlabel("Temps (s)")
        ax2.set_ylabel("Fréquence cardiaque (bpm)")
        ax2.set_title("Cinétique cardiaque - Test 2")
        ax2.legend()
        st.pyplot(fig2)

# ===================================================
# 📊 Analyse générale (VC + Indice d'endurance)
# ===================================================

if (dist1_m is not None) and (t1_s is not None) and (dist2_m is not None) and (t2_s is not None) and t1_s>0 and t2_s>0:
    st.divider()
    st.header("📊 Analyse générale : Vitesse Critique & Indice d’endurance")

    # Ordonner court/long pour l’indice d’endurance
    # (on considère "court" = durée la plus courte)
    if t1_s <= t2_s:
        drift_short = drift1
        drift_long  = drift2
        label_short = "Test 1"
        label_long  = "Test 2"
    else:
        drift_short = drift2
        drift_long  = drift1
        label_short = "Test 2"
        label_long  = "Test 1"

    # Calcul VC (2 points)
    D1, T1 = float(dist1_m), float(t1_s)
    D2, T2 = float(dist2_m), float(t2_s)

    if T2 == T1 or D1 <= 0 or D2 <= 0 or T1 <= 0 or T2 <= 0:
        st.error("Impossible de calculer la VC automatiquement (distances/temps invalides).")
    else:
        CS = (D2 - D1) / (T2 - T1)                 # m/s
        D_prime = D1 - CS * T1                     # m
        V_kmh = CS * 3.6

        st.subheader("Résultats VC (automatique depuis tes segments)")

        if V_kmh <= 0 or not math.isfinite(V_kmh):
            st.error("Vitesse critique non calculable (vérifie les segments).")
        else:
            pace = format_pace_min_per_km(V_kmh)
            if pace is not None:
                minutes, seconds, min_per_km = pace
            st.write(f"**Vitesse critique (CS)** = {CS:.2f} m/s")
            st.write(f"**Vitesse critique** = {V_kmh:.2f} km/h")
            if pace is not None:
                st.write(f"**Allure VC** = {minutes}:{seconds:02d} min/km  ({min_per_km:.2f} min/km)")
            st.write(f"**Capacité anaérobie (D′)** = {D_prime:.2f} m")

        # Indice d’endurance basé sur les dérives
        st.subheader("Indice d’endurance (basé dérive)")
        if (drift_short is None) or (drift_long is None) or not np.isfinite(drift_short) or drift_short == 0:
            st.warning("Indice d’endurance non calculable (dérive courte invalide).")
        else:
            IE = 1.0 - (drift_long / drift_short)
            st.write(f"**Indice d’endurance** = {IE:.3f} (courte = {label_short}, longue = {label_long})")
            st.caption("Interprétation : proche de 1 ⇒ très bonne stabilité relative sur l'effort long ; proche de 0 ⇒ dérive longue similaire à la courte ; négatif ⇒ dérive longue > dérive courte (endurance à renforcer).")

    # Rappel des mesures utilisées
    st.subheader("Mesures utilisées")
    colA, colB = st.columns(2)
    with colA:
        st.write(f"**Test 1** : Distance = {dist1_m:.1f} m, Temps = {t1_s:.1f} s")
    with colB:
        st.write(f"**Test 2** : Distance = {dist2_m:.1f} m, Temps = {t2_s:.1f} s")
