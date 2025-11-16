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

# ========================= CONFIG ==============================
st.set_page_config(page_title="Analyse Tests Endurance + VC", layout="wide")

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
}
</style>
""", unsafe_allow_html=True)

ACCEPTED_TYPES = ["fit","FIT","gpx","GPX","csv","CSV","tcx","TCX"]

# ========================= LECTURE FICHIERS ==============================

def load_activity(file):
    name = file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(file)

    elif name.endswith(".fit"):
        data = []
        try:
            with fitdecode.FitReader(file) as fit:
                for frame in fit:
                    if isinstance(frame, fitdecode.records.FitDataMessage) and frame.name=="record":
                        data.append({f.name:f.value for f in frame.fields})
            df = pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Erreur FIT : {e}")

    elif name.endswith(".gpx"):
        gpx = gpxpy.parse(file)
        data = []
        for trk in gpx.tracks:
            for seg in trk.segments:
                for pt in seg.points:
                    data.append({"timestamp": pt.time, "lat":pt.latitude,
                                 "lon":pt.longitude,"alt":pt.elevation})
        df = pd.DataFrame(data)

    elif name.endswith(".tcx"):
        try:
            content = file.read().decode("utf-8", errors="ignore")
            root = ET.fromstring(content)

            data=[]
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

            df=pd.DataFrame(data)
            if df.empty:
                raise ValueError("TCX vide")
        except Exception as e:
            raise ValueError(f"Erreur TCX : {e}")

    else:
        raise ValueError("Format non supporté")

    # Harmonisation timestamp
    for c in df.columns:
        if "time" in c.lower():
            df=df.rename(columns={c:"timestamp"})
            break

    if "heart_rate" not in df.columns:
        raise ValueError("Pas de FC détectée")
    df = df.dropna(subset=["heart_rate"]).reset_index(drop=True)

    # Nettoyage
    for c in ["heart_rate","speed","enhanced_speed","power","distance"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# ========================= OUTILS ==============================

def get_speed_col(df):
    if "enhanced_speed" in df.columns: return "enhanced_speed"
    if "speed" in df.columns: return "speed"
    return None

def smooth_hr(df):
    df=df.copy()
    df["timestamp"]=pd.to_datetime(df["timestamp"], errors="coerce")
    df=df.dropna(subset=["timestamp"]).reset_index(drop=True)

    df["delta_t"]=df["timestamp"].diff().dt.total_seconds().fillna(0)
    median_step=np.median(df["delta_t"][df["delta_t"]>0])
    if np.isnan(median_step) or median_step==0: median_step=1
    df.loc[df["delta_t"]>2*median_step,"delta_t"]=median_step
    df["time_s"]=df["delta_t"].cumsum()
    total_dur=df["time_s"].iloc[-1]

    window_sec = 5 if total_dur<360 else 10 if total_dur<900 else 20
    step = np.median(np.diff(df["time_s"])); step = step if step>0 else 1
    win = max(1,int(window_sec/step))

    df["hr_smooth"]=df["heart_rate"].rolling(win,min_periods=1).mean()

    sp=get_speed_col(df)
    if sp: df["speed_smooth"]=df[sp].rolling(win,min_periods=1).mean()
    if "power" in df.columns: df["power_smooth"]=df["power"].rolling(win,min_periods=1).mean()

    pauses=(df["delta_t"]>2*median_step).sum()
    return df, window_sec, total_dur, pauses

def analyze_heart_rate(df):
    hr=df["hr_smooth"].dropna()
    mean_hr=hr.mean()
    max_hr=hr.max()
    slope,_,_,_,_=linregress(df["time_s"], df["hr_smooth"])
    drift_per_min=slope*60
    drift_pct=(drift_per_min/mean_hr)*100 if mean_hr>0 else None
    dur=df["time_s"].iloc[-1] - df["time_s"].iloc[0]

    stats={
        "FC moyenne": round(mean_hr,1),
        "FC max": round(max_hr,1),
        "Durée": round(dur,1)
    }
    return stats, float(drift_per_min), (None if drift_pct is None else float(drift_pct))

def segment_distance_m(df):
    if "distance" in df.columns and df["distance"].notna().sum()>1:
        return float(df["distance"].iloc[-1] - df["distance"].iloc[0])

    sp=get_speed_col(df)
    if sp and "delta_t" in df.columns:
        return float(np.nansum(df[sp] * df["delta_t"]))

    return 0.0

def compute_pace(v_kmh):
    if v_kmh<=0: return None
    pace_min = 60/v_kmh
    sec=int(round(pace_min*60))
    return sec//60, sec%60

def pace_formatter(v, pos):
    if v<=0 or not math.isfinite(v): return ""
    m=int(v); s=int(round((v-m)*60))
    return f"{m}:{s:02d}"

def plot_multi(ax, df, t0, color_fc, color_pace, color_pow,
               show_fc=True, show_pace=True, show_pow=True):
    tt=df["time_s"]-t0
    if show_fc:
        ax.plot(tt, df["hr_smooth"], color=color_fc, lw=1.8)
        ax.set_ylabel("FC (bpm)")

    if show_pace and "speed_smooth" in df.columns:
        pace_ax=ax.twinx()
        pace_series = 1000/(df["speed_smooth"]*60)
        pace_ax.plot(tt, pace_series, color=color_pace, lw=1.8)
        pace_ax.yaxis.set_major_formatter(FuncFormatter(pace_formatter))
        pace_ax.invert_yaxis()
        pace_ax.set_ylabel("Allure (min/km)")

    if show_pow and "power_smooth" in df.columns:
        pow_ax=ax.twinx()
        pow_ax.spines["right"].set_position(("outward",60))
        pow_ax.plot(tt, df["power_smooth"], color=color_pow, lw=1.8)
        pow_ax.set_ylabel("Puissance (W)")

def fig_to_pdf_bytes(figs):
    buf=BytesIO()
    with PdfPages(buf) as pdf:
        for f in figs:
            f.tight_layout()
            pdf.savefig(f, bbox_inches="tight")
    buf.seek(0)
    return buf

# ========================= INITIALISATION ==============================
if "tests_data" not in st.session_state:
    st.session_state.tests_data = []   # liste des tests (df, stats, etc.)

# --------------------------------------------------------------
# STOP ICI — L’Onglet 1 va commencer dans le BLOC 2/4
# --------------------------------------------------------------

# =====================================================
# ========= ONGLET 1 : MULTI-TESTS (2 à 6 tests) =======
# =====================================================

with tabs[0]:
    st.header("🧪 Tests d'endurance (2 à 6 tests)")

    # On stocke ici toutes les données tests analysées
    tests_data = []   # chaque entrée : dict avec df, segment, stats, dérive, distance, etc.

    # Nombre de tests sélectionnés
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

    # =====================================================
    # ==== Création dynamique des tests 1 → n = 2..6 ======
    # =====================================================

    for i in range(1, st.session_state.nb_tests + 1):

        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader(f"📌 Test {i}")

        # ---------- Upload ----------
        uploaded = st.file_uploader(
            f"Fichier Test {i} (FIT, GPX, CSV, TCX)",
            type=ACCEPTED_TYPES,
            key=f"file_{i}"
        )

        test_date = st.date_input(f"📅 Date du test {i}", value=date.today(), key=f"date_{i}")

        # Options affichage par test
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

            # Nettoyage temps + lissage
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

            lag = st.slider(
                f"Correction du décalage capteur (s) — Test {i}", 
                0, 10, 0, key=f"lag_{i}"
            )
            df["timestamp"] -= pd.to_timedelta(lag, unit="s")

            df, window, total_dur, pauses = smooth_hr(df)
            st.caption(f"Durée détectée : {total_dur:.1f}s • Lissage : {window}s • Pauses détectées : {pauses}")

            # ---------- Sélection du segment ----------
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

            # Limite si dépasse temps réel fichier
            if end_sec > df["time_s"].max():
                st.warning(f"⚠️ Fin > fichier ({df['time_s'].max():.0f}s). Limitation auto.")
                end_sec = df["time_s"].max()

            segment = df[(df["time_s"] >= start_sec) & (df["time_s"] <= end_sec)]

            if len(segment) < 10:
                st.warning("Segment trop court pour analyse.")
                st.markdown("</div>", unsafe_allow_html=True)
                continue

            # ---------- Analyse FC / dérive ----------
            stats, drift_bpm, drift_pct = analyze_heart_rate(segment)
            dist_m = segment_distance_m(segment)
            t_s = float(end_sec - start_sec)
            v_kmh = 3.6 * dist_m / t_s if t_s > 0 else 0

            pace = format_pace_min_per_km(v_kmh)
            pace_str = f"{pace[0]}:{pace[1]:02d} min/km" if pace else "–"

            # Tableau
            df_table = pd.DataFrame({
                "Métrique": [
                    "FC moyenne (bpm)", "FC max (bpm)",
                    "Dérive (bpm/min)", "Dérive (%/min)",
                    "Durée (s)", "Distance (m)", "Vitesse (km/h)", "Allure (min/km)"
                ],
                "Valeur": [
                    stats["FC moyenne (bpm)"], stats["FC max (bpm)"],
                    drift_bpm, drift_pct,
                    t_s, round(dist_m, 1), round(v_kmh, 2), pace_str
                ]
            })
            st.dataframe(df_table, hide_index=True, use_container_width=True)

            # ---------- Graphique ----------
            fig, ax = plt.subplots(figsize=(9, 4.6))
            plot_multi_signals(
                ax, segment, t0=start_sec, who=f"T{i}",
                show_fc=show_fc,
                show_pace=show_pace and (get_speed_col(segment) is not None),
                show_power=show_power and ("power_smooth" in segment.columns)
            )
            ax.set_title(f"Cinétique — Test {i} ({test_date})")
            ax.set_xlabel("Temps segment (s)")
            ax.grid(True, alpha=0.2)

            # Légende fusionnée multi-axes
            handles, labels = [], []
            for a in fig.axes:
                h, l = a.get_legend_handles_labels()
                handles += h; labels += l
            if handles:
                ax.legend(handles, labels, fontsize=8, loc="upper left", frameon=False)

            st.pyplot(fig)

            # ---------- Enregistrement dans tests_data ----------
            tests_data.append({
                "i": i,
                "df": df,
                "segment": segment,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "stats": stats,
                "drift_bpm": drift_bpm,
                "drift_pct": drift_pct,
                "dist_m": dist_m,
                "t_s": t_s,
                "v_kmh": v_kmh,
                "pace_str": pace_str,
                "date": test_date,
            })

        st.markdown('</div>', unsafe_allow_html=True)

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

if len(tests_data) >= 2:

    # Récupération distances – durées
    D = np.array([t["dist_m"] for t in tests_data])
    T = np.array([t["t_s"] for t in tests_data])

    # Modèle hyperbolique :
    # D = W' + VC * T
    # On réécrit => D = a + b*T où b = VC
    # Régression linéaire standard
    slope, intercept = np.polyfit(T, D, 1)

    VC_m_s = slope                   # vitesse critique (m/s)
    D_prime = intercept              # capacité anaérobie W'

    VC_kmh = VC_m_s * 3.6

    # conversion allure min/km
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
        f"(Régression hyperbolique sur {len(T)} tests)"
    )

else:
    st.info("Il faut au moins deux tests pour calculer la VC (modèle hyperbolique).")

st.markdown('</div>', unsafe_allow_html=True)



# ============================================================
# ===================== MODÈLE LOG-LOG ========================
# ============================================================

st.markdown('<div class="report-card">', unsafe_allow_html=True)
st.subheader("📈 Modèle Log-Log (T = A · V^{-k})")

if len(tests_data) >= 2:

    # Conversion V = vitesse moyenne (m/s)
    V = np.array([t["dist_m"] / t["t_s"] for t in tests_data])
    TT = np.array([t["t_s"] for t in tests_data])

    # log-log
    X = np.log(1 / V)
    Y = np.log(TT)

    k, lnA = np.polyfit(X, Y, 1)
    A = np.exp(lnA)

    st.write(f"**k = {k:.3f}**, **A = {A:.2f}** (modèle log-log)")

else:
    st.info("Au moins 2 tests requis pour le modèle log-log.")

st.markdown('</div>', unsafe_allow_html=True)



# ============================================================
# ====================== INDEX CINÉTIQUE ======================
# ============================================================

st.markdown('<div class="report-card">', unsafe_allow_html=True)
st.subheader("⚙️ Index de Cinétique (sélection tests)")

if len(tests_data) >= 2:

    # Choix des 2 tests :
    test_names = [f"Test {t['i']}" for t in tests_data]

    colA, colB = st.columns(2)
    with colA:
        sel_a = st.selectbox("Test court", test_names, key="ic_a")
    with colB:
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

    # === Graphique général (superposé) ===
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

        # Légende
        handles, labels = [], []
        for a in figG.axes:
            h, l = a.get_legend_handles_labels()
            handles += h; labels += l
        if handles:
            axG.legend(handles, labels, fontsize=7, loc="upper left")

        figs_export.append(figG)



    # === Chaque test individuellement ===
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



    # === Export PDF en mémoire ===
    pdf_buffer = fig_to_pdf_bytes(figs_export)

    st.download_button(
        label="📥 Télécharger le rapport PDF",
        data=pdf_buffer,
        file_name=f"rapport_tests_endurance_{date.today()}.pdf",
        mime="application/pdf"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Onglet 2 : Analyse entraînement ----------
with tabs[1]:
    st.session_state.active_tab = "training"
    st.header("⚙️ Analyse entraînement (1 séance + intervalles + graphique combiné)")

    # --- Initialisation ---
    if "training_session" not in st.session_state:
        st.session_state.training_session = None
    if "training_intervals" not in st.session_state:
        st.session_state.training_intervals = []

    # --- Import d'une seule séance ---
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

    # ------- INTERVALLES -------
    st.markdown("## 📏 Définition des intervalles")

    for i, (start_s, end_s) in enumerate(st.session_state.training_intervals):
        c1, c2, c3 = st.columns([1, 1, 0.3])

        with c1:
            s_str = st.text_input(
                f"Début intervalle {i+1} (hh:mm:ss)",
                value=f"{int(start_s//60)}:{int(start_s%60):02d}",
                key=f"int_start_{i}"
            )
        with c2:
            e_str = st.text_input(
                f"Fin intervalle {i+1}",
                value=f"{int(end_s//60)}:{int(end_s%60):02d}",
                key=f"int_end_{i}"
            )
        with c3:
            if st.button("🗑️", key=f"del_int_{i}"):
                st.session_state.training_intervals.pop(i)
                st.rerun()

        try:
            s_sec = parse_time_to_seconds(s_str)
            e_sec = parse_time_to_seconds(e_str)
            if e_sec > s_sec:
                st.session_state.training_intervals[i] = (s_sec, e_sec)
        except:
            st.warning(f"⛔ Format invalide intervalle {i+1}")

    if st.button("➕ Ajouter un intervalle"):
        st.session_state.training_intervals.append((0, 300))
        st.rerun()

    # ------- ANALYSE DES INTERVALLES -------
    st.markdown("## 🔍 Analyse des intervalles")

    interval_segments = []

    for i, (s_sec, e_sec) in enumerate(st.session_state.training_intervals):

        seg = df[(df["time_s"] >= s_sec) & (df["time_s"] <= e_sec)]
        if seg.empty:
            continue

        interval_segments.append((i+1, seg, s_sec, e_sec))

        stats, d_bpm, d_pct = analyze_heart_rate(seg)
        dist_m = segment_distance_m(seg)
        t_s = e_sec - s_sec
        v_kmh = 3.6 * dist_m / t_s if t_s > 0 else 0
        pace = format_pace_min_per_km(v_kmh)
        pace_str = f"{pace[0]}:{pace[1]:02d} min/km" if pace else "–"

        st.markdown(f"### Intervalle {i+1} ({s_sec:.0f}s → {e_sec:.0f}s)")
        st.dataframe(pd.DataFrame({
            "Métrique": ["FC moyenne", "Dérive bpm/min", "Dérive %/min",
                         "Durée (s)", "Distance (m)", "Vitesse (km/h)", "Allure"],
            "Valeur": [stats["FC moyenne (bpm)"], d_bpm, d_pct,
                       t_s, round(dist_m, 1), round(v_kmh, 2), pace_str]
        }), hide_index=True, use_container_width=True)

        # Graphique individuel
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

    # ------- GRAPHIQUE COMBINÉ -------
    if interval_segments:
        st.markdown("## 📊 Graphique combiné — tous les intervalles superposés")

        show_fc = st.checkbox("☑ FC", True, key="comb_fc")
        show_pace = st.checkbox("☑ Allure", False, key="comb_pace")
        show_power = st.checkbox("☑ Puissance", False, key="comb_pow")

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
