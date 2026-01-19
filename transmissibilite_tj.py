import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px

# ==============================
# CONFIGURATION
# ==============================

COL_DATE = "Date"
COL_DEPT = "Département"
COL_TAUX = "Taux de transmissibilité"
COL_NN_FLUX = "Nb de procédures NN transmises par flux"
COL_NN_REMAT = "Nb de procédures NN rematérialisées"
COL_PAPIER = "Nb de procédures papier transmises"

# Dates minimales par source
MIN_DATE_BY_SOURCE = {
    "PN": pd.Timestamp("2022-04-01"),
    "GN": pd.Timestamp("2022-05-01"),
    "PN+GN": pd.Timestamp("2022-05-01")  # on exclut avril 2022 pour PN+GN
}

FORECAST_MONTHS = 14
SARIMAX_ORDER = (1, 0, 0)
SARIMAX_SEASONAL_ORDER = (0, 1, 1, 12)
SMOOTHING_WINDOW = 3


# ==============================
# FONCTIONS UTILITAIRES
# ==============================

def ensure_rate(df):
    """Calcule le taux de transmissibilité si absent, et le borne entre 0 et 1."""
    if COL_TAUX not in df.columns:
        num = df[COL_NN_FLUX].fillna(0) + df[COL_NN_REMAT].fillna(0)
        den = num + df[COL_PAPIER].fillna(0)
        df[COL_TAUX] = np.where(den > 0, num / den, np.nan)
    df[COL_TAUX] = df[COL_TAUX].astype(float).clip(0, 1)
    return df


def prepare_dates(df, min_date):
    """
    Convertit les dates écrites avec des mois français,
    filtre à partir de min_date, aligne au début de mois.
    """
    df = df.copy()

    mois_fr_en = {
        "janvier": "January",
        "février": "February",
        "fevrier": "February",
        "mars": "March",
        "avril": "April",
        "mai": "May",
        "juin": "June",
        "juillet": "July",
        "août": "August",
        "aout": "August",
        "septembre": "September",
        "octobre": "October",
        "novembre": "November",
        "décembre": "December",
        "decembre": "December"
    }

    # On passe tout en string, minuscule, et on remplace les mois FR par EN
    df[COL_DATE] = (
        df[COL_DATE]
        .astype(str)
        .str.lower()
        .replace(mois_fr_en, regex=True)
    )

    # Conversion en datetime
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")

    # Filtre sur la date minimale propre à la source
    df = df[df[COL_DATE] >= min_date]

    # Alignement au début du mois (monthly)
    df[COL_DATE] = df[COL_DATE].values.astype("datetime64[M]")

    return df


def global_mean(df):
    """Moyenne globale mensuelle des taux (tous départements confondus)."""
    return (
        df
        .groupby(COL_DATE)[COL_TAUX]
        .mean()
        .sort_index()
    )


def threshold_last_6_months(series):
    """Seuil = moyenne des 6 derniers mois de la série globale."""
    last6 = series.dropna().tail(6)
    return float(last6.mean()) if len(last6) else 0.0


def series_dept(df, dept):
    """Série mensuelle du taux pour un département, avec fréquence MS."""
    s = (
        df[df[COL_DEPT] == dept]
        .groupby(COL_DATE)[COL_TAUX]
        .mean()
        .sort_index()
    )
    # On force une fréquence mensuelle (Monthly Start)
    s = s.asfreq("MS")
    return s


def smooth(s, w=3):
    """Moyenne mobile centrée, robuste."""
    return s.rolling(window=w, min_periods=1, center=True).mean()


def build_exog(global_series, index):
    """Aligne la série globale sur l'index, étend avec la dernière valeur si nécessaire."""
    exog = global_series.reindex(index)
    if exog.isna().any() and len(global_series) > 0:
        exog = exog.fillna(global_series.iloc[-1])
    return exog


def model_and_forecast(y, global_series, floor):
    """
    Ajuste SARIMAX sur y (série départementale),
    avec exog = moyenne globale, lissage,
    et contrainte de plancher (floor) sur les prévisions.
    Retourne une Series avec index datetime, historique + futur.
    """
    y = y.dropna()
    if y.empty:
        return None

    # S'assure que l'index est de type datetime avec freq mensuelle
    y.index = pd.to_datetime(y.index)
    y = y.asfreq("MS")

    exog_hist = build_exog(global_series, y.index)

    model = SARIMAX(
        y,
        exog=exog_hist,
        order=SARIMAX_ORDER,
        seasonal_order=SARIMAX_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)

    future_index = pd.date_range(
        start=y.index.max() + pd.offsets.MonthBegin(1),
        periods=FORECAST_MONTHS,
        freq="MS"
    )
    exog_future = build_exog(global_series, future_index)

    forecast = res.get_forecast(steps=FORECAST_MONTHS, exog=exog_future).predicted_mean

    # Concat historique + prévisions
    full = pd.concat([y, forecast])
    full.index = pd.to_datetime(full.index)

    # Lissage
    full = smooth(full, SMOOTHING_WINDOW)

    # Séparation historique / futur
    hist = full.loc[full.index <= y.index.max()]
    fut = full.loc[full.index > y.index.max()]

    # Application du plancher sur le futur uniquement
    fut = fut.clip(lower=floor, upper=1.0)

    full_final = pd.concat([hist, fut])
    return full_final


# ==============================
# STREAMLIT UI
# ==============================

st.title("📊 Modélisation du Taux de Transmissibilité par Département")
st.write("Prévisions SARIMAX avec seuil minimal, dates FR, lissage et sélection PN / GN / PN+GN.")

uploaded = st.file_uploader(
    "Charge ton fichier Excel Taux_de_transmissibilité_données.xlsx",
    type=["xlsx"]
)

if uploaded is not None:

    sheets = {
        "PN": "TT_PN",
        "GN": "TT_GN",
        "PN+GN": "TT_PN+GN"
    }

    source_choice = st.selectbox("Choisis la source", list(sheets.keys()))

    # Lecture de la bonne feuille
    df = pd.read_excel(uploaded, sheet_name=sheets[source_choice])

    # Préparation dates avec min_date spécifique à la source
    min_date = MIN_DATE_BY_SOURCE[source_choice]
    df = prepare_dates(df, min_date)

    # Calcul / nettoyage du taux
    df = ensure_rate(df)

    if df.empty:
        st.warning("Aucune donnée après le filtre de date minimale pour cette source.")
    else:
        # Série globale
        global_series = global_mean(df)
        floor = threshold_last_6_months(global_series)

        st.info(f"Seuil minimal appliqué aux prévisions : **{floor:.4f}**")

        # Liste des départements
        dept_list = sorted(df[COL_DEPT].dropna().unique())
        dept_choice = st.selectbox("Choisis un département", dept_list)

        y = series_dept(df, dept_choice)

        if y.dropna().empty:
            st.warning("Aucune donnée exploitable pour ce département.")
        else:
            result = model_and_forecast(y, global_series, floor)

            if result is None or result.dropna().empty:
                st.error("Impossible de modéliser ce département (série trop courte ou vide après nettoyage).")
            else:
                # Construction du DataFrame de visualisation
                df_plot = pd.DataFrame({
                    "Date": result.index,
                    "Taux": result.values
                })

                # Forcer Date en datetime
                df_plot["Date"] = pd.to_datetime(df_plot["Date"], errors="coerce")

                # Dernière date historique (à partir de y)
                last_hist_date = pd.to_datetime(y.index.max())

                # Marquage Historique / Prévision selon la date
                df_plot["Type"] = np.where(
                    df_plot["Date"] <= last_hist_date,
                    "Historique",
                    "Prévision"
                )

                # Graphique
                fig = px.line(
                    df_plot,
                    x="Date",
                    y="Taux",
                    color="Type",
                    title=f"Taux de transmissibilité – {dept_choice} ({source_choice})",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)

                # Tableau
                st.subheader("Tableau des valeurs")
                st.dataframe(df_plot)

                # Export CSV
                csv = df_plot.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Télécharger les données (CSV)",
                    csv,
                    f"taux_transmissibilite_{dept_choice}_{source_choice}.csv",
                    "text/csv"
                )

else:
    st.info("Charge ton fichier Excel pour commencer.")
