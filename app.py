import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import re

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

import pandas_datareader.data as web
import datetime

# ==============================
# Catégories multi-secteurs avec EMOJIS
# ==============================
CATEGORIES = {
    "🌐 Indices Mondiaux (ETFs)": {
        "S&P 500": "SPY", "NASDAQ 100": "QQQ", "Dow Jones": "DIA", "MSCI World": "URTH",
        "CAC 40 (France)": "EWQ", "FTSE 100 (UK)": "EWU", "DAX (Allemagne)": "EWG", "Nikkei 225 (Japon)": "EWJ",
    },
    "🇺🇸 Actions (US)": {
        "🍎 Apple": "AAPL", "💻 Microsoft": "MSFT", "🚗 Tesla": "TSLA", "📦 Amazon": "AMZN",
        "Alphabet (Google)": "GOOGL", "NVIDIA": "NVDA",
    },
    "🇨🇦 Actions (Canada)": {
        "🏦 Royal Bank": "RY.TO", "🛍️ Shopify": "SHOP.TO", "🛢️ Enbridge": "ENB.TO",
    },
    "🇫🇷 Actions (France)": {
        "👜 LVMH": "MC.PA", "💅 L'Oréal": "OR.PA", "⛽ TotalEnergies": "TTE.PA",
    },
    "🇬🇧 Actions (Royaume-Uni)": {
        "🛢️ Shell": "SHEL.L", "💊 AstraZeneca": "AZN.L", "🏦 HSBC": "HSBA.L",
    },
    "🇩🇪 Actions (Allemagne)": {
        "💻 SAP": "SAP.DE", "🔩 Siemens": "SIE.DE", "🚗 Volkswagen": "VOW3.DE",
    },
    "🇨🇭 Actions (Suisse)": {
        "🍫 Nestlé": "NESN.SW", "⚕️ Roche": "ROG.SW", " Novartis": "NOVN.SW",
    },
    "🇯🇵 Actions (Japon)": {
        "🚗 Toyota": "7203.T", "🎮 Sony": "6758.T", "SoftBank": "9984.T",
    },
    "🇨🇳 Actions (Chine & HK)": {
        "🍶 Kweichow Moutai": "600519.SS", "🛍️ Alibaba": "BABA", "🎮 Tencent": "0700.HK",
    },
    "🇮🇳 Actions (Inde)": {
        "Reliance Industries": "RELIANCE.NS", "Tata (TCS)": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
    },
    "🇧🇷 Actions (Brésil)": {
        "🛢️ Petrobras": "PBR", "Vale": "VALE", "🏦 Itaú Unibanco": "ITUB",
    },
    "🪙 Cryptomonnaies": {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Ripple": "XRP-USD", "Cardano": "ADA-USD",
    },
    "💱 Forex (Taux de Change)": {
        "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "GBP/USD": "GBPUSD=X", "AUD/USD": "AUDUSD=X",
    },
    "⛏️ Énergies": {
        "Pétrole Brut WTI": "CL=F", "Pétrole Brent": "BZ=F", "Gaz Naturel": "NG=F",
    },
    "💎 Métaux Précieux & Industriels": {
        "🥇 Or": "GC=F", "🥈 Argent": "SI=F", "Cuivre": "HG=F", "💎 Platine": "PL=F", "💍 Palladium": "PA=F",
    },
    "🚜 Agriculture": {
        "🌽 Maïs": "ZC=F", "🌾 Blé": "ZW=F", "🌱 Soja": "ZS=F", "☕ Café": "KC=F", "🍬 Sucre": "SB=F", "🧶 Coton": "CT=F",
    },
    "🏛️ Économie (FRED - US)": {
        "📉 Taux de chômage": "UNRATE", "📈 PIB (GDP)": "GDP", "💲 Inflation (CPI)": "CPIAUCSL",
        "Taux d'intérêt 10 ans": "DGS10", "Masse Monétaire M2": "M2SL",
    }
}

# Nom de la colonne cible utilisée dans tout le script
TARGET_COL_NAME = "Target_Value"

# ==============================
# Config Streamlit
# ==============================
st.set_page_config(page_title="📊 Prédictions Multi-Secteurs avec M.Haithem BERKANE", layout="wide")
st.title("📈 Plateforme de Prédiction Multi-Secteurs V2.0 Par M.Haithem BERKANE octobre 2025")

# ==============================
# Choix utilisateur
# ==============================
sector_display = st.sidebar.selectbox("Secteur", list(CATEGORIES.keys()))
symbol_name_display = st.sidebar.selectbox("Variable à prédire", list(CATEGORIES[sector_display].keys()))

# Récupérer les vraies valeurs sans emojis pour le traitement interne
sector = re.sub(r'[\U00010000-\U0010ffff]', '', sector_display).strip()
symbol = CATEGORIES[sector_display][symbol_name_display]

horizon_choice = st.sidebar.selectbox(
    "Horizon de prédiction",
    ["6 mois", "1 an", "3 ans", "5 ans"]
)

HORIZON_MAP = {
    "6 mois": {"train_years": "2y", "predict_days": 180},
    "1 an": {"train_years": "3y", "predict_days": 365},
    "3 ans": {"train_years": "7y", "predict_days": 3*365},
    "5 ans": {"train_years": "10y", "predict_days": 5*365},
}
train_period = HORIZON_MAP[horizon_choice]["train_years"]
future_days = HORIZON_MAP[horizon_choice]["predict_days"]

# ==============================
# Chargement et SELECTION AUTOMATIQUE des données
# ==============================
st.subheader(f"📊 Données historiques - {symbol_name_display}")

df = pd.DataFrame()
selected_column_name_display = "N/A" 

try:
    if sector == "Économie & Société (FRED)":
        start = datetime.datetime.now() - datetime.timedelta(days=365*20)
        end = datetime.datetime.now()
        df = web.DataReader(symbol, "fred", start, end)
        
    else:
        # Données yfinance
        years = int(train_period.replace('y', ''))
        start_date = datetime.datetime.now() - datetime.timedelta(days=365.25 * years)
        df = yf.download(symbol, start=start_date, end=datetime.datetime.now(), interval="1d")
        
        # Aplatir le MultiIndex des colonnes YFinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[-1] if isinstance(col, tuple) else col for col in df.columns.values]
            df.columns.name = None
        
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
    st.stop()

if df.empty:
    st.error("Aucune donnée disponible.")
    st.stop()

# Nettoyage des NaNs
df = df.dropna(axis=0, how='all')

# Identifier les colonnes numériques disponibles
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if not numeric_cols:
    st.error("Aucune colonne numérique trouvée dans les données chargées pour l'analyse.")
    st.stop()

# Logique de sélection automatique de la colonne
col_to_select = None
if sector == "Économie & Société (FRED)":
    col_to_select = symbol
elif 'Adj Close' in numeric_cols:
    col_to_select = 'Adj Close'
elif 'Close' in numeric_cols:
    col_to_select = 'Close'
else:
    col_to_select = numeric_cols[0]

if col_to_select not in numeric_cols:
    st.error(f"La colonne attendue '{col_to_select}' est manquante. Veuillez vérifier le symbole ou les données disponibles.")
    st.stop()

# ATTRIBUTION: Création de la colonne cible unique
selected_column_name_display = col_to_select 
try:
    df[TARGET_COL_NAME] = df[col_to_select].squeeze() 
except ValueError:
    df[TARGET_COL_NAME] = df[[col_to_select]].iloc[:, 0]
    
st.info(f"Colonne sélectionnée automatiquement pour l'analyse : **{selected_column_name_display}**")

# Affichage des données
st.line_chart(df[TARGET_COL_NAME])

# ==============================
# Préparation des données
# ==============================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[[TARGET_COL_NAME]])

look_back = 60

def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

if len(scaled_data) <= look_back:
    st.error(f"Pas assez de données pour le look_back de {look_back}. Veuillez choisir un horizon de prédiction plus long.")
    st.stop()
    
X, y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split train/test
split = int(len(X) * 0.8)
if split == 0 or split == len(X):
    st.error("Le jeu de données est trop petit pour être divisé en entraînement/test.")
    st.stop()
    
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==============================
# Modèle LSTM avec Keras Tuner
# ==============================
def build_lstm_model(hp):
    model = keras.Sequential()
    model.add(layers.LSTM(
        units=hp.Int('units', min_value=32, max_value=256, step=32),
        return_sequences=False,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(layers.Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [1e-4, 1e-3, 1e-2])
        ),
        loss='mse'
    )
    return model

# ==============================
# Optimisation + Entraînement
# ==============================
if st.button("🚀 Optimiser et entraîner le modèle LSTM"):
    # Sauvegarde du Scaler et des noms pour l'état de session
    st.session_state['scaler'] = scaler 
    st.session_state['target_col'] = TARGET_COL_NAME
    st.session_state['selected_col_name'] = selected_column_name_display

    safe_sector = re.sub(r'[^a-zA-Z0-9_]', '_', sector)
    safe_selected_column = re.sub(r'[^a-zA-Z0-9_]', '_', selected_column_name_display)

    with st.spinner("Optimisation en cours..."):
        
        tuner = kt.RandomSearch(
            build_lstm_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='lstm_tuning',
            project_name=f'forecast_{safe_sector}_{symbol}_{safe_selected_column}' 
        )

        tuner.search(X_train, y_train, epochs=20, validation_split=0.2, verbose=0)
        
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        # Sauvegarde dans l'état de session
        st.session_state['best_model'] = best_model
        st.session_state['best_hp'] = best_hp

    st.success("✅ Optimisation et Entraînement terminés")
    
    # Récupération des données pour l'évaluation
    preds = st.session_state['best_model'].predict(X_test, verbose=0)
    preds_rescaled = st.session_state['scaler'].inverse_transform(preds)
    y_test_rescaled = st.session_state['scaler'].inverse_transform(y_test.reshape(-1, 1))

    # Métriques
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, preds_rescaled))
    mae = mean_absolute_error(y_test_rescaled, preds_rescaled)
    
    # ------------------------------------
    # DÉTAILS DU MODÈLE ET ÉVALUATION
    # ------------------------------------
    st.subheader("🔬 Détails du Modèle LSTM Optimal")
    st.write(f"**Architecture :** 1 couche LSTM, 1 couche Dense (Output).")
    st.write(f"**Unités LSTM :** {st.session_state['best_hp'].get('units')} (paramètre clé de capacité).")
    st.write(f"**Taux de Dropout :** {st.session_state['best_hp'].get('dropout')}")
    st.write(f"**Learning Rate (Optimiseur Adam) :** {st.session_state['best_hp'].get('learning_rate')}")

    st.subheader("📏 Métriques d'Erreur (Jeu de Test, Dénormalisé)")
    st.metric(label="RMSE (Root Mean Squared Error)", value=f"{rmse:.4f}")
    st.metric(label="MAE (Mean Absolute Error)", value=f"{mae:.4f}")
    st.write("*Ces valeurs sont dans l'unité d'origine de la variable sélectionnée.*")
    
    # --- GRAPHE INTERACTIF ---
    results_df = pd.DataFrame({
        "Date": df.index[split + look_back:],
        "Réel": y_test_rescaled.flatten(),
        "Prédit": preds_rescaled.flatten()
    })

    fig = go.Figure()
    # Réel
    fig.add_trace(go.Scatter(
        x=results_df["Date"], 
        y=results_df["Réel"], 
        mode="lines", 
        name="Valeurs Réelles",
        hovertemplate="<b>Date:</b> %{x}<br><b>Réel:</b> %{y:.4f}<extra></extra>"
    ))
    # Prédit
    fig.add_trace(go.Scatter(
        x=results_df["Date"], 
        y=results_df["Prédit"], 
        mode="lines", 
        name="Prédictions",
        hovertemplate="<b>Date:</b> %{x}<br><b>Prédit:</b> %{y:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Comparaison Réel vs. Prédit sur le Jeu de Test ({selected_column_name_display})",
        xaxis_title="Date",
        yaxis_title=f"Valeur de {selected_column_name_display}",
        hovermode="x unified" 
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================
# Prédiction future (conditionnel)
# ==============================
if 'best_model' in st.session_state and st.session_state['best_model'] is not None:
    st.subheader(f"🔮 Projection future ({horizon_choice})")

    best_model = st.session_state['best_model']
    scaler = st.session_state['scaler']
    target_column_name = st.session_state['target_col']
    selected_column_name_display = st.session_state['selected_col_name'] 

    try:
        scaled_data = scaler.transform(df[[target_column_name]])
    except Exception as e:
        st.error(f"Erreur lors de la mise à l'échelle des données historiques : {e}")
        st.stop()
        
    last_seq = scaled_data[-look_back:] 
    predictions = []
    seq = last_seq.reshape(1, look_back, 1)

    with st.spinner(f"Génération de la prédiction pour les {future_days} prochains jours..."):
        for _ in range(future_days):
            pred = best_model.predict(seq, verbose=0)
            predictions.append(pred[0, 0])
            
            new_pred_reshaped = pred[0].reshape(1, 1, 1)
            seq = np.concatenate([seq[:, 1:, :], new_pred_reshaped], axis=1)

    predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Génération des dates futures
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date, periods=future_days + 1, freq="D")[1:] 

    future_df = pd.DataFrame({
        "Date": future_dates,
        f"Prévision {selected_column_name_display}": predictions_rescaled.flatten()
    }).set_index("Date")

    st.dataframe(future_df.head(20))

    fig_future = go.Figure()
    # Historique
    fig_future.add_trace(go.Scatter(
        x=df.index, 
        y=df[target_column_name], 
        mode="lines", 
        name="Historique", 
        line=dict(color='blue'),
        hovertemplate="<b>Date:</b> %{x}<br><b>Historique:</b> %{y:.4f}<extra></extra>"
    ))
    # Prévision
    fig_future.add_trace(go.Scatter(
        x=future_df.index, 
        y=future_df.iloc[:, 0], 
        mode="lines", 
        name="Prévision future", 
        line=dict(color='red', dash='dash'),
        hovertemplate="<b>Date:</b> %{x}<br><b>Prévision:</b> %{y:.4f}<extra></extra>"
    ))
    
    # Ajout d'une ligne verticale pour séparer l'historique de la prévision
    # CORRECTION APPLIQUÉE ICI: Retrait de l'annotation_text
    fig_future.add_vline(x=df.index[-1], line_width=2, line_dash="dash", line_color="green")
    
    fig_future.update_layout(
        title=f"Projection de {symbol_name_display} - {selected_column_name_display} ({horizon_choice})",
        xaxis_title="Date",
        yaxis_title=f"Valeur ({selected_column_name_display})",
        hovermode="x unified"
    )
    st.plotly_chart(fig_future, use_container_width=True)

    # ==============================
    # Téléchargement CSV
    # ==============================
    csv = future_df.to_csv().encode('utf-8')
    safe_sector = re.sub(r'[^a-zA-Z0-9_]', '_', sector)
    safe_selected_column = re.sub(r'[^a-zA-Z0-9_]', '_', selected_column_name_display)
    
    st.download_button(
        label="📥 Télécharger les prévisions (CSV)",
        data=csv,
        file_name=f"forecast_{safe_sector}_{symbol_name_display.split(' ')[-1]}_{safe_selected_column}.csv",
        mime="text/csv"
    )
