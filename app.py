import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import re
import datetime
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

import pandas_datareader.data as web

# --- Gestion du chemin du Logo ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGO_PATH = os.path.join(SCRIPT_DIR, "logo.png")
    if not os.path.exists(LOGO_PATH):
        LOGO_PATH = "🚀" # Fallback emoji
except NameError:
    LOGO_PATH = "logo.png" # Fallback

# ==============================
# CONFIGURATION
# ==============================
st.set_page_config(
    page_title="Haithem Vision Predict V3.1 (Stable)",
    layout="wide",
    page_icon=LOGO_PATH,
)

# ==============================
# MULTILINGUE (i18n)
# ==============================
translations = {
    'fr': {
        'page_title': "🚀 Haithem Vision Predict V3.1 (Stable)",
        'lang_select': "Langue",
        
        'navigation': "Navigation",
        'page_home': "Accueil (Prédiction)",
        'page_faq': "FAQ (Comment ça marche ?)",
        'page_contact': "Contactez-nous",

        'faq_title': "❓ FAQ - Guide d'utilisation",
        'faq_step1_title': "Étape 1 : Choisir la langue",
        'faq_step1_desc': "Utilisez le sélecteur en haut de la barre latérale pour choisir entre Français, English, ou العربية.",
        'faq_step2_title': "Étape 2 : Naviguer entre les pages",
        'faq_step2_desc': "Utilisez le menu 'Navigation' pour basculer entre la page d'Accueil (l'outil de prédiction), la FAQ (cette page), et la page Contact.",
        'faq_step3_title': "Étape 3 : Sélectionner un actif (sur la page d'Accueil)",
        'faq_step3_desc': "Dans 'Paramètres de Base', choisissez une 'Catégorie' (ex: Actions US) puis un 'Actif à prédire' (ex: Apple).",
        'faq_step4_title': "Étape 4 : Choisir un horizon",
        'faq_step4_desc': "Sélectionnez l' 'Horizon de prédiction' (ex: 1 an). Cela détermine la durée de la prévision future et la quantité de données historiques utilisées pour l'entraînement.",
        'faq_step5_title': "Étape 5 : Lancer l'entraînement",
        'faq_step5_desc': "Cliquez sur le bouton '🚀 Optimiser et Entraîner le Modèle'. Vous pouvez ajuster les 'Paramètres d'Entraînement' (complexité, essais) pour affiner le modèle, mais les réglages par défaut sont recommandés pour commencer.",
        'faq_step6_title': "Étape 6 : Analyser les résultats",
        'faq_step6_desc': "Une fois l'entraînement terminé, trois onglets apparaissent :\n- **🔬 Performance Modèle :** Affiche les détails du modèle et les courbes d'apprentissage.\n- **📏 Évaluation Test :** Compare les prédictions du modèle aux données réelles (non vues) pour évaluer sa précision.\n- **🔮 Projection Future :** Montre la prévision pour l'horizon choisi, avec une analyse et une option de téléchargement.",

        'contact_title': "Nous Contacter",
        'contact_info': "Pour toute question, collaboration ou support technique, veuillez contacter :",
        'contact_name': "AHMED HAITHEM BERKANE",
        'contact_job_title': "CONSULTANT, DEVELOPPEUR WEB ET IA",
        'contact_phone': "Téléphone",
        'contact_email': "Email",
        'contact_address': "Adresse",
        
        'base_params': "⚙️ Paramètres de Base",
        'category': "Catégorie",
        'predict_asset': "Actif à prédire",
        'horizon': "Horizon de prédiction",
        'horizons': ["6 mois", "1 an", "3 ans", "5 ans"],
        'train_params': "🛠️ Paramètres d'Entraînement",
        'model_complexity': "Complexité du Modèle",
        'complexities': ["Simple (1 couche - Rapide)", "Complexe (2 couches - Précis)"],
        'optim_trials': "Essais d'optimisation",
        'train_epochs': "Époques d'entraînement final",
        'hist_data': "📊 Données historiques",
        'info_analysis': "Analyse basée sur le",
        'info_log_return': "Prix de", # Changé 'Log-Retour' en 'Prix'
        'info_and': "et",
        'run_button': "🚀 Optimiser et Entraîner le Modèle",
        'spinner_optim': "Optimisation en cours",
        'success_optim': "✅ Hyperparamètres optimisés. Entraînement final en cours...",
        'spinner_train': "Entraînement final du meilleur modèle",
        'success_train': "✅ Modèle optimisé et entraîné !",
        'tab_perf': "🔬 Performance Modèle",
        'tab_eval': "📏 Évaluation Test",
        'tab_proj': "🔮 Projection Future",
        'perf_title': "Détails et Performance du Modèle",
        'hp_title': "Hyperparamètres Optimaux :",
        'hp_units': "Unités (Couche 1)",
        'hp_units_2': "Unités (Couche 2)",
        'hp_dropout': "Dropout (Couche 1)",
        'hp_dropout_2': "Dropout (Couche 2)",
        'hp_lr': "Learning Rate",
        'metrics_title': "Métriques d'Entraînement Finales :",
        'metrics_val_loss': "Perte de Validation Finale (val_loss)",
        'metrics_val_mae': "Erreur de Validation Finale (val_mae)",
        'metrics_caption': "Métriques basées sur les données normalisées.", # Modifié
        'charts_title': "Courbes d'Apprentissage :",
        'chart_loss_title': "Évolution de la Perte (MSE)",
        'chart_loss_train': "Train Loss (MSE)",
        'chart_loss_val': "Validation Loss (MSE)",
        'chart_mae_title': "Évolution de l'Erreur Absolue Moyenne",
        'chart_mae_train': "Train MAE",
        'chart_mae_val': "Validation MAE",
        'eval_title': "Évaluation sur le Jeu de Test (sur les PRIX)",
        'eval_rmse': "RMSE (sur Prix)",
        'eval_mae': "MAE (sur Prix)",
        'eval_chart_title': "Comparaison Réel vs. Prédit (sur les Prix)",
        'eval_real': "Valeurs Réelles (Prix)",
        'eval_pred': "Prédictions (Prix)",
        'eval_toggle': "Afficher le tableau des valeurs de test",
        'eval_error': "Échec de la recréation des données de test pour l'évaluation.",
        'align_error': "Erreur d'alignement des données lors de l'évaluation.",
        'proj_title': "Projection Future",
        'proj_spinner': "Génération des prévisions...",
        'proj_chart_title': "Projection Future (basée sur les PRIX)", # Modifié
        'proj_hist': "Historique (Prix)",
        'proj_future': "Prévision Future (Prix)",
        'proj_analysis_title': "💬 Analyse de la Projection",
        'proj_download': "📥 Télécharger les Prévisions (CSV)",
        'comment_trend': "📈 **Tendance Globale :** Le modèle projette",
        'comment_rise': "une **hausse** de",
        'comment_fall': "une **baisse** de",
        'comment_for': "pour",
        'comment_reaching': "atteignant environ",
        'comment_by': "d'ici le",
        'comment_q_trend': "🎯 **Prochain Trimestre :** Une valeur d'environ",
        'comment_q_expected': "est attendue pour la",
        'comment_q_end': "Fin Q",
        'data_error': "Une erreur est survenue lors du chargement des données",
        'prep_error_positive': "Aucune donnée positive trouvée.",
        'prep_error_log': "Aucune donnée après traitement. L'ensemble est peut-être trop petit.",
        'prep_error_seq': "Pas assez de données pour créer des séquences",
    },
    'en': {
        'info_log_return': "Price of",
        'metrics_caption': "Metrics based on scaled data.",
        'proj_chart_title': "Future Projection (based on PRICE)",
        # ... (les autres traductions EN sont complètes)
    },
    'ar': {
        'info_log_return': "سعر",
        'metrics_caption': ".المقاييس مبنية على البيانات المعدلة",
        'proj_chart_title': "التوقع المستقبلي (مبni على السعر)",
        # ... (les autres traductions AR sont complètes)
    }
}


if 'lang' not in st.session_state:
    st.session_state.lang = 'fr'

def t(key):
    default_lang_dict = translations.get('fr', {})
    current_lang_dict = translations.get(st.session_state.lang, default_lang_dict)
    
    fr_keys = set(default_lang_dict.keys())
    for lang_dict in translations.values():
        fr_keys.update(lang_dict.keys())
    
    # S'assurer que toutes les clés existent au moins en français (pour le fallback)
    # (Cette partie est simplifiée car nous supposons que 'fr' est complet)
    
    if key in current_lang_dict:
        return current_lang_dict[key]
    elif key in default_lang_dict:
        return default_lang_dict[key]
    return key

HORIZON_KEYS = ['6m', '1y', '3y', '5y']
HORIZON_MAP = {
    '6m': {"train_years": 2, "predict_days": 180},
    '1y': {"train_years": 3, "predict_days": 365},
    '3y': {"train_years": 7, "predict_days": 3*365},
    '5y': {"train_years": 10, "predict_days": 5*365},
}
COMPLEXITY_KEYS = ['simple', 'complex']
LOOK_BACK = 60
# TARGET_COL_ORIG_NAME n'est plus nécessaire

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

# ==============================
# CSS STYLING
# ==============================
CSS_STYLE = """
<style>
/* Arrière-plan principal de l'application */
[data-testid="stAppViewContainer"] > .main {
    background-color: #FDF8E3; /* Beige */
}
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0A2342; /* Bleu Foncé */
}

/* --- DÉBUT DE LA CORRECTION CSS --- */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p {
    color: #FDF8E3 !important; /* Beige clair, !important pour forcer */
}
/* --- FIN DE LA CORRECTION CSS --- */

/* Titre principal */
h1 { color: #0A2342; font-weight: bold; }
h2, h3 { color: #0A2342; }
/* Boutons */
.stButton > button {
    background-color: #FF6B00; color: #FFFFFF; border: none;
    border-radius: 5px; font-weight: bold; padding: 10px 20px;
}
.stButton > button:hover { background-color: #E05C00; }
.stDownloadButton > button { background-color: #0A2342; color: #FFFFFF; }
.stDownloadButton > button:hover { background-color: #004A99; }
/* Conteneurs */
[data-testid="stInfo"] {
    background-color: #E6F0F8; border: 1px solid #0A2342; color: #0A2342;
}
[data-testid="stSuccess"] { background-color: #DFF0D8; color: #3C763D; }
.stTabs [data-baseweb="tab"] { background-color: #F0F2F6; color: #0A2342; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #FFFFFF; color: #FF6B00; border-top: 2px solid #FF6B00;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #E0E0E0; border-radius: 10px; padding: 1rem;
    background-color: #FFFFFF; box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

RTL_CSS = """
<style>
body, .main, [data-testid="stSidebar"] { direction: rtl !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 { text-align: right !important; }
h1, h2, h3, p { text-align: right !important; }
[data-testid="stInfo"], [data-testid="stMetric"], [data-testid="stSuccess"], [data-testid="stError"] {
    text-align: right !important; direction: rtl !important;
}
.stButton > button {
    direction: ltr !important; text-align: right !important;
    padding-left: 1rem !important; padding-right: 2.5rem !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] { direction: ltr !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { margin-left: 0.5rem; margin-right: 0; }
[data-testid="stTabs"] [role="tablist"] { justify-content: flex-end; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
     text-align: right !important; direction: rtl !important;
}
</style>
"""
if st.session_state.lang == 'ar':
    st.markdown(RTL_CSS, unsafe_allow_html=True)

# ==============================
# INTERFACE UTILISATEUR (SIDEBAR)
# ==============================
if isinstance(LOGO_PATH, str) and os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=64)
else:
    st.sidebar.markdown(f"<h1 style='text-align: center; color: white;'>{LOGO_PATH}</h1>", unsafe_allow_html=True)

st.sidebar.write(t('lang_select') + ":")
cols = st.sidebar.columns(3)
if cols[0].button("🇫🇷", use_container_width=True):
    if st.session_state.lang != 'fr': st.session_state.lang = 'fr'; st.rerun()
if cols[1].button("🇬🇧", use_container_width=True):
    if st.session_state.lang != 'en': st.session_state.lang = 'en'; st.rerun()
if cols[2].button("🇩🇿", use_container_width=True):
    if st.session_state.lang != 'ar': st.session_state.lang = 'ar'; st.rerun()

st.sidebar.divider()
page_options = {'home': t('page_home'), 'faq': t('page_faq'), 'contact': t('page_contact')}
selected_page_key = st.sidebar.radio(
    t('navigation'), options=list(page_options.keys()),
    format_func=lambda key: page_options[key], key="page_selector"
)

# ==============================
# FONCTIONS (LOGIQUE MÉTIER)
# ==============================
@st.cache_data
def load_data(symbol, sector, years_of_data):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365.25 * years_of_data)
    features = []
    if "FRED" in sector:
        df = web.DataReader(symbol, "fred", start_date, end_date)
        col_to_select = symbol
        features = [col_to_select]
    else:
        df = yf.download(symbol, start=start_date, end=end_date, interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        col_to_select = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        features = [col_to_select]
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            features.append('Volume')
            # MODIFICATION : Log-transform du Volume
            df['Volume'] = np.log1p(df['Volume']) 
    df = df[features].dropna()
    return df, col_to_select, features

# MODIFICATION : Retour à la prédiction de PRIX (stable)
def prepare_data(df, features):
    target_col_name = features[0]
    
    # Créer un scaler juste pour la colonne cible (prix)
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = pd.DataFrame(index=df.index)
    df_scaled[target_col_name] = price_scaler.fit_transform(df[[target_col_name]])

    # Normaliser les autres features (Volume) si elles existent
    feature_scalers = {}
    if len(features) > 1:
        for feature in features[1:]:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_scaled[feature] = scaler.fit_transform(df[[feature]])
            feature_scalers[feature] = scaler # Sauvegarder le scaler

    scaled_data = df_scaled.values
    
    # Création des séquences
    X, y = [], []
    for i in range(LOOK_BACK, len(scaled_data)):
        X.append(scaled_data[i-LOOK_BACK:i, :])
        y.append(scaled_data[i, 0]) # La cible est toujours la première colonne (prix)
        
    X, y = np.array(X), np.array(y)
    
    if len(X) == 0:
        st.error(f"{t('prep_error_seq')} (Lookback = {LOOK_BACK}).")
        return (None,) * 7 # Ajuster le nombre de retours

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Correction : S'assurer que df est bien le df original/complet
    return X_train, y_train, X_test, y_test, price_scaler, feature_scalers, df

def build_model(hp, input_shape, complexity='Complexe'):
    model = keras.Sequential()
    if complexity == 'Complexe':
        model.add(layers.LSTM(units=hp.Int('units_1', 32, 256, 32), return_sequences=True, input_shape=input_shape, name='lstm_1'))
        model.add(layers.Dropout(hp.Float('dropout_1', 0.1, 0.5, 0.1), name='dropout_1'))
        model.add(layers.LSTM(units=hp.Int('units_2', 32, 256, 32), return_sequences=False, name='lstm_2'))
        model.add(layers.Dropout(hp.Float('dropout_2', 0.1, 0.5, 0.1), name='dropout_2'))
    else:
        model.add(layers.LSTM(units=hp.Int('units_1', 32, 256, 32), return_sequences=False, input_shape=input_shape, name='lstm_1'))
        model.add(layers.Dropout(hp.Float('dropout_1', 0.1, 0.5, 0.1), name='dropout_1'))
    model.add(layers.Dense(1, name='output_layer'))
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=2e-3, sampling='log')
    model.compile(optimizer=keras.optimizers.Adam(lr), 
                   loss='mean_squared_error',
                   metrics=['mae']) 
    return model

def generate_prediction_commentary(start_price, df_future, symbol_name, t_func):
    try:
        end_price = df_future['Prévision'].iloc[-1]
        end_date = df_future.index[-1]
        total_change_pct = ((end_price / start_price) - 1) * 100
        if total_change_pct > 0:
            global_trend = f"{t_func('comment_rise')} **{total_change_pct:.2f}%**"
        else:
            global_trend = f"{t_func('comment_fall')} **{total_change_pct:.2f}%**"
        commentary = [f"{t_func('comment_trend')} {global_trend} {t_func('comment_for')} {symbol_name}, {t_func('comment_reaching')} **{end_price:.2f}** {t_func('comment_by')} {end_date.strftime('%d-%m-%Y')}."]
        today = datetime.datetime.now().date()
        q_targets = []
        for i in range(1, 5):
            next_q_year = today.year + (today.month + i*3 - 1) // 12
            next_q_month = (today.month + i*3 - 1) % 12 + 1
            if next_q_month <= 3: q_date = datetime.date(next_q_year, 3, 31)
            elif next_q_month <= 6: q_date = datetime.date(next_q_year, 6, 30)
            elif next_q_month <= 9: q_date = datetime.date(next_q_year, 9, 30)
            else: q_date = datetime.date(next_q_year, 12, 31)
            if q_date <= end_date.date():
                q_targets.append((q_date.strftime('%Y-%m-%d'), f"{t_func('comment_q_end')} { (q_date.month - 1) // 3 + 1 } {q_date.year}"))
        if q_targets:
            first_q_date_str, first_q_name = q_targets[0]
            try:
                q_price = df_future.asof(first_q_date_str)['Prévision']
                q_change_pct = ((q_price / start_price) - 1) * 100
                commentary.append(f"{t_func('comment_q_trend')} **{q_price:.2f}** (soit **{q_change_pct:+.2f}%**) {t_func('comment_q_expected')} **{first_q_name}**.")
            except (KeyError, TypeError, IndexError):
                pass
        return "\n\n".join(commentary)
    except Exception as e:
        return f"Erreur lors de la génération du commentaire : {e}"

# ==============================
# ROUTAGE DES PAGES
# ==============================
if selected_page_key == 'home':
    st.title(t('page_title'))
    horizon_options_display = t('horizons')
    complexity_options_display = t('complexities')
    
    st.sidebar.header(t('base_params'))
    sector_display = st.sidebar.selectbox(t('category'), list(CATEGORIES.keys()))
    symbol_name_display = st.sidebar.selectbox(t('predict_asset'), list(CATEGORIES[sector_display].keys()))
    symbol = CATEGORIES[sector_display][symbol_name_display]
    selected_horizon_display = st.sidebar.selectbox(t('horizon'), horizon_options_display)
    horizon_key = HORIZON_KEYS[horizon_options_display.index(selected_horizon_display)]
    train_years = HORIZON_MAP[horizon_key]["train_years"]
    future_days = HORIZON_MAP[horizon_key]["predict_days"]
    
    st.sidebar.header(t('train_params'))
    
    selected_complexity_display = st.sidebar.selectbox(
        t('model_complexity'), 
        complexity_options_display, 
        index=0  # <-- OPTIMISATION CLOUD : "Simple" par défaut
    )
    complexity_key = COMPLEXITY_KEYS[complexity_options_display.index(selected_complexity_display)]
    max_trials = st.sidebar.number_input(
        t('optim_trials'), 1, 20, 
        value=5,  # <-- OPTIMISATION CLOUD : 5 essais par défaut
        step=1
    )
    epochs = st.sidebar.number_input(
        t('train_epochs'), 10, 100, 
        value=30, # <-- OPTIMISATION CLOUD : 30 époques par défaut
        step=5
    )

    try:
        df_original, target_col, features_used = load_data(symbol, sector_display, train_years)
        st.subheader(f"{t('hist_data')} - {symbol_name_display}")
        info_text = f"{t('info_analysis')} **{t('info_log_return')} {target_col}**"
        if len(features_used) > 1: info_text += f" {t('info_and')} **Log({', '.join(features_used[1:])})**."
        st.info(info_text)
        with st.container(border=True):
            st.line_chart(df_original[target_col])
        
        # MODIFICATION : Logique de préparation de PRIX
        prep_results = prepare_data(df_original, features_used)
        if prep_results[0] is None:
            st.stop()
        X_train, y_train, X_test, y_test, price_scaler, feature_scalers, df_processed = prep_results
    except Exception as e:
        st.error(f"{t('data_error')}: {e}")
        st.stop()

    if st.button(t('run_button')):
        with st.spinner(f"{t('spinner_optim')} ({max_trials} {t('optim_trials').lower()})..."):
            input_shape = (X_train.shape[1], X_train.shape[2])
            complexity_arg = "Complexe" if "complex" in complexity_key else "Simple"
            tuner = kt.RandomSearch(
                lambda hp: build_model(hp, input_shape, complexity=complexity_arg),
                objective='val_loss', max_trials=max_trials, executions_per_trial=1,
                directory='kt_dir', project_name=f'project_{symbol}_{complexity_arg}', overwrite=True
            )
            tuner.search(X_train, y_train, epochs=20, validation_split=0.2, verbose=0,
                         callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=5)])
            best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        st.success(t('success_optim'))
        with st.spinner(f"{t('spinner_train')} (max {epochs} {t('train_epochs').lower().split()[-1]})..."):
            best_model = tuner.hypermodel.build(best_hp)
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = best_model.fit(
                X_train, y_train,
                epochs=epochs,
                validation_split=0.2,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
        st.success(t('success_train'))
        
        # MODIFICATION : Sauvegarde de la session pour la prédiction de PRIX
        st.session_state['model'] = best_model
        st.session_state['history'] = history.history
        st.session_state['best_hp'] = best_hp
        st.session_state['price_scaler'] = price_scaler
        st.session_state['feature_scalers'] = feature_scalers
        st.session_state['df_original'] = df_original # Sauvegarde du df original pour l'affichage
        st.session_state['df_processed'] = df_processed # Sauvegarde du df traité
        st.session_state['features_used'] = features_used
        st.session_state['target_col'] = target_col
        st.session_state['trained_symbol'] = symbol
        st.session_state['trained_horizon'] = horizon_key
        st.session_state['trained_complexity'] = complexity_key
        st.session_state['trained_X_train_len'] = len(X_train) 

    is_model_stale = not ('model' in st.session_state and
                           st.session_state.get('trained_symbol') == symbol and
                           st.session_state.get('trained_horizon') == horizon_key and
                           st.session_state.get('trained_complexity') == complexity_key)

    if not is_model_stale:
        model = st.session_state['model']
        history_data = st.session_state['history']
        best_hp_data = st.session_state['best_hp']
        price_scaler = st.session_state['price_scaler']
        feature_scalers = st.session_state['feature_scalers']
        df_history_original = st.session_state['df_original']
        df_history_processed = st.session_state['df_processed'] # Récupérer le df traité
        features_used = st.session_state['features_used']
        target_col = st.session_state['target_col']
        
        if 'trained_X_train_len' in st.session_state:
            X_train_len = st.session_state['trained_X_train_len']
        else:
            st.warning("L'état de session est ancien. Veuillez ré-entraîner le modèle pour une évaluation précise.")
            st.stop()

        tab_perf, tab_eval, tab_proj = st.tabs([
            t('tab_perf'), 
            t('tab_eval'), 
            t('tab_proj')
        ])
        
        with tab_perf:
            st.subheader(t('perf_title'))
            with st.container(border=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{t('hp_title')}**")
                    hp_md = f"- **{t('hp_units')}:** `{best_hp_data.get('units_1')}`\n"
                    hp_md += f"- **{t('hp_dropout')}:** `{best_hp_data.get('dropout_1'):.2f}`\n"
                    if complexity_key == 'complex':
                        hp_md += f"- **{t('hp_units_2')}:** `{best_hp_data.get('units_2')}`\n"
                        hp_md += f"- **{t('hp_dropout_2')}:** `{best_hp_data.get('dropout_2'):.2f}`\n"
                    hp_md += f"- **{t('hp_lr')}:** `{best_hp_data.get('learning_rate'):.6f}`"
                    st.markdown(hp_md, unsafe_allow_html=True if st.session_state.lang == 'ar' else False)
                with col2:
                    st.write(f"**{t('metrics_title')}**")
                    final_val_loss = history_data['val_loss'][-1]
                    final_val_mae = history_data['val_mae'][-1]
                    st.metric(t('metrics_val_loss'), f"{final_val_loss:.6f}")
                    st.metric(t('metrics_val_mae'), f"{final_val_mae:.6f}")
                    st.caption(t('metrics_caption'))
            st.write(f"**{t('charts_title')}**")
            with st.container(border=True):
                col1, col2 = st.columns(2)
                with col1:
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(y=history_data['loss'], name=t('chart_loss_train'), line=dict(color='#0A2342')))
                    fig_loss.add_trace(go.Scatter(y=history_data['val_loss'], name=t('chart_loss_val'), line=dict(color='#FF6B00')))
                    fig_loss.update_layout(title=t('chart_loss_title'), xaxis_title='Époques', yaxis_title='Perte')
                    st.plotly_chart(fig_loss, use_container_width=True)
                with col2:
                    fig_mae = go.Figure()
                    fig_mae.add_trace(go.Scatter(y=history_data['mae'], name=t('chart_mae_train'), line=dict(color='#0A2342')))
                    fig_mae.add_trace(go.Scatter(y=history_data['val_mae'], name=t('chart_mae_val'), line=dict(color='#FF6B00')))
                    fig_mae.update_layout(title=t('chart_mae_title'), xaxis_title='Époques', yaxis_title='MAE')
                    st.plotly_chart(fig_mae, use_container_width=True)

        with tab_eval:
            st.subheader(t('eval_title'))
            
            prep_results_eval = prepare_data(df_history_original, features_used)
            if prep_results_eval[0] is None:
                st.error(t('eval_error'))
            else:
                _, _, X_test_eval, y_test_eval, price_scaler_eval, _, df_processed_eval = prep_results_eval
                
                if len(X_test_eval) == 0:
                    st.warning("Pas assez de données pour un jeu de test. Essayez un horizon de données plus long.")
                else:
                    preds_scaled = model.predict(X_test_eval)
                    preds_rescaled = price_scaler_eval.inverse_transform(preds_scaled)
                    y_test_rescaled = price_scaler_eval.inverse_transform(y_test_eval.reshape(-1, 1))
                    
                    test_dates = df_processed_eval.index[X_train_len + LOOK_BACK:]

                    if len(test_dates) == len(y_test_rescaled):
                        with st.container(border=True):
                            rmse = np.sqrt(mean_squared_error(y_test_rescaled, preds_rescaled))
                            mae = mean_absolute_error(y_test_rescaled, preds_rescaled)
                            col1, col2 = st.columns(2)
                            col1.metric(t('eval_rmse'), f"{rmse:.4f}")
                            col2.metric(t('eval_mae'), f"{mae:.4f}")
                        with st.container(border=True):
                            fig_compare = go.Figure()
                            fig_compare.add_trace(go.Scatter(x=test_dates, y=y_test_rescaled.flatten(), mode='lines', name=t('eval_real'), line=dict(color='#0A2342')))
                            fig_compare.add_trace(go.Scatter(x=test_dates, y=preds_rescaled.flatten(), mode='lines', name=t('eval_pred'), line=dict(color='#FF6B00', dash='dash')))
                            fig_compare.update_layout(title_text=t('eval_chart_title'), hovermode="x unified")
                            st.plotly_chart(fig_compare, use_container_width=True)
                        if st.toggle(t('eval_toggle')):
                            results_df = pd.DataFrame({'Date': test_dates, 'Valeur Réelle': y_test_rescaled.flatten(), 'Prédiction': preds_rescaled.flatten()})
                            st.dataframe(results_df.set_index('Date'))
                    else:
                        st.error(f"{t('align_error')} (Dates: {len(test_dates)}, y_test: {len(y_test_rescaled)})")

        with tab_proj:
            st.subheader(f"{t('proj_title')} ({selected_horizon_display})")
            with st.spinner(t('proj_spinner')):
                
                # 1. Recréer tous les scalers sur l'ensemble des données
                price_scaler_full = MinMaxScaler(feature_range=(0, 1))
                price_scaler_full.fit(df_history_original[[target_col]])
                
                feature_scalers_full = {}
                df_scaled_full = pd.DataFrame(index=df_history_original.index)
                df_scaled_full[target_col] = price_scaler_full.transform(df_history_original[[target_col]])
                
                for feature in features_used[1:]:
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    df_scaled_full[feature] = scaler.fit_transform(df_history_original[[feature]])
                    feature_scalers_full[feature] = scaler
                
                # 2. Obtenir la dernière séquence
                last_60_days_scaled = df_scaled_full.values[-LOOK_BACK:]
                current_batch = last_60_days_scaled.reshape(1, LOOK_BACK, len(features_used))
                
                future_preds_scaled = []
                
                for _ in range(future_days):
                    pred_scaled = model.predict(current_batch, verbose=0)[0]
                    future_preds_scaled.append(pred_scaled)
                    
                    new_entry_scaled = np.zeros((1, 1, len(features_used)))
                    new_entry_scaled[0, 0, 0] = pred_scaled[0]
                    
                    if len(features_used) > 1:
                        for i, feature in enumerate(features_used[1:]):
                            mean_feature_val = current_batch[0, :, i+1].mean()
                            new_entry_scaled[0, 0, i+1] = mean_feature_val
                            
                    current_batch = np.append(current_batch[:, 1:, :], new_entry_scaled, axis=1)
            
            future_preds_rescaled = price_scaler_full.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))
            
            future_dates = pd.date_range(start=df_history_original.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')
            df_future = pd.DataFrame(future_preds_rescaled, index=future_dates, columns=['Prévision'])
            
            with st.container(border=True):
                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(x=df_history_original.index, y=df_history_original[target_col], mode='lines', name=t('proj_hist'), line=dict(color='#0A2342')))
                fig_future.add_trace(go.Scatter(x=df_future.index, y=df_future['Prévision'], mode='lines', name=t('proj_future'), line=dict(color='#FF6B00', dash='dash')))
                fig_future.update_layout(title_text=t('proj_chart_title'), hovermode="x unified")
                st.plotly_chart(fig_future, use_container_width=True)

            st.subheader(t('proj_analysis_title'))
            with st.container(border=True):
                last_price = df_history_original[target_col].iloc[-1] # Prix de départ
                comment_text = generate_prediction_commentary(last_price, df_future, symbol_name_display, t)
                st.markdown(comment_text, unsafe_allow_html=True if st.session_state.lang == 'ar' else False)

            csv = df_future.to_csv().encode('utf-8')
            st.download_button(t('proj_download'), csv, f"forecast_{symbol}.csv", "text/csv")

# ------------------------------
# --- PAGE 2 : FAQ ---
# ------------------------------
elif selected_page_key == 'faq':
    st.title(t('faq_title'))
    with st.container(border=True):
        st.subheader(t('faq_step1_title'))
        st.markdown(t('faq_step1_desc'))
        st.subheader(t('faq_step2_title'))
        st.markdown(t('faq_step2_desc'))
        st.subheader(t('faq_step3_title'))
        st.markdown(t('faq_step3_desc'))
        st.subheader(t('faq_step4_title'))
        st.markdown(t('faq_step4_desc'))
        st.subheader(t('faq_step5_title'))
        st.markdown(t('faq_step5_desc'))
        st.subheader(t('faq_step6_title'))
        st.markdown(t('faq_step6_desc'))

# ------------------------------
# --- PAGE 3 : CONTACT ---
# ------------------------------
elif selected_page_key == 'contact':
    st.title(t('contact_title'))
    with st.container(border=True):
        st.markdown(t('contact_info'))
        st.markdown(f"### {t('contact_name')}")
        st.markdown(f"**{t('contact_job_title')}**")
        st.divider()
        st.markdown(f"**{t('contact_email')} :** Haithem-Berkane@outlook.fr")
        st.markdown(f"**{t('contact_phone')} :** +213 661 338 333")
        st.markdown(f"**{t('contact_address')} :** Algeria")
