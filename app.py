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

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGO_PATH = os.path.join(SCRIPT_DIR, "logo.png")
    if not os.path.exists(LOGO_PATH):
        LOGO_PATH = "🚀"
except NameError:
    LOGO_PATH = "logo.png"

# ==============================
# CONFIGURATION
# ==============================
st.set_page_config(
    page_title="Haithem Vision Predict V3.0",
    layout="wide",
    page_icon=LOGO_PATH,
)

# ==============================
# MULTILINGUE (i18n)
# ==============================
translations = {
    'fr': {
        'page_title': "🚀 Haithem Vision Predict V3.0",
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
        'info_log_return': "Log-Retour de",
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
        'metrics_caption': "Métriques basées sur les log-retours scalés.",
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
        'proj_chart_title': "Projection Future (basée sur Log-Retours)",
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
        'prep_error_positive': "Aucune donnée positive trouvée pour le calcul des log-retours.",
        'prep_error_log': "Aucune donnée après calcul des log-retours. L'ensemble est peut-être trop petit.",
        'prep_error_seq': "Pas assez de données pour créer des séquences",
    },
    'en': {
        'page_title': "🚀 Haithem Vision Predict V3.0",
        'lang_select': "Language",
        
        'navigation': "Navigation",
        'page_home': "Home (Prediction)",
        'page_faq': "FAQ (How it works)",
        'page_contact': "Contact Us",

        'faq_title': "❓ FAQ - User Guide",
        'faq_step1_title': "Step 1: Choose Language",
        'faq_step1_desc': "Use the selector at the top of the sidebar to choose between Français, English, or العربية.",
        'faq_step2_title': "Step 2: Navigate Pages",
        'faq_step2_desc': "Use the 'Navigation' menu to switch between the Home page (the prediction tool), the FAQ (this page), and the Contact page.",
        'faq_step3_title': "Step 3: Select an Asset (on Home page)",
        'faq_step3_desc': "Under 'Base Parameters', choose a 'Category' (e.g., US Stocks) and then an 'Asset to Predict' (e.g., Apple).",
        'faq_step4_title': "Step 4: Choose a Horizon",
        'faq_step4_desc': "Select the 'Prediction Horizon' (e.g., 1 year). This determines the length of the future forecast and the amount of historical data used for training.",
        'faq_step5_title': "Step 5: Run the Training",
        'faq_step5_desc': "Click the '🚀 Optimize and Train Model' button. You can adjust the 'Training Parameters' (complexity, trials) to fine-tune the model, but the default settings are recommended to start.",
        'faq_step6_title': "Step 6: Analyze the Results",
        'faq_step6_desc': "Once training is complete, three tabs will appear:\n- **🔬 Model Performance:** Shows model details and learning curves.\n- **📏 Test Evaluation:** Compares the model's predictions against real (unseen) data to evaluate its accuracy.\n- **🔮 Future Projection:** Displays the forecast for your chosen horizon, with analysis and a download option.",

        'contact_title': "Contact Us",
        'contact_info': "For any questions, collaboration, or technical support, please contact:",
        'contact_name': "AHMED HAITHEM BERKANE",
        'contact_job_title': "CONSULTANT, DEVELOPPEUR WEB ET IA",
        'contact_phone': "Phone",
        'contact_email': "Email",
        'contact_address': "Address",

        'base_params': "⚙️ Base Parameters",
        'category': "Category",
        'predict_asset': "Asset to Predict",
        'horizon': "Prediction Horizon",
        'horizons': ["6 months", "1 year", "3 years", "5 years"],
        'train_params': "🛠️ Training Parameters",
        'model_complexity': "Model Complexity",
        'complexities': ["Simple (1 layer - Fast)", "Complex (2 layers - Accurate)"],
        'optim_trials': "Optimization Trials",
        'train_epochs': "Final Training Epochs",
        'hist_data': "📊 Historical Data",
        'info_analysis': "Analysis based on",
        'info_log_return': "Log-Return of",
        'info_and': "and",
        'run_button': "🚀 Optimize and Train Model",
        'spinner_optim': "Optimizing",
        'success_optim': "✅ Hyperparameters optimized. Final training in progress...",
        'spinner_train': "Final training of the best model",
        'success_train': "✅ Model optimized and trained!",
        'tab_perf': "🔬 Model Performance",
        'tab_eval': "📏 Test Evaluation",
        'tab_proj': "🔮 Future Projection",
        'perf_title': "Model Details and Performance",
        'hp_title': "Optimal Hyperparameters:",
        'hp_units': "Units (Layer 1)",
        'hp_units_2': "Units (Layer 2)",
        'hp_dropout': "Dropout (Layer 1)",
        'hp_dropout_2': "Dropout (Layer 2)",
        'hp_lr': "Learning Rate",
        'metrics_title': "Final Training Metrics:",
        'metrics_val_loss': "Final Validation Loss (val_loss)",
        'metrics_val_mae': "Final Validation Error (val_mae)",
        'metrics_caption': "Metrics based on scaled log-returns.",
        'charts_title': "Learning Curves:",
        'chart_loss_title': "Loss (MSE) Evolution",
        'chart_loss_train': "Train Loss (MSE)",
        'chart_loss_val': "Validation Loss (MSE)",
        'chart_mae_title': "Mean Absolute Error Evolution",
        'chart_mae_train': "Train MAE",
        'chart_mae_val': "Validation MAE",
        'eval_title': "Evaluation on Test Set (on PRICES)",
        'eval_rmse': "RMSE (on Price)",
        'eval_mae': "MAE (on Price)",
        'eval_chart_title': "Real vs. Predicted (on Prices)",
        'eval_real': "Real Values (Price)",
        'eval_pred': "Predictions (Price)",
        'eval_toggle': "Show test values table",
        'eval_error': "Failed to recreate test data for evaluation.",
        'align_error': "Data alignment error during evaluation.",
        'proj_title': "Future Projection",
        'proj_spinner': "Generating predictions...",
        'proj_chart_title': "Future Projection (based on Log-Returns)",
        'proj_hist': "History (Price)",
        'proj_future': "Future Prediction (Price)",
        'proj_analysis_title': "💬 Projection Analysis",
        'proj_download': "📥 Download Predictions (CSV)",
        'comment_trend': "📈 **Global Trend:** The model projects a",
        'comment_rise': "rise of",
        'comment_fall': "fall of",
        'comment_for': "for",
        'comment_reaching': "reaching approx.",
        'comment_by': "by",
        'comment_q_trend': "🎯 **Next Quarter:** A value of approx.",
        'comment_q_expected': "is expected for",
        'comment_q_end': "End Q",
        'data_error': "An error occurred while loading data",
        'prep_error_positive': "No positive data found for log-return calculation.",
        'prep_error_log': "No data after log-return calculation. Dataset might be too small.",
        'prep_error_seq': "Not enough data to create sequences",
    },
    'ar': {
        'page_title': "🚀 Haithem Vision Predict V3.0",
        'lang_select': "اللغة",

        'navigation': "التنقل",
        'page_home': "الرئيسية (التنبؤ)",
        'page_faq': "الأسئلة الشائعة (كيف يعمل)",
        'page_contact': "اتصل بنا",

        'faq_title': "❓ الأسئلة الشائعة - دليل المستخدم",
        'faq_step1_title': "الخطوة 1: اختيار اللغة",
        'faq_step1_desc': "استخدم المحدد في أعلى الشريط الجانبي للاختيار بين الفرنسية، الإنجليزية، أو العربية.",
        'faq_step2_title': "الخطوة 2: التنقل بين الصفحات",
        'faq_step2_desc': "استخدم قائمة 'التنقل' للتبديل بين الصفحة الرئيسية (أداة التنبؤ)، صفحة الأسئلة الشائعة (هذه الصفحة)، وصفحة الاتصال.",
        'faq_step3_title': "الخطوة 3: اختيار أصل (في الصفحة الرئيسية)",
        'faq_step3_desc': "تحت 'الإعدادات الأساسية'، اختر 'الفئة' (مثل: أسهم أمريكية) ثم 'الأصل المراد توقعه' (مثل: آبل).",
        'faq_step4_title': "الخطوة 4: اختيار أفق التنبؤ",
        'faq_step4_desc': "اختر 'أفق التنبؤ' (مثل: سنة واحدة). هذا يحدد مدة التوقع المستقبلي وكمية البيانات التاريخية المستخدمة للتدريب.",
        'faq_step5_title': "الخطوة 5: بدء التدريب",
        'faq_step5_desc': "انقر على زر '🚀 تحسين وتدريب النموذج'. يمكنك تعديل 'إعدادات التدريب' (التعقيد، المحاولات) لتحسين النموذج، ولكن الإعدادات الافتراضية موصى بها للبدء.",
        'faq_step6_title': "الخطوة 6: تحليل النتائج",
        'faq_step6_desc': "بمجرد اكتمال التدريب، ستظهر ثلاث علامات تبويب:\n- **🔬 أداء النموذج:** يعرض تفاصيل النموذج ومنحنيات التعلم.\n- **📏 تقييم الاختبار:** يقارن تنbؤات النموذج بالبيانات الحقيقية (التي لم يرها) لتقييم دقته.\n- **🔮 التوقع المستقبلي:** يعرض التنبؤ للأفق المختار، مع تحليل وخيار للتنزيل.",

        'contact_title': "اتصل بنا",
        'contact_info': "لأية أسئلة، تعاون، أو دعم فني، يرجى الاتصال بـ:",
        'contact_name': "AHMED HAITHEM BERKANE",
        'contact_job_title': "CONSULTANT, DEVELOPPEUR WEB ET IA",
        'contact_phone': "الهاتف",
        'contact_email': "البريد الإلكتروني",
        'contact_address': "العنوان",

        'base_params': "⚙️ الإعدادات الأساسية",
        'category': "الفئة",
        'predict_asset': "الأصل المراد توقعه",
        'horizon': "أفق التنبؤ",
        'horizons': ["6 أشهر", "سنة واحدة", "3 سنوات", "5 سنوات"],
        'train_params': "🛠️ إعدادات التدريب",
        'model_complexity': "تعقيد النموذج",
        'complexities': ["بسيط (طبقة واحدة - سريع)", "معقد (طبقتين - دقيق)"],
        'optim_trials': "محاولات التحسين",
        'train_epochs': "مراحل التدريب النهائية",
        'hist_data': "📊 البيانات التاريخية",
        'info_analysis': "التحليل مبني على",
        'info_log_return': "لوغاريتم العائد لـ",
        'info_and': "و",
        'run_button': "🚀 تحسين وتدريب النموذج",
        'spinner_optim': "جاري التحسين",
        'success_optim': "✅ تم تحسين المتغيرات. جاري التدريب النهائي...",
        'spinner_train': "جاري التدريب النهائي لأفضل نموذج",
        'success_train': "✅ تم تحسين النموذج وتدريبه!",
        'tab_perf': "🔬 أداء النموذج",
        'tab_eval': "📏 تقييم الاختبار",
        'tab_proj': "🔮 التوقع المستقبلي",
        'perf_title': "تفاصيل وأداء النموذج",
        'hp_title': ":المتغيرات الفائقة المثلى",
        'hp_units': "الوحدات (الطبقة 1)",
        'hp_units_2': "الوحدات (الطبقة 2)",
        'hp_dropout': "التسرب (الطبقة 1)",
        'hp_dropout_2': "التسرب (الطبقة 2)",
        'hp_lr': "معدل التعلم",
        'metrics_title': ":مقاييس التدريب النهائية",
        'metrics_val_loss': "خسارة التحقق النهائية (val_loss)",
        'metrics_val_mae': "خطأ التحقق النهائي (val_mae)",
        'metrics_caption': ".المقاييس مبنية على لوغاريتم العائد المعدل",
        'charts_title': ":منحنيات التعلم",
        'chart_loss_title': "تطور الخسارة (MSE)",
        'chart_loss_train': "خسارة التدريب (MSE)",
        'chart_loss_val': "خسارة التحقق (MSE)",
        'chart_mae_title': "تطور متوسط الخطأ المطلق",
        'chart_mae_train': "متوسط خطأ التدريب (MAE)",
        'chart_mae_val': "متوسط خطأ التحقق (MAE)",
        'eval_title': "التقييم على مجموعة الاختبار (على الأسعار)",
        'eval_rmse': "(RMSE) على السعر",
        'eval_mae': "(MAE) على السعر",
        'eval_chart_title': "المقارنة بين الفعلي والمتوقع (على الأسعار)",
        'eval_real': "(القيم الحقيقية (السعر",
        'eval_pred': "(التنبؤات (السعر",
        'eval_toggle': "إظهار جدول بيانات الاختبار",
        'eval_error': "فشل في إعادة إنشاء بيانات الاختبار للتقييم.",
        'align_error': "خطأ في محاذاة البيانات أثناء التقييم.",
        'proj_title': "التوقع المستقبلي",
        'proj_spinner': "...جاري إنشاء التنبؤات",
        'proj_chart_title': "التوقع المستقبلي (مبني على لوغاريتم العائد)",
        'proj_hist': "(التاريخ (السعر",
        'proj_future': "(التوقع المستقبلي (السعر",
        'proj_analysis_title': "💬 تحليل التوقع",
        'proj_download': "📥 تحميل التنبؤات (CSV)",
        'comment_trend': "📈 **الاتجاه العام:** يتوقع النموذج",
        'comment_rise': "ارتفاعًا بنسبة",
        'comment_fall': "انخفاضًا بنسبة",
        'comment_for': "لـ",
        'comment_reaching': "ليصل إلى حوالي",
        'comment_by': "بحلول",
        'comment_q_trend': "🎯 **الربع القادم:** يُتوقع قيمة حوالي",
        'comment_q_expected': "لـ",
        'comment_q_end': "نهاية الربع",
        'data_error': "حدث خطأ أثناء تحميل البيانات",
        'prep_error_positive': "لم يتم العثور على بيانات موجبة لحساب لوغاريتم العائد.",
        'prep_error_log': "لا توجد بيانات بعد حساب لوغاريتم العائد. قد تكون مجموعة البيانات صغيرة جدًا.",
        'prep_error_seq': "لا توجد بيانات كافية لإنشاء تسلسلات",
    }
}

if 'lang' not in st.session_state:
    st.session_state.lang = 'fr'

def t(key):
    return translations.get(st.session_state.lang, translations['fr']).get(key, key)

HORIZON_KEYS = ['6m', '1y', '3y', '5y']
HORIZON_MAP = {
    '6m': {"train_years": 2, "predict_days": 180},
    '1y': {"train_years": 3, "predict_days": 365},
    '3y': {"train_years": 7, "predict_days": 3*365},
    '5y': {"train_years": 10, "predict_days": 5*365},
}
COMPLEXITY_KEYS = ['simple', 'complex']
LOOK_BACK = 60
TARGET_COL_ORIG_NAME = "Original_Price"

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

/* Cible tous les textes (paragraphes) DANS la sidebar */
[data-testid="stSidebar"] p {
    color: #FDF8E3; /* Beige clair pour le texte */
}

/* Cible tous les labels de widgets (radio, selectbox, etc.) DANS la sidebar */
[data-testid="stSidebar"] label {
    color: #FDF8E3 !important; /* Beige clair, !important pour forcer */
}

/* Cible tous les en-têtes (h1, h2, h3) DANS la sidebar */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FDF8E3; /* Beige clair pour le texte */
}

/* Cible spécifiquement le texte à l'intérieur des selectbox (la valeur sélectionnée) */
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p {
     color: #FDF8E3;
}

/* --- FIN DE LA CORRECTION CSS --- */


/* Titre principal */
h1 {
    color: #0A2342; /* Bleu Foncé */
    font-weight: bold;
}
/* Sous-titres */
h2, h3 {
    color: #0A2342; /* Bleu Foncé */
}
/* Bouton principal */
.stButton > button {
    background-color: #FF6B00; /* Orange */
    color: #FFFFFF; /* Texte blanc */
    border: none;
    border-radius: 5px;
    font-weight: bold;
    padding: 10px 20px;
}
.stButton > button:hover {
    background-color: #E05C00; /* Orange plus foncé au survol */
}
/* Bouton de téléchargement */
.stDownloadButton > button {
    background-color: #0A2342; /* Bleu Foncé */
    color: #FFFFFF;
}
.stDownloadButton > button:hover {
    background-color: #004A99; /* Bleu plus clair au survol */
}
/* Boîte d'information */
[data-testid="stInfo"] {
    background-color: #E6F0F8; /* Bleu très clair */
    border: 1px solid #0A2342;
    color: #0A2342;
}
/* Boîte de succès */
[data-testid="stSuccess"] {
    background-color: #DFF0D8;
    color: #3C763D;
}
/* Style des onglets (Tabs) */
.stTabs [data-baseweb="tab"] {
    background-color: #F0F2F6; /* Fond d'onglet inactif */
    color: #0A2342;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #FFFFFF;
    color: #FF6B00; /* Orange pour le texte de l'onglet actif */
    border-top: 2px solid #FF6B00;
}
/* Conteneurs "Carte" */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 1rem;
    background-color: #FFFFFF; /* Fond blanc pour les cartes */
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

RTL_CSS = """
<style>
body, .main, [data-testid="stSidebar"] {
    direction: rtl !important;
}
[data-testid="stSidebar"] .st-emotion-cache-16txtl3, 
[data-testid="stSidebar"] .st-emotion-cache-183lzff,
[data-testid="stSidebar"] .st-emotion-cache-1d8k8ss p,
[data-testid="stSidebar"] .st-emotion-cache-16idsys p {
    text-align: right !important;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    text-align: right !important;
}
h1, h2, h3, p {
    text-align: right !important;
}
[data-testid="stInfo"], [data-testid="stMetric"], [data-testid="stSuccess"], [data-testid="stError"] {
    text-align: right !important;
    direction: rtl !important;
}
.stButton > button {
    direction: ltr !important;
    text-align: right !important;
    padding-left: 1rem !important;
    padding-right: 2.5rem !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] {
    direction: ltr !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    margin-left: 0.5rem;
    margin-right: 0;
}
[data-testid="stTabs"] {
    width: 100%;
}
[data-testid="stTabs"] [role="tablist"] {
    justify-content: flex-end;
}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
     text-align: right !important;
     direction: rtl !important;
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
    if st.session_state.lang != 'fr':
        st.session_state.lang = 'fr'
        st.rerun()
if cols[1].button("🇬🇧", use_container_width=True):
    if st.session_state.lang != 'en':
        st.session_state.lang = 'en'
        st.rerun()
if cols[2].button("🇩🇿", use_container_width=True):
    if st.session_state.lang != 'ar':
        st.session_state.lang = 'ar'
        st.rerun()

st.sidebar.divider()

page_options = {
    'home': t('page_home'),
    'faq': t('page_faq'),
    'contact': t('page_contact')
}
selected_page_key = st.sidebar.radio(
    t('navigation'),
    options=list(page_options.keys()),
    format_func=lambda key: page_options[key],
    key="page_selector"
)

# ==============================
# FONCTIONS (LOGIQUE INCHANGÉE)
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
            df['Volume'] = np.log1p(df['Volume'])
    df = df[features].dropna()
    return df, col_to_select, features

def prepare_data(df, features):
    target_col_name = features[0]
    df_copy = df.copy()
    df_copy = df_copy[df_copy[target_col_name] > 0]
    if df_copy.empty:
        st.error(t('prep_error_positive'))
        return (None,) * 7
    df_copy[TARGET_COL_ORIG_NAME] = df_copy[target_col_name]
    df_copy[target_col_name] = np.log(df_copy[target_col_name] / df_copy[target_col_name].shift(1))
    df_copy = df_copy.dropna()
    if df_copy.empty:
        st.error(t('prep_error_log'))
        return (None,) * 7
    df_scaled = pd.DataFrame(index=df_copy.index)
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled[target_col_name] = price_scaler.fit_transform(df_copy[[target_col_name]])
    feature_scalers = {}
    if len(features) > 1:
        for feature in features[1:]:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_scaled[feature] = scaler.fit_transform(df_copy[[feature]])
            feature_scalers[feature] = scaler
    scaled_data = df_scaled.values
    X, y = [], []
    for i in range(LOOK_BACK, len(scaled_data)):
        X.append(scaled_data[i-LOOK_BACK:i, :])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        st.error(f"{t('prep_error_seq')} (Lookback = {LOOK_BACK}).")
        return (None,) * 7
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, y_train, X_test, y_test, price_scaler, feature_scalers, df_copy

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
    selected_complexity_display = st.sidebar.selectbox(t('model_complexity'), complexity_options_display, index=1)
    complexity_key = COMPLEXITY_KEYS[complexity_options_display.index(selected_complexity_display)]
    max_trials = st.sidebar.number_input(t('optim_trials'), 1, 20, 10, 1)
    epochs = st.sidebar.number_input(t('train_epochs'), 10, 100, 50, 5)

    try:
        df_original, target_col, features_used = load_data(symbol, sector_display, train_years)
        st.subheader(f"{t('hist_data')} - {symbol_name_display}")
        info_text = f"{t('info_analysis')} **{t('info_log_return')} {target_col}**"
        if len(features_used) > 1: info_text += f" {t('info_and')} **Log({', '.join(features_used[1:])})**."
        st.info(info_text)
        with st.container(border=True):
            st.line_chart(df_original[target_col])
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
        st.session_state['model'] = best_model
        st.session_state['history'] = history.history
        st.session_state['best_hp'] = best_hp
        st.session_state['price_scaler'] = price_scaler
        st.session_state['feature_scalers'] = feature_scalers
        st.session_state['df_original'] = df_original
        st.session_state['df_processed'] = df_processed
        st.session_state['features_used'] = features_used
        st.session_state['target_col'] = target_col
        st.session_state['trained_symbol'] = symbol
        st.session_state['trained_horizon'] = horizon_key
        st.session_state['trained_complexity'] = complexity_key

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
        df_history_processed = st.session_state['df_processed']
        features_used = st.session_state['features_used']
        target_col = st.session_state['target_col']
        prep_results_full = prepare_data(df_history_original, features_used)
        if prep_results_full[0] is not None:
            _, _, _, _, _, _, df_processed_full = prep_results_full
            X_train_len = int(len(df_processed_full) * 0.8) - LOOK_BACK
        else:
            st.error("Impossible de recalculer les données pour l'affichage.")
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
                _, _, X_test_eval, y_test_eval, _, _, df_processed_eval = prep_results_eval
                if len(X_test_eval) == 0:
                    st.warning("Pas assez de données pour un jeu de test. Essayez un horizon de données plus long.")
                else:
                    preds_scaled = model.predict(X_test_eval)
                    preds_log_returns = price_scaler.inverse_transform(preds_scaled)
                    test_start_index = X_train_len + LOOK_BACK
                    if test_start_index < len(df_processed_eval) and (test_start_index - 1) < len(df_processed_eval):
                        y_test_true_prices = df_processed_eval[TARGET_COL_ORIG_NAME].iloc[test_start_index:].values
                        y_test_previous_prices = df_processed_eval[TARGET_COL_ORIG_NAME].iloc[test_start_index - 1 : -1].values
                        if len(y_test_previous_prices) == len(preds_log_returns):
                            preds_rescaled = y_test_previous_prices * np.exp(preds_log_returns.flatten())
                            y_test_rescaled = y_test_true_prices
                            with st.container(border=True):
                                rmse = np.sqrt(mean_squared_error(y_test_rescaled, preds_rescaled))
                                mae = mean_absolute_error(y_test_rescaled, preds_rescaled)
                                col1, col2 = st.columns(2)
                                col1.metric(t('eval_rmse'), f"{rmse:.4f}")
                                col2.metric(t('eval_mae'), f"{mae:.4f}")
                            with st.container(border=True):
                                fig_compare = go.Figure()
                                test_dates = df_processed_eval.index[test_start_index:]
                                fig_compare.add_trace(go.Scatter(x=test_dates, y=y_test_rescaled.flatten(), mode='lines', name=t('eval_real'), line=dict(color='#0A2342')))
                                fig_compare.add_trace(go.Scatter(x=test_dates, y=preds_rescaled.flatten(), mode='lines', name=t('eval_pred'), line=dict(color='#FF6B00', dash='dash')))
                                fig_compare.update_layout(title_text=t('eval_chart_title'), hovermode="x unified")
                                st.plotly_chart(fig_compare, use_container_width=True)
                            if st.toggle(t('eval_toggle')):
                                results_df = pd.DataFrame({'Date': test_dates, 'Valeur Réelle': y_test_rescaled.flatten(), 'Prédiction': preds_rescaled.flatten()})
                                st.dataframe(results_df.set_index('Date'))
                        else:
                            st.error(t('align_error'))
                    else:
                        st.error("Erreur d'indexation lors de la création du jeu de test.")
        with tab_proj:
            st.subheader(f"{t('proj_title')} ({selected_horizon_display})")
            with st.spinner(t('proj_spinner')):
                last_60_days_processed = df_history_processed.iloc[-LOOK_BACK:]
                last_60_days_scaled = pd.DataFrame(index=last_60_days_processed.index)
                last_60_days_scaled[target_col] = price_scaler.transform(last_60_days_processed[[target_col]])
                for feature in features_used[1:]:
                    last_60_days_scaled[feature] = feature_scalers[feature].transform(last_60_days_processed[[feature]])
                current_batch = last_60_days_scaled.values.reshape(1, LOOK_BACK, len(features_used))
                future_preds_prices = []
                last_known_price = df_history_processed[TARGET_COL_ORIG_NAME].iloc[-1]
                for _ in range(future_days):
                    pred_scaled_log_return = model.predict(current_batch, verbose=0)[0]
                    pred_log_return = price_scaler.inverse_transform(pred_scaled_log_return.reshape(1, -1))
                    new_price = last_known_price * np.exp(pred_log_return[0, 0])
                    future_preds_prices.append(new_price)
                    last_known_price = new_price
                    new_entry = np.zeros((1, 1, len(features_used)))
                    new_entry[0, 0, 0] = pred_scaled_log_return[0]
                    if len(features_used) > 1:
                        for i, feature in enumerate(features_used[1:]):
                            mean_feature_val = current_batch[0, :, i+1].mean()
                            new_entry[0, 0, i+1] = mean_feature_val
                    current_batch = np.append(current_batch[:, 1:, :], new_entry, axis=1)
            future_preds_rescaled = np.array(future_preds_prices).flatten()
            future_dates = pd.date_range(start=df_history_processed.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')
            df_future = pd.DataFrame(future_preds_rescaled, index=future_dates, columns=['Prévision'])
            with st.container(border=True):
                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(x=df_history_original.index, y=df_history_original[target_col], mode='lines', name=t('proj_hist'), line=dict(color='#0A2342')))
                fig_future.add_trace(go.Scatter(x=df_future.index, y=df_future['Prévision'], mode='lines', name=t('proj_future'), line=dict(color='#FF6B00', dash='dash')))
                fig_future.update_layout(title_text=t('proj_chart_title'), hovermode="x unified")
                st.plotly_chart(fig_future, use_container_width=True)
            st.subheader(t('proj_analysis_title'))
            with st.container(border=True):
                last_price = df_history_processed[TARGET_COL_ORIG_NAME].iloc[-1]
                comment_text = generate_prediction_commentary(last_price, df_future, symbol_name_display, t)
                st.markdown(comment_text, unsafe_allow_html=True if st.session_state.lang == 'ar' else False)
            csv = df_future.to_csv().encode('utf-8')
            st.download_button(t('proj_download'), csv, f"forecast_{symbol}.csv", "text/csv")

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

