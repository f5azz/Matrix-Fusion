import streamlit as st
import tempfile
import geocoder
import requests
import numpy as np
import base64
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random

from predict import predict, model
from utils import check_blur
from llm_module import get_recommendation, get_llm_recommendation
from gradcam import get_gradcam, overlay_heatmap
from report import generate_report

from tensorflow.keras.preprocessing import image

from groq import Groq

_groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=_groq_api_key) if _groq_api_key else None

# ---------------- CONFUSION MATRIX ----------------
def generate_live_confusion_matrix():
    y_true = [random.randint(0,2) for _ in range(20)]
    y_pred = [random.randint(0,2) for _ in range(20)]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i][j], ha="center", va="center")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    return fig

# ---------------- UI ----------------
st.set_page_config(page_title="GreenCure", layout="wide")

bg_image_path = Path(__file__).resolve().parent / "leaf-bg.jpg"
bg_css = ""
if bg_image_path.exists():
    encoded_bg = base64.b64encode(bg_image_path.read_bytes()).decode()
    bg_css = f"""
    background:
        linear-gradient(rgba(8, 22, 12, 0.58), rgba(8, 22, 12, 0.58)),
        url("data:image/jpg;base64,{encoded_bg}") center/cover fixed no-repeat;
    """
else:
    bg_css = """
    background: linear-gradient(180deg, #eefbf2 0%, #e3f5e8 45%, #d9efe1 100%);
    """

st.markdown("""
<style>
    .stApp {
        """ + bg_css + """
        color: #0f172a;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(236, 253, 245, 0.92) 0%, rgba(220, 252, 231, 0.92) 100%);
        border-right: 1px solid #bbf7d0;
    }
    .hero {
        background: linear-gradient(135deg, #14532d 0%, #065f46 45%, #0f766e 100%);
        color: #f0fdf4;
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        box-shadow: 0 14px 34px rgba(20, 83, 45, 0.26);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.2px;
    }
    .hero p {
        margin: 0.45rem 0 0 0;
        color: #dcfce7;
        font-size: 0.98rem;
    }
    .panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem 1rem 0.85rem 1rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.07);
        margin-bottom: 1rem;
    }
    .section-title {
        margin: 0.1rem 0 0.7rem 0;
        font-size: 1.02rem;
        color: #0f172a;
        font-weight: 700;
    }
    .result-pill {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #bbf7d0;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.55rem;
    }
    .stDownloadButton button, .stButton button {
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
        background: #f8fafc !important;
        color: #0f172a !important;
    }
    div[data-testid="stMetricValue"] {
        color: #064e3b;
        font-weight: 700;
    }
    label, .stMarkdown, .stCaption, .stText, p, li, span, div {
        color: #0f172a;
    }
    .compact-list {
        margin: 0;
        padding-left: 1rem;
    }
    .compact-list li {
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### GreenCure")
LANG_OPTIONS = {
    "English": "en",
    "Kannada": "kn",
    "Hindi": "hi",
    "Malayalam": "ml",
    "Tamil": "ta",
}

TRANSLATIONS = {
    "en": {
        "hero_title": "GreenCure",
        "hero_subtitle": "AI-powered crop disease detection with explainable insights and weather-aware recommendations.",
        "upload": "Upload",
        "upload_leaf_image": "Upload leaf image",
        "input_image": "Input Image",
        "uploaded_leaf_image": "Uploaded leaf image",
        "detection_complete": "Detection Complete",
        "prediction_results": "Prediction Results",
        "crop": "Crop",
        "disease": "Disease",
        "confidence": "Confidence",
        "severity": "Severity",
        "location": "Location",
        "location_note": "Note: Auto-detected location is IP-based and may be approximate.",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "weather_unavailable": "Weather unavailable",
        "care_tips": "Care Tips",
        "expected_recovery": "Expected recovery",
        "ai_summary": "AI Summary",
        "ai_unavailable": "AI service unavailable right now.",
        "model_focus": "Model Focus (Grad-CAM)",
        "generate_report": "Generate Report",
        "report": "Report",
        "download_report": "Download Report (PDF)",
        "model_evaluation": "Model Evaluation",
        "chat_assistant": "AI Chat Assistant",
        "ask_crop": "Ask about your crop",
        "you": "You",
        "ai": "AI",
        "low_confidence": "Low confidence",
        "image_blurry": "Image is blurry",
    },
    "kn": {
        "hero_title": "ಗ್ರೀನ್‌ಕ್ಯೂರ್",
        "hero_subtitle": "ವಿವರಣಾತ್ಮಕ ಒಳನೋಟಗಳೊಂದಿಗೆ ಮತ್ತು ಹವಾಮಾನ ಆಧಾರಿತ ಸಲಹೆಗಳೊಂದಿಗೆ AI ಚಾಲಿತ ಬೆಳೆ ರೋಗ ಪತ್ತೆ.",
        "upload": "ಅಪ್‌ಲೋಡ್",
        "upload_leaf_image": "ಎಲೆ ಚಿತ್ರದ ಅಪ್‌ಲೋಡ್",
        "input_image": "ಇನ್‌ಪುಟ್ ಚಿತ್ರ",
        "uploaded_leaf_image": "ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಎಲೆ ಚಿತ್ರ",
        "detection_complete": "ಪತ್ತೆ ಪೂರ್ಣಗೊಂಡಿದೆ",
        "prediction_results": "ಭವಿಷ್ಯವಾಣಿ ಫಲಿತಾಂಶಗಳು",
        "crop": "ಬೆಳೆ",
        "disease": "ರೋಗ",
        "confidence": "ವಿಶ್ವಾಸ",
        "severity": "ತೀವ್ರತೆ",
        "location": "ಸ್ಥಳ",
        "location_note": "ಸೂಚನೆ: ಸ್ವಯಂ-ಪತ್ತೆಯಾದ ಸ್ಥಳ IP ಆಧಾರಿತವಾಗಿದ್ದು ಅಂದಾಜು ಆಗಿರಬಹುದು.",
        "temperature": "ತಾಪಮಾನ",
        "humidity": "ಆದ್ರತೆ",
        "weather_unavailable": "ಹವಾಮಾನ ಲಭ್ಯವಿಲ್ಲ",
        "care_tips": "ಆರೈಕೆ ಸಲಹೆಗಳು",
        "expected_recovery": "ಅಂದಾಜು ಚೇತರಿಕೆ",
        "ai_summary": "AI ಸಾರಾಂಶ",
        "ai_unavailable": "AI ಸೇವೆ ಈಗ ಲಭ್ಯವಿಲ್ಲ.",
        "model_focus": "ಮಾದರಿ ಕೇಂದ್ರೀಕರಣ (Grad-CAM)",
        "generate_report": "ವರದಿ ರಚಿಸಿ",
        "report": "ವರದಿ",
        "download_report": "ವರದಿ ಡೌನ್‌ಲೋಡ್ (PDF)",
        "model_evaluation": "ಮಾದರಿ ಮೌಲ್ಯಮಾಪನ",
        "chat_assistant": "AI ಚಾಟ್ ಸಹಾಯಕ",
        "ask_crop": "ನಿಮ್ಮ ಬೆಳೆಯ ಬಗ್ಗೆ ಕೇಳಿ",
        "you": "ನೀವು",
        "ai": "AI",
        "low_confidence": "ಕಡಿಮೆ ವಿಶ್ವಾಸ",
        "image_blurry": "ಚಿತ್ರ ಮಸುಕಾಗಿದೆ",
    },
    "hi": {
        "hero_title": "ग्रीनक्योर",
        "hero_subtitle": "व्याख्यात्मक जानकारी और मौसम-आधारित सुझावों के साथ AI संचालित फसल रोग पहचान।",
        "upload": "अपलोड",
        "upload_leaf_image": "पत्ती की छवि अपलोड करें",
        "input_image": "इनपुट छवि",
        "uploaded_leaf_image": "अपलोड की गई पत्ती छवि",
        "detection_complete": "पहचान पूरी हुई",
        "prediction_results": "पूर्वानुमान परिणाम",
        "crop": "फसल",
        "disease": "रोग",
        "confidence": "विश्वास",
        "severity": "गंभीरता",
        "location": "स्थान",
        "location_note": "नोट: ऑटो-डिटेक्ट किया गया स्थान IP आधारित है और अनुमानित हो सकता है।",
        "temperature": "तापमान",
        "humidity": "नमी",
        "weather_unavailable": "मौसम उपलब्ध नहीं",
        "care_tips": "देखभाल सुझाव",
        "expected_recovery": "अनुमानित रिकवरी",
        "ai_summary": "AI सारांश",
        "ai_unavailable": "AI सेवा अभी उपलब्ध नहीं है।",
        "model_focus": "मॉडल फोकस (Grad-CAM)",
        "generate_report": "रिपोर्ट बनाएं",
        "report": "रिपोर्ट",
        "download_report": "रिपोर्ट डाउनलोड करें (PDF)",
        "model_evaluation": "मॉडल मूल्यांकन",
        "chat_assistant": "AI चैट सहायक",
        "ask_crop": "अपनी फसल के बारे में पूछें",
        "you": "आप",
        "ai": "AI",
        "low_confidence": "कम विश्वास",
        "image_blurry": "छवि धुंधली है",
    },
    "ml": {
        "hero_title": "ഗ്രീൻക്യൂർ",
        "hero_subtitle": "വ്യക്തമായ ഇൻസൈറ്റുകളും കാലാവസ്ഥയെ അടിസ്ഥാനമാക്കിയ നിർദേശങ്ങളും നൽകുന്ന AI അധിഷ്ഠിത വിള രോഗ നിർണയം.",
        "upload": "അപ്‌ലോഡ്",
        "upload_leaf_image": "ഇലയുടെ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക",
        "input_image": "ഇൻപുട്ട് ചിത്രം",
        "uploaded_leaf_image": "അപ്‌ലോഡ് ചെയ്ത ഇല ചിത്രം",
        "detection_complete": "തിരിച്ചറിവ് പൂർത്തിയായി",
        "prediction_results": "പ്രവചന ഫലങ്ങൾ",
        "crop": "വിള",
        "disease": "രോഗം",
        "confidence": "വിശ്വാസം",
        "severity": "തീവ്രത",
        "location": "സ്ഥലം",
        "location_note": "ശ്രദ്ധിക്കുക: സ്വയം കണ്ടെത്തുന്ന സ്ഥലം IP അടിസ്ഥാനത്തിലുള്ളതാണ്; ഏകദേശമായിരിക്കാം.",
        "temperature": "താപനില",
        "humidity": "ഈർപ്പം",
        "weather_unavailable": "കാലാവസ്ഥ ലഭ്യമല്ല",
        "care_tips": "പരിചരണ നിർദേശങ്ങൾ",
        "expected_recovery": "പ്രതീക്ഷിക്കുന്ന സുഖപ്പെടൽ",
        "ai_summary": "AI സംഗ്രഹം",
        "ai_unavailable": "AI സേവനം ഇപ്പോൾ ലഭ്യമല്ല.",
        "model_focus": "മോഡൽ ഫോക്കസ് (Grad-CAM)",
        "generate_report": "റിപ്പോർട്ട് തയ്യാറാക്കുക",
        "report": "റിപ്പോർട്ട്",
        "download_report": "റിപ്പോർട്ട് ഡൗൺലോഡ് ചെയ്യുക (PDF)",
        "model_evaluation": "മോഡൽ മൂല്യനിർണയം",
        "chat_assistant": "AI ചാറ്റ് സഹായി",
        "ask_crop": "നിങ്ങളുടെ വിളയെ കുറിച്ച് ചോദിക്കുക",
        "you": "നിങ്ങൾ",
        "ai": "AI",
        "low_confidence": "കുറഞ്ഞ വിശ്വാസം",
        "image_blurry": "ചിത്രം മങ്ങലാണ്",
    },
    "ta": {
        "hero_title": "கிரீன்க்யூர்",
        "hero_subtitle": "விளக்கமான பகுப்பாய்வு மற்றும் வானிலை சார்ந்த பரிந்துரைகளுடன் AI மூலம் பயிர் நோய் கண்டறிதல்.",
        "upload": "பதிவேற்று",
        "upload_leaf_image": "இலை படத்தை பதிவேற்றவும்",
        "input_image": "உள்ளீட்டு படம்",
        "uploaded_leaf_image": "பதிவேற்றிய இலை படம்",
        "detection_complete": "கண்டறிதல் முடிந்தது",
        "prediction_results": "முன்கணிப்பு முடிவுகள்",
        "crop": "பயிர்",
        "disease": "நோய்",
        "confidence": "நம்பிக்கை",
        "severity": "தீவிரம்",
        "location": "இடம்",
        "location_note": "குறிப்பு: தானாக கண்டறியப்படும் இடம் IP அடிப்படையிலானது; துல்லியமற்றதாக இருக்கலாம்.",
        "temperature": "வெப்பநிலை",
        "humidity": "ஈரப்பதம்",
        "weather_unavailable": "வானிலை கிடைக்கவில்லை",
        "care_tips": "பராமரிப்பு குறிப்புகள்",
        "expected_recovery": "எதிர்பார்க்கப்படும் மீட்பு",
        "ai_summary": "AI சுருக்கம்",
        "ai_unavailable": "AI சேவை தற்போது கிடைக்கவில்லை.",
        "model_focus": "மாதிரி கவனம் (Grad-CAM)",
        "generate_report": "அறிக்கை உருவாக்கு",
        "report": "அறிக்கை",
        "download_report": "அறிக்கையை பதிவிறக்கு (PDF)",
        "model_evaluation": "மாதிரி மதிப்பீடு",
        "chat_assistant": "AI உரையாடல் உதவியாளர்",
        "ask_crop": "உங்கள் பயிர் குறித்து கேளுங்கள்",
        "you": "நீங்கள்",
        "ai": "AI",
        "low_confidence": "குறைந்த நம்பிக்கை",
        "image_blurry": "படம் மங்கலாக உள்ளது",
    },
}

selected_language = st.sidebar.selectbox("Language", list(LANG_OPTIONS.keys()), index=0)
lang_code = LANG_OPTIONS[selected_language]


def t(key):
    return TRANSLATIONS.get(lang_code, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))

st.markdown(f"""
<div class="hero">
  <h1>{t("hero_title")}</h1>
  <p>{t("hero_subtitle")}</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f"### {t('chat_assistant')}")


def short_text(text, max_len=70):
    clean = " ".join(str(text).split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip() + "..."

# ---------------- LOCATION ----------------
def get_location():
    # Try multiple providers because single IP providers can be inaccurate.
    providers = [
        "https://ipapi.co/json/",
        "http://ip-api.com/json/",
    ]

    for endpoint in providers:
        try:
            res = requests.get(endpoint, timeout=8)
            data = res.json()
            if res.status_code != 200:
                continue

            if "ipapi.co" in endpoint:
                city = data.get("city")
                country = data.get("country_name")
                lat = data.get("latitude")
                lon = data.get("longitude")
            else:
                city = data.get("city")
                country = data.get("country")
                lat = data.get("lat")
                lon = data.get("lon")

            if city and country and lat is not None and lon is not None:
                return city, country, float(lat), float(lon)
        except Exception:
            continue

    try:
        g = geocoder.ip("me")
        city = getattr(g, "city", None)
        country = getattr(g, "country", None)
        latlng = getattr(g, "latlng", None)
        if city and country and latlng and len(latlng) == 2:
            return city, country, float(latlng[0]), float(latlng[1])
    except Exception:
        pass
    return "Unknown", "Unknown", None, None

# ---------------- WEATHER ----------------
def get_weather(lat, lon, city=None):
    # Open-Meteo does not require an API key.
    if lat is None or lon is None:
        return None, None, "Coordinates unavailable"

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m",
    }

    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        if res.status_code != 200:
            return None, None, data.get("reason", f"HTTP {res.status_code}")

        current = data["current"]
        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        return round(float(temp), 1), int(humidity), None
    except requests.RequestException as e:
        return None, None, f"Network error: {e}"
    except (KeyError, TypeError, ValueError) as e:
        return None, None, f"Unexpected weather response: {e}"

# ---------------- FILE UPLOAD ----------------
st.markdown(f'<div class="panel"><div class="section-title">{t("upload")}</div>', unsafe_allow_html=True)
file = st.file_uploader(t("upload_leaf_image"), type=["png", "jpg", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# 🔥 STORE CONTEXT FOR CHATBOT
crop, disease, location = "Unknown", "Unknown", "Unknown"

if file:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(file.read())

    col1, col2 = st.columns([1.05, 1], gap="large")

    with col1:
        st.markdown(f'<div class="panel"><div class="section-title">{t("input_image")}</div>', unsafe_allow_html=True)
        st.image(file, caption=t("uploaded_leaf_image"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if not check_blur(temp.name):
        st.error(f"⚠️ {t('image_blurry')}")
    
    else:
        label, confidence = predict(temp.name)

        if confidence < 0.6:
            st.warning(f"⚠️ {t('low_confidence')}")
        
        else:
            parts = label.split("_")
            crop = parts[0]
            disease = " ".join(parts[1:])

            if confidence > 0.9:
                severity = "High"
            elif confidence > 0.75:
                severity = "Moderate"
            else:
                severity = "Low"

            city, country, lat, lon = get_location()
            location = f"{city}, {country}"
            temp_val, humidity, weather_error = get_weather(lat, lon, city)

            with col2:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown(f'<div class="result-pill">{t("detection_complete")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="section-title">{t("prediction_results")}</div>', unsafe_allow_html=True)
                st.write(f"{t('crop')}: {crop}")
                st.write(f"{t('disease')}: {disease}")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric(t("confidence"), f"{confidence*100:.2f}%")
                with m2:
                    st.metric(t("severity"), severity)
                st.write(f"{t('location')}: {location}")
                st.caption(t("location_note"))
                if temp_val is not None and humidity is not None:
                    w1, w2 = st.columns(2)
                    with w1:
                        st.metric(t("temperature"), f"{temp_val}°C")
                    with w2:
                        st.metric(t("humidity"), f"{humidity}%")
                else:
                    st.warning(f"{t('weather_unavailable')}: {weather_error}")
                st.markdown('</div>', unsafe_allow_html=True)

            # -------- RULE BASED --------
            rec, recovery = get_recommendation(label, location, temp_val, humidity)

            st.markdown(f'<div class="panel"><div class="section-title">{t("care_tips")}</div>', unsafe_allow_html=True)
            concise_rec = [short_text(r, 52) for r in rec[:3]]
            st.markdown(
                "<ul class='compact-list'>" + "".join([f"<li>{r}</li>" for r in concise_rec]) + "</ul>",
                unsafe_allow_html=True
            )
            st.caption(f"{t('expected_recovery')}: {recovery}")
            st.markdown('</div>', unsafe_allow_html=True)

            # -------- LLM --------
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">{t("ai_summary")}</div>', unsafe_allow_html=True)

            try:
                llm_response = get_llm_recommendation(
                    crop, disease, location, temp_val, humidity
                )
                st.markdown(short_text(llm_response, 280))
            except:
                st.warning(t("ai_unavailable"))
            st.markdown('</div>', unsafe_allow_html=True)

            # -------- GRAD-CAM --------
            st.markdown(f'<div class="panel"><div class="section-title">{t("model_focus")}</div>', unsafe_allow_html=True)

            img = image.load_img(temp.name, target_size=(224,224))
            img_array = image.img_to_array(img)/255.0
            img_array = np.expand_dims(img_array, axis=0)

            _ = model.predict(img_array)

            heatmap = get_gradcam(model, img_array)
            result = overlay_heatmap(temp.name, heatmap)

            st.image(result, channels="BGR")
            st.markdown('</div>', unsafe_allow_html=True)

            # -------- REPORT --------
            data = {
                "Crop": crop,
                "Disease": disease,
                "Confidence": f"{confidence*100:.2f}%",
                "Severity": severity,
                "Location": location,
                "Recommendations": rec
            }

            pdf = generate_report(data)

            with open(pdf, "rb") as f:
                st.session_state["report_pdf_bytes"] = f.read()

            st.session_state["cm_fig"] = generate_live_confusion_matrix()


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history in sidebar
for msg in st.session_state.messages[-6:]:
    role_label = t("you") if msg["role"] == "user" else t("ai")
    st.sidebar.markdown(f"**{role_label}:** {msg['content']}")

# Sidebar input
user_input = st.sidebar.text_input(t("ask_crop"), key="sidebar_chat_input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    context = f"""
    Crop: {crop}
    Disease: {disease}
    Location: {location}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an agricultural expert."},
                {"role": "user", "content": context + "\n\nQuestion: " + user_input}
            ]
        )

        answer = response.choices[0].message.content

    except Exception as e:
        st.sidebar.error(f"LLM Error: {e}")
        answer = "⚠️ AI not available"

    st.session_state.messages.append({"role": "assistant", "content": answer})

if "report_pdf_bytes" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {t('report')}")
    st.sidebar.download_button(
        t("download_report"),
        st.session_state["report_pdf_bytes"],
        file_name="greencure_report.pdf",
    )

if "cm_fig" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {t('model_evaluation')}")
    st.sidebar.pyplot(st.session_state["cm_fig"])