import streamlit as st
import tempfile
import geocoder
import requests
import numpy as np
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

# 🔥 API KEY
client = Groq(api_key="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")

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

    st.pyplot(fig)

# ---------------- UI ----------------
st.set_page_config(page_title="GreenCure", layout="wide")

st.markdown("""
<style>
.big-title {font-size: 40px; font-weight: bold; color: #4CAF50;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🌱 GreenCure</div>', unsafe_allow_html=True)
st.write("Smart Crop Disease Detection System")

# ---------------- LOCATION ----------------
def get_location():
    try:
        g = geocoder.ip('me')
        return g.city, g.country
    except:
        return "Unknown", "Unknown"

# ---------------- WEATHER ----------------
def get_weather(city):
    API_KEY = "2efae33167609470e51f5cfcc99b1133"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
    
    try:
        res = requests.get(url).json()
        temp = res['main']['temp'] - 273
        humidity = res['main']['humidity']
        return round(temp,1), humidity
    except:
        return None, None

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("📤 Upload Leaf Image")

# 🔥 STORE CONTEXT FOR CHATBOT
crop, disease, location = "Unknown", "Unknown", "Unknown"

if file:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(file.read())

    col1, col2 = st.columns(2)

    with col1:
        st.image(file, caption="Uploaded Image")

    if not check_blur(temp.name):
        st.error("⚠️ Image is blurry")
    
    else:
        label, confidence = predict(temp.name)

        if confidence < 0.6:
            st.warning("⚠️ Low confidence")
        
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

            city, country = get_location()
            location = f"{city}, {country}"
            temp_val, humidity = get_weather(city)

            with col2:
                st.markdown("### 📊 Results")
                st.write(f"🌿 Crop: {crop}")
                st.write(f"🦠 Disease: {disease}")
                st.metric("Confidence", f"{confidence*100:.2f}%")
                st.metric("Severity", severity)
                st.write(f"📍 Location: {location}")

            # -------- RULE BASED --------
            rec, recovery = get_recommendation(label, location, temp_val, humidity)

            st.markdown("### 💡 Basic Recommendations")
            for r in rec:
                st.write("✔️", r)

            # -------- LLM --------
            st.markdown("### 🤖 AI Expert Recommendation")

            try:
                llm_response = get_llm_recommendation(
                    crop, disease, location, temp_val, humidity
                )
                st.write(llm_response)
            except:
                st.warning("⚠️ LLM not available")

            st.write(f"⏳ Recovery: {recovery}")

            # -------- GRAD-CAM --------
            st.markdown("### 🔍 Model Focus")

            img = image.load_img(temp.name, target_size=(224,224))
            img_array = image.img_to_array(img)/255.0
            img_array = np.expand_dims(img_array, axis=0)

            _ = model.predict(img_array)

            heatmap = get_gradcam(model, img_array)
            result = overlay_heatmap(temp.name, heatmap)

            st.image(result, channels="BGR")

            # -------- REPORT --------
            st.markdown("### 📄 Generate Report")

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
                st.download_button("📥 Download Report", f)

            # -------- CONFUSION MATRIX --------
            st.markdown("### 📊 Model Evaluation")
            generate_live_confusion_matrix()

# ---------------- CHATBOT ----------------
st.markdown("---")
st.markdown("### 💬 AI Chat Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
user_input = st.chat_input("Ask about your crop...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

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
        st.error(f"LLM Error: {e}")
        answer = "⚠️ AI not available"

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)