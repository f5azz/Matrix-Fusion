import os
from groq import Groq

_groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=_groq_api_key) if _groq_api_key else None


# ---------------- RULE-BASED BACKUP ----------------
def get_recommendation(disease, location, temperature=None, humidity=None):

    recommendations = []
    recovery = "Unknown"

    if "Tomato" in disease:
        recommendations = [
            "Remove infected leaves",
            "Apply neem oil spray",
            "Improve airflow",
            "Avoid overwatering"
        ]
        recovery = "14–21 days"

    elif "Apple" in disease:
        recommendations = [
            "Use fungicide spray",
            "Prune infected branches"
        ]
        recovery = "10–15 days"

    elif "Grape" in disease:
        recommendations = [
            "Apply sulfur-based fungicide",
            "Ensure sunlight exposure"
        ]
        recovery = "12–18 days"

    else:
        recommendations = ["Monitor plant condition"]

    return recommendations, recovery


# ---------------- LLM FUNCTION ----------------
def get_llm_recommendation(crop, disease, location, temperature, humidity):
    if client is None:
        raise ValueError("Missing GROQ_API_KEY environment variable")

    prompt = f"""
    You are an agricultural expert writing for a production web app.

    Crop: {crop}
    Disease: {disease}
    Location: {location}
    Temperature: {temperature}
    Humidity: {humidity}

    Provide a concise, professional response in exactly 4 short bullet points:
    - Issue
    - Likely cause
    - Immediate action
    - Prevention
    Keep each bullet under 12 words. No extra intro or outro text.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content