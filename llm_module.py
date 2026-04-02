from groq import Groq

# 🔥 Add your API key here
client = Groq(api_key="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")


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

    prompt = f"""
    You are an agricultural expert.

    Crop: {crop}
    Disease: {disease}
    Location: {location}
    Temperature: {temperature}
    Humidity: {humidity}

    Provide:
    1. Explanation
    2. Causes
    3. Treatment
    4. Prevention

    Keep it simple.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content