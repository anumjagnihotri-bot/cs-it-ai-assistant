import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def extract_reply(response):
    if response is None:
        return ""

    if isinstance(response, dict):
        text = response.get("text") or response.get("reply") or response.get("content")
        if text:
            return str(text).strip()

        candidates = response.get("candidates") or []
        for candidate in candidates:
            content = candidate.get("content", {}) if isinstance(candidate, dict) else getattr(candidate, "content", {})
            parts = content.get("parts", []) if isinstance(content, dict) else getattr(content, "parts", [])
            for part in parts:
                part_text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if part_text:
                    return str(part_text).strip()
        return ""

    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                return str(part_text).strip()
    return ""


@app.route("/api/chat", methods=["POST"])
def chat_with_ai():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        system_prompt = (
            "You are an expert Computer Science and IT professor and tutor. "
            "Provide clear, accurate, structured, and easy-to-understand explanations "
            "for any computer science or IT related questions."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_ONLY_HIGH"
                    ),
                ]
            )
        )

        reply = extract_reply(response)

        if not reply:
            print("Warning: Model returned an empty response or blocked output.")
            return jsonify({
                "reply": "I'm sorry, I couldn't generate a response for that. Please try rephrasing your question."
            })

        return jsonify({"reply": reply})

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)