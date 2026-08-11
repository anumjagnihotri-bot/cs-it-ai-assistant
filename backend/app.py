import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = genai.Client()

@app.route("/api/chat", methods=["POST"])
def chat_with_ai():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        system_prompt = (
            "You are an expert Computer Science and IT professor and tutor. "
            "Provide clear, accurate, structured, and easy-to-understand explanations "
            "for any computer science or IT related questions."
        )
        full_prompt = f"{system_prompt}\n\nUser Question: {user_message}"

        response = client.models.generate_content(
         model="gemini-3.6-flash",
            contents=full_prompt,
        )

        return jsonify({"reply": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)