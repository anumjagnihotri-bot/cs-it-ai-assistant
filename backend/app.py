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

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                # Adjust safety settings to prevent empty blocks on academic queries
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

        # Check if response text is empty or blocked
        if not response.text:
            print("Warning: Model returned an empty response text.")
            return jsonify({"reply": "I'm sorry, I couldn't generate a response for that. Please try rephrasing your question."})

        return jsonify({"reply": response.text})

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)