import os
import json
from flask import Flask, request, render_template,jsonify
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, SpeechRecognizer

app = Flask(__name__)

# Set environment variables
os.environ["AZURE_SPEECH_KEY"] = "dcf1618bbfb44cd0ab3e8155f5e35d8a"
os.environ["AZURE_REGION"] = "eastus"
os.environ["CUSTOM_SPEECH_ENDPOINT"] = "1ca0480b-c67b-45e4-a644-b0b028ac581c"

# Initialize Azure Speech services
speech_config = SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region=os.getenv("AZURE_REGION"))
speech_config.endpoint_id = os.getenv("CUSTOM_SPEECH_ENDPOINT")
speech_config.speech_recognition_language = "en-US"

def ask_custom_speech_model(prompt):
    speech_recognizer = SpeechRecognizer(speech_config=speech_config)
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    # Create a speech synthesizer to generate audio from the text prompt
    speech_synthesizer = SpeechSynthesizer(speech_config=speech_config)
    # Generate audio from the text prompt
    audio_result = speech_synthesizer.speak_text_async(prompt).get()
    # Create a speech recognizer to recognize the generated audio
    speech_recognizer = SpeechRecognizer(speech_config=speech_config)
    # Recognize the generated audio
    speech_recognition_result = speech_recognizer.recognize_once_async(audio_result).get()

    return speech_recognition_result.text


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    prompt = "Hello, how are you?"  # Hardcode the prompt

    response = ask_custom_speech_model(prompt)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)