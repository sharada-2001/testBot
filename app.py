
from flask import Flask, request, render_template, jsonify
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
import json
import os
import requests
import spacy
import re
from neo4j import GraphDatabase
import xml.etree.ElementTree as ET
from urllib.parse import quote

app = Flask(__name__)

# Set environment variables for OpenAI and Azure Speech
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://talkyopenai.openai.azure.com/"
os.environ["AZURE_OPENAI_KEY"] = "e52149679a3f4d21a8056bfdf9ed3b1d"
os.environ["AZURE_SPEECH_KEY"] = "898e53adf8ad46d6930721237b562aa7"
region = "swedencentral"

# Initialize Azure OpenAI and Speech services
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15"
)
deployment_id = "gpt-35-turbo-16k"
speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region=region)
speech_config.speech_recognition_language = "en-US"
speech_config.speech_synthesis_voice_name = 'en-US-JennyMultilingualNeural'

# # Load health datasets
# with open('datasets/health_assistant_115.json', 'r', encoding='utf-8') as f:
#     health_data = json.load(f)
# with open('datasets/disease_database_en.json', 'r', encoding='utf-8') as f:
#     disease_data = json.load(f)

# Initialize SpaCy for NLP
nlp = spacy.load('en_core_web_sm')

# Initialize Neo4j driver
neo4j_uri = "bolt://localhost:7687"
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=("neo4j", "password"))

# Function to create nodes and relationships in Neo4j
def create_medical_knowledge_graph(disease, symptom, treatment):
    with neo4j_driver.session() as session:
        session.write_transaction(_create_medical_knowledge_graph, disease, symptom, treatment)

def _create_medical_knowledge_graph(tx, disease, symptom, treatment):
    tx.run("MERGE (d:Disease {name: $disease}) "
           "MERGE (s:Symptom {name: $symptom}) "
           "MERGE (t:Treatment {name: $treatment}) "
           "MERGE (d)-[:HAS_SYMPTOM]->(s) "
           "MERGE (d)-[:TREATED_WITH]->(t)",
           disease=disease, symptom=symptom, treatment=treatment)

# Function to query the knowledge graph in Neo4j
def search_knowledge_graph(disease):
    with neo4j_driver.session() as session:
        result = session.run("MATCH (d:Disease)-[:HAS_SYMPTOM]->(s) "
                             "WHERE d.name = $disease "
                             "RETURN s.name", disease=disease)
        symptoms = [record["s.name"] for record in result]
    return symptoms

# Function to process and extract entities from the user's query using SpaCy
def process_query_with_spacy(user_query):
    doc = nlp(user_query)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to fetch data from OpenFDA
def fetch_openfda_drug_data(question, query_type=None):
    clean_question = re.sub(r'[^\w\s]', '', question)  # Remove punctuation
    query = clean_question.replace(" ", "+")
    url = f"https://api.fda.gov/drug/label.json?search={query}&limit=1"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'results' in data and len(data['results']) > 0:
            results = data['results'][0]
            openfda = results.get('openfda', {})

            if query_type == 'side effects':
                return results.get('adverse_reactions', 'No side effects information found.')
            elif query_type == 'uses':
                return results.get('indications_and_usage', 'No usage information found.')
            elif query_type == 'warnings':
                return results.get('warnings', 'No warning information found.')
            elif query_type == 'symptoms':
                return results.get('purpose', 'No symptoms information found.')
            else:
                return {
                    'name': openfda.get('generic_name', ['No name found'])[0],
                    'dosage': results.get('dosage_and_administration', 'No dosage information found.'),
                    'warnings': results.get('warnings', 'No warnings found.'),
                    'indications': results.get('indications_and_usage', 'No indications found.')
                }
        else:
            return "No relevant data found."

    except requests.RequestException as e:
        print(f"Error fetching data from OpenFDA: {e}")
        return "Error fetching data from OpenFDA."


# Function to fetch MedlinePlus summary
def fetch_medlineplus_summary(query):
    base_url = "https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term="
    encoded_query = quote(query)
    url = base_url + encoded_query

    try:
        response = requests.get(url)
        response.raise_for_status()

        if response.headers.get('Content-Type') == 'text/xml; charset=UTF-8':
            root = ET.fromstring(response.content)

            spelling_correction = root.find('.//spellingCorrection')
            if spelling_correction is not None:
                corrected_query = spelling_correction.text.strip('"')
                return fetch_medlineplus_summary(corrected_query)

            documents = root.findall('.//document')
            summaries = []
            for doc in documents:
                summary_element = doc.find('.//content[@name="FullSummary"]')
                summary = summary_element.text if summary_element is not None else "No summary information found."
                summaries.append(summary)

            return "\n\n".join(summaries) if summaries else "No relevant information found."
        else:
            return f"Unexpected content type: {response.headers.get('Content-Type')}, raw response: {response.text}"

    except requests.exceptions.RequestException as e:
        return f"Error fetching data from MedlinePlus: {str(e)}"


# Function to detect query type, process it with SpaCy, and search datasets
def search_dataset(question):
    question_lower = question.lower()
    print(f"Searching dataset for question: {question}")

    drug_patterns = {
        'side effects': r'\b(side effects|adverse effects)\b',
        'uses': r'\b(uses|usage)\b',
        'warnings': r'\b(warnings|precautions)\b',
        'symptoms': r'\b(symptoms|purpose)\b'
    }

    for query_type, pattern in drug_patterns.items():
        if re.search(pattern, question_lower):
            print(f"Matched query type: {query_type}")
            openfda_data = fetch_openfda_drug_data(question, query_type=query_type)
            if openfda_data and "Error" not in openfda_data:
                return ask_openai(f"Refine this information: {openfda_data}")

    if 'what is' in question_lower or 'define' in question_lower:
        openfda_data = fetch_openfda_drug_data(question)
        if openfda_data and "Error" not in openfda_data:
            return ask_openai(f"Refine this information: {openfda_data}")

    health_keywords = ['disease', 'symptoms', 'treatment', 'therapy', 'diagnosis', 'prevention', 'infection']

    if any(keyword in question_lower for keyword in health_keywords):
        print("Searching MedlinePlus for health keywords.")
        return fetch_medlineplus_summary(question)

    print("No match found, calling OpenAI.")
    return ask_openai(question)

# Function to call OpenAI for natural, human-like responses
def ask_openai(prompt, openfda_data=None):
    try:
        refined_prompt = prompt
        if openfda_data:
            refined_prompt = f"""
            Please improve the following medical information and fill in any missing details based on your knowledge:
            {openfda_data}
            """
        
        response = client.chat.completions.create(
            model=deployment_id,
            max_tokens=300,
            messages=[{"role": "user", "content": refined_prompt}]
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return "Sorry, I couldn't generate a response from OpenAI."

# Function to convert text to speech
def text_to_speech(text, language="en-US"):
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    
    # Check if the text is a dictionary and convert it to string
    if isinstance(text, dict):
        text = json.dumps(text)

    result = speech_synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized: {text}")
        return "Speech synthesized successfully."
    else:
        print(f"Speech synthesis failed: {result.reason}")
        return "Error synthesizing speech."


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        audio_input = speechsdk.AudioConfig(use_default_microphone=True)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

        print("Listening for speech input...")
        result = speech_recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            user_query = result.text
            print(f"Recognized speech: {user_query}")

            # Search dataset and external data
            response_text = search_dataset(user_query)

            # If no dataset match, call OpenAI
            if response_text == "Sorry, I couldn't find a relevant response.":
                response_text = ask_openai(user_query)

        else:
            response_text = "Sorry, I couldn't recognize the speech."

        # Ensure response_text is passed as string for TTS
        text_to_speech(str(response_text))
        return jsonify({"response": response_text}), 200

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/speak', methods=['POST'])
def speak():
    text = request.json.get('text', '')
    language = request.json.get('language', 'en-US')

    print(f"Text to synthesize: {text}, Language: {language}")  # Log the text and language

    result = text_to_speech(text, language)
    
    # Log the result of speech synthesis
    print(f"Speech synthesis result: {result}")

    return jsonify({'message': result})

if __name__ == '__main__':
    app.run(debug=True)
