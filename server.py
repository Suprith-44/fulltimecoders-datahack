from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import faiss
import google.generativeai as genai
from google.generativeai import GenerativeModel

app = Flask(__name__)
CORS(app)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("Credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = GenerativeModel('gemini-pro')

# Global variable for the vector store
vectorstore = None

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    if not all([name, email, password]):
        return jsonify({"error": "Missing required fields"}), 400
    try:
        db.collection('users').add({
            'name': name,
            'email': email,
            'password': password,
            'createdAt': firestore.SERVER_TIMESTAMP
        })
        return jsonify({"message": "User signed up successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    if not all([email, password]):
        return jsonify({"success": False, "message": "Missing required fields"}), 400
    try:
        users_ref = db.collection('users')
        query = users_ref.where('email', '==', email).where('password', '==', password).get()
        if query:
            user = query[0].to_dict()
            return jsonify({"success": True, "user": {"name": user["name"], "email": user["email"]}}), 200
        else:
            return jsonify({"success": False, "message": "Invalid email or password"}), 401
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/get-network-security-question', methods=['GET'])
def get_network_security_question():
    global vectorstore
    if vectorstore is None:
        return jsonify({"error": "Vector store not initialized"}), 500
    try:
        # Use FAISS to retrieve a random question from the vector store
        question = vectorstore.similarity_search("", k=1)[0].page_content
        return jsonify({"question": question}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/check-answer', methods=['POST'])
def check_answer():
    data = request.json
    question = data.get('question')
    user_answer = data.get('answer')

    if not all([question, user_answer]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # Use Google's Generative AI to evaluate the answer
        prompt = f"Question: {question}\nUser's Answer: {user_answer}\nEvaluate the user's answer and provide feedback."
        response = model.generate_content(prompt)
        feedback = response.text

        return jsonify({"feedback": feedback}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def load_vector_store():
    global vectorstore
    try:
        # Load FAISS index from disk
        index = faiss.read_index("Network.faiss")
        
        # Recreate the vector store object with the loaded index
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS(index=index, embedding_function=embeddings)
        print("Vector store loaded successfully")
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")

if __name__ == '__main__':
    load_vector_store()
    app.run(debug=True)