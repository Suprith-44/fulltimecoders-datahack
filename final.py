import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Cohere
import cohere
import os
from dotenv import load_dotenv
import tempfile
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Set up Cohere API Key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(COHERE_API_KEY)

# Initialize Cohere client and embedding model
cohere_embeddings = CohereEmbeddings(model="embed-english-v2.0", cohere_api_key=COHERE_API_KEY)

# Define the prompt template for vulnerability identification and follow-up questions
prompt_template = """
You are an AI assistant helping to analyze cybersecurity audit documents. Use the following context to identify vulnerabilities found in the audit. 
For each vulnerability, provide an explanation of why it exists and how it may be exploited.

{context}

Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context"]
)

# Initialize the Cohere model for LLM
llm = Cohere(client=cohere_client, model="command-xlarge-nightly")

# Function to load PDF documents
def load_pdf_documents(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    data = loader.load()
    os.unlink(temp_file_path)
    return data

# Function to create a VectorStore from documents
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    vector = FAISS.from_documents(split_docs, cohere_embeddings)
    return vector

# Function to create a RetrievalQA pipeline
def create_qa_pipeline(vector):
    retriever = vector.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 most relevant documents
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Function to handle automatic vulnerability identification and querying
def analyze_vulnerabilities(qa_pipeline):
    query = "Identify the vulnerabilities in this document. For each vulnerability, provide a brief explanation of why it exists and how it may be exploited."
    vulnerabilities = qa_pipeline({"query": query})["result"]
    return vulnerabilities

# Function to generate follow-up questions for a vulnerability
def generate_followup_questions(qa_pipeline, vulnerability):
    query = f"""
    Based on the vulnerability: "{vulnerability}", generate 3 specific and detailed follow-up questions 
    to gather maximum information regarding its impact, likelihood of exploitation, and mitigation strategies.
    """
    questions = qa_pipeline({"query": query})["result"]
    return questions.split('\n')

# Function to calculate risk score based on likelihood and impact
def calculate_risk_score(likelihood, impact):
    risk_score = (likelihood + impact) / 2
    return risk_score

# Function to generate the final vulnerability report
def generate_vulnerability_report(vulnerabilities, chat_history):
    report = []
    for i, vulnerability in enumerate(vulnerabilities):
        report_entry = {}
        report_entry['vulnerability'] = vulnerability
        report_entry['followup_responses'] = [entry['answer'] for entry in chat_history if i == entry.get('vulnerability_index', 0)]

        # Example logic for determining likelihood and impact based on responses
        likelihood = 3  # This could be based on user input or the nature of the vulnerability
        impact = 3      # This could be based on how critical the vulnerability is
        risk_score = calculate_risk_score(likelihood, impact)

        report_entry['likelihood'] = likelihood
        report_entry['impact'] = impact
        report_entry['risk_score'] = risk_score
        report.append(report_entry)
    
    return report

# Streamlit UI
def main():
    st.set_page_config(layout="wide")

    # Use session state to manage the current page
    if 'page' not in st.session_state:
        st.session_state.page = 'welcome'

    if st.session_state.page == 'welcome':
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

        st.title("Welcome to CyberAnalyst.ai")
        st.write("Revolutionizing cybersecurity audits with AI-powered document analysis and vulnerability assessment.")
        
        if st.button("Proceed"):
            st.session_state.page = 'main'
            st.experimental_rerun()

    elif st.session_state.page == 'main':
        st.title("Cybersecurity Audit Document Analyzer")

        # Initialize session state variables
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'vulnerabilities' not in st.session_state:
            st.session_state.vulnerabilities = None
        if 'current_vulnerability_index' not in st.session_state:
            st.session_state.current_vulnerability_index = 0
        if 'followup_questions' not in st.session_state:
            st.session_state.followup_questions = []
        if 'current_question_index' not in st.session_state:
            st.session_state.current_question_index = 0
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # File uploader for cybersecurity audit documents
        uploaded_file = st.file_uploader("Upload a Cybersecurity Audit Document (PDF)", type="pdf")

        if uploaded_file is not None:
            with st.spinner("Analyzing document for vulnerabilities..."):
                documents = load_pdf_documents(uploaded_file)
                st.success("Document loaded successfully!")

            if st.session_state.vector_store is None:
                st.session_state.vector_store = create_vector_store(documents)
                st.info("Vector store created.")

            qa_pipeline = create_qa_pipeline(st.session_state.vector_store)
            st.info("Vulnerability analysis pipeline ready.")

            if st.session_state.vulnerabilities is None:
                with st.spinner("Identifying vulnerabilities..."):
                    vulnerabilities = analyze_vulnerabilities(qa_pipeline)
                    st.session_state.vulnerabilities = vulnerabilities.split("\n\n")
                    st.session_state.current_vulnerability_index = 0
                    st.success("Vulnerabilities identified!")

            # Display all vulnerabilities
            st.subheader("Identified Vulnerabilities:")
            for i, vuln in enumerate(st.session_state.vulnerabilities, 1):
                st.write(f"{i}. {vuln}")

        # Display vulnerabilities one by one and ask follow-up questions
        if st.session_state.vulnerabilities:
            current_index = st.session_state.current_vulnerability_index
            current_vulnerability = st.session_state.vulnerabilities[current_index]

            st.subheader(f"Investigating Vulnerability {current_index + 1}:")
            st.write(current_vulnerability)

            if not st.session_state.followup_questions:
                st.session_state.followup_questions = generate_followup_questions(qa_pipeline, current_vulnerability)
                st.session_state.current_question_index = 0

            current_question = st.session_state.followup_questions[st.session_state.current_question_index]

            # Display the chat history so far
            st.subheader("Chat History")
            for entry in st.session_state.chat_history:
                st.write(f"**Q:** {entry['question']}")
                st.write(f"**A:** {entry['answer']}")

            # Display the current question
            st.write(f"Follow-up Question: {current_question}")
            user_response = st.text_input("Your response:")

            if st.button("Submit Response"):
                # Store the current question and user's response
                st.session_state.chat_history.append({
                    "question": current_question,
                    "answer": user_response,
                    "vulnerability_index": current_index
                })

                st.session_state.current_question_index += 1

                # Move to the next question or vulnerability if necessary
                if st.session_state.current_question_index >= len(st.session_state.followup_questions):
                    st.session_state.current_vulnerability_index += 1
                    st.session_state.followup_questions = []
                    st.session_state.current_question_index = 0

                if st.session_state.current_vulnerability_index >= len(st.session_state.vulnerabilities):
                    st.write("All vulnerabilities have been investigated.")
                    st.session_state.current_vulnerability_index = 0  # Reset the index
                else:
                    st.experimental_rerun()

        # Generate and display the final report
        if st.session_state.vulnerabilities and st.session_state.chat_history:
            st.subheader("Final Vulnerability Report")

            if st.button("Generate Report"):
                vulnerability_report = generate_vulnerability_report(st.session_state.vulnerabilities, st.session_state.chat_history)
                
                for i, entry in enumerate(vulnerability_report):
                    st.write(f"**Vulnerability {i+1}:** {entry['vulnerability']}")
                    st.write(f"**Likelihood of Exploitation:** {entry['likelihood']}/5")
                    st.write(f"**Impact:** {entry['impact']}/5")
                    st.write(f"**Risk Score:** {entry['risk_score']}/5")
                    st.write("**Follow-up Responses:**")
                    for response in entry['followup_responses']:
                        st.write(f"- {response}")
                    st.write("----")

            st.info("Click the 'Generate Report' button to analyze the vulnerabilities and risk scores.")

if __name__ == "__main__":
    main()