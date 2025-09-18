import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialize AzureOpenAI client for chat completions
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)
deployment = os.getenv("AZURE_DEPLOYMENT_NAME")

st.title("🏦 Banking Chatbot with PDF Upload (Local Embeddings + Azure LLM)")

# Initialize Sentence Transformer embedding model once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": (
            "You are a helpful banking customer support assistant. "
            "ONLY answer questions related to banking, "
            "including information provided in uploaded documents."
        )}
    ]

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "docs" not in st.session_state:
    st.session_state.docs = []

uploaded_file = st.file_uploader("Upload a banking-related PDF document", type=["pdf"])

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([text])
    st.session_state.docs = docs

    # Create embeddings for docs
    doc_texts = [doc.page_content for doc in docs]
    embeddings = embedding_model.encode(doc_texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    st.session_state.faiss_index = index

    st.success("Document processed and embeddings created!")

def retrieve_similar_docs(query, k=3):
    if st.session_state.faiss_index is None or not st.session_state.docs:
        return []

    query_embedding = embedding_model.encode([query])[0].reshape(1, -1)
    distances, indices = st.session_state.faiss_index.search(query_embedding, k)
    return [st.session_state.docs[i] for i in indices[0]]

def generate_response(messages, question=None):
    if st.session_state.faiss_index and question:
        relevant_docs = retrieve_similar_docs(question, k=3)
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

        messages = [
            {"role": "system", "content": (
                "You are a helpful banking customer support assistant. "
                "Answer questions based on the following context from uploaded documents:\n\n" + context_text
            )}
        ] + messages[1:]  # keep other conversation history except original system prompt

    response = client.chat.completions.create(
        model=deployment,
        messages=messages
    )
    return response.choices[0].message.content

user_input = st.text_input("Ask your banking question here:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Generating response..."):
        answer = generate_response(st.session_state.messages, question=user_input)

    st.session_state.messages.append({"role": "assistant", "content": answer})

for msg in st.session_state.messages[1:]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
