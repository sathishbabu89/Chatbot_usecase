import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)
deployment = os.getenv("AZURE_DEPLOYMENT_NAME")

st.title("🏦 Banking Customer Support Chatbot")

# Initialize session state to keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": (
            "You are a helpful banking customer support assistant. "
            "Help users with queries about password reset, transactions, "
            "account info, product recommendations, and general banking FAQs."
            "If the user asks about anything outside banking, politely respond: "
            "'Sorry, I can only assist with banking related queries.'"
        )}
    ]

def generate_response(messages):
    response = client.chat.completions.create(
        model=deployment,
        messages=messages
    )
    return response.choices[0].message.content

def is_banking_query(text):
    banking_keywords = ["account", "transaction", "password", "loan", "bank", "credit", "debit", "statement"]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in banking_keywords)

# User input
user_input = st.text_input("Ask your banking question here:")

if user_input:
    # Append user message
    if not is_banking_query(user_input):
        st.markdown("**Bot:** Sorry, I can only assist with banking related queries.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate assistant response
        with st.spinner("Thinking..."):
            assistant_response = generate_response(st.session_state.messages)

        # Append assistant response
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Display chat messages
for msg in st.session_state.messages[1:]:  # skip system message in display
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
