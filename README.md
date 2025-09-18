

````markdown
# Banking Customer Support Chatbot

An intelligent banking chatbot built with **Streamlit** and **Azure OpenAI GPT**. Users can upload banking-related PDF documents to extend the chatbot’s knowledge base. It uses semantic search with **Sentence Transformers** and **FAISS** for fast retrieval and provides accurate, domain-specific answers.

## Features

- Conversational banking assistant for FAQs and support queries  
- Upload PDFs to dynamically enhance knowledge  
- Semantic search using embeddings and FAISS  
- Built with Azure OpenAI GPT for natural language understanding  
- Streamlit UI for easy deployment and interaction  
- Guardrails to keep responses focused on banking topics

## Tech Stack

- Python  
- Streamlit  
- Azure OpenAI GPT  
- PyPDF2 (PDF extraction)  
- Sentence Transformers (embeddings)  
- FAISS (vector similarity search)  
- LangChain (optional chaining utilities)  

## Setup Instructions

1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/banking-chatbot.git
   cd banking-chatbot
````

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Azure OpenAI credentials and deployment info:

   ```env
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_ENDPOINT=https://your-resource-name.openai.azure.com/
   OPENAI_API_VERSION=2023-05-15
   AZURE_DEPLOYMENT_NAME=your_deployment_name
   ```

4. Run the app

   ```bash
   streamlit run chatbot.py
   ```

5. Upload a banking PDF and start chatting!

## Usage

* Upload banking-related PDF documents to provide the chatbot with up-to-date info.
* Ask banking questions related to products, services, policies, transactions, and more.
* The chatbot uses retrieved document context plus Azure OpenAI GPT to generate accurate answers.

## License

MIT License

---
