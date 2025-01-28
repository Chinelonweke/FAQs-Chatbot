# FORVR MOOD FAQs Chat
A Streamlit application that provides an interactive chat interface for answering frequently asked questions (FAQs) about FORVR MOOD products. This application leverages the power of Groq's Mixtral-8x7b-32768 model and LangChain to deliver accurate and context-aware responses based on the content from the FORVR MOOD FAQ page.

## Features
Interactive Chat Interface: Engage in real-time conversations with the AI to get answers to your questions.

Context-Aware Responses: The AI provides accurate answers based on the context from the FORVR MOOD FAQ page.

Document Similarity Search: Explore relevant sections of the FAQ page that match your query.

Customizable UI: Stylish and user-friendly interface with custom CSS for a seamless experience.

## How to Run

## 1. Setup Environment

## Clone the repository
git clone https://github.com/your-repo/forvr-mood-faq-chat.git

cd forvr-mood-faq-chat

## Create and activate a virtual environment (optional but recommended)

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install dependencies
pip install -r requirements.txt

## 2. Get API Key
Visit Groq Cloud and sign up for an API key.

Add the API key to your .env file:


GROQ_API_KEY=your_groq_api_key_here

## 3. Run the Application
Start the Streamlit app:


streamlit run app.py

## 4. Use the Application
Enter your question in the input box.

View the AI's response in the chat interface.

Explore the document similarity search results for more context.

## Technical Stack
Frontend: Streamlit

AI Model: Groq's Mixtral-8x7b-32768

Embeddings: HuggingFace Embeddings

Vector Store: FAISS

Framework: LangChain

## Tips for Best Results
Ask Clear Questions: Formulate your questions clearly to get the most accurate responses.

Explore Similarity Search: Use the document similarity search to find relevant sections of the FAQ page.

Provide Context: If needed, provide additional context to refine the AI's responses.

### Example Use Cases
FAQ Lookup: Quickly find answers to common questions about FORVR MOOD products.

Customer Support: Assist customers by providing instant, accurate responses to their queries.

Content Exploration: Explore the FAQ page for detailed information on various topics.

Contributing
We welcome contributions! If you'd like to contribute to this project, please follow these steps:


## Contact
For questions, feedback, or support, please reach out to:

Email: nwekechinelo25@yahoo.com



## Acknowledgments
Groq: For providing the powerful Mixtral-8x7b-32768 model.

Streamlit: For enabling the creation of interactive web applications.

LangChain: For simplifying the integration of AI models and document processing.

Enjoy using the FORVR MOOD FAQs Chat to get instant, accurate answers to your questions! 🚀

