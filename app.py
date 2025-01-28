import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS 
import time

from dotenv import load_dotenv
load_dotenv()

## Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    # Initialize HuggingFace embeddings
    st.session_state.embeddings = HuggingFaceEmbeddings()

    # Load documents from the web
    st.session_state.loader = WebBaseLoader("https://forvrmood.com/pages/faqs")
    st.session_state.docs = st.session_state.loader.load()

    # Split documents into chunks
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

    # Create FAISS vector store
    st.session_state.vectors = FAISS.from_documents(
    documents=st.session_state.final_documents,
    embedding=st.session_state.embeddings
)

# Custom CSS to style the title
st.markdown("""
<style>
.title {
    color: #FF69B4; /* Pink color */
    font-size: 30px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Streamlit app title with custom color
st.markdown('<div class="title">FORVR MOOD FAQs Chat</div>', unsafe_allow_html=True)


# Initialize Groq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

# Create document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Custom CSS to style the chat interface
st.markdown("""
<style>
.chat-message {
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    max-width: 70%;
}
.chat-message.user {
    background-color: #f5bad7;
    margin-left: 30%;
}
.chat-message.bot {
    background-color: #FF69B4;
    margin-right: 30%;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input prompt
prompt_input = st.text_input("Input your prompt here")

if prompt_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt_input})

    # Get response from the retrieval chain
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt_input})
    response_time = time.process_time() - start

    # Add bot response to chat history
    st.session_state.chat_history.append({"role": "bot", "content": response['answer']})

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot">{message["content"]}</div>', unsafe_allow_html=True)

    # Display document similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")