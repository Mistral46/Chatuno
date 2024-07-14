import streamlit as st
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from utilidades.file_loader import load_document
from models.chat import get_answer, load_chat
import os
from dotenv import load_dotenv

load_dotenv()

texto = """Tú eres un asistente para tareas de respuesta a preguntas. 
Usa los siguientes fragmentos de contexto recuperado para responder la pregunta. 
Si no sabes la respuesta, di que no sabes. Usa un máximo de diez oraciones."""

def main():
    st.title("Asistente de Implementación ISO 27001")
    
    # Identificar sesión del usuario
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    session_id = st.session_state.session_id

    api_choice = st.selectbox("Seleccione la API a utilizar", ("OpenAI", "Groq"))

    uploaded_files = st.file_uploader("Cargue documentos", type=["pdf", "txt"], accept_multiple_files=True)
    pregunta = st.text_input("Ingrese su pregunta:")
    
    # Cargar historial de chat
    chat_history = load_chat(session_id)
    
    if chat_history:
        st.subheader("Historial de Conversación")
        for chat in chat_history:
            st.write(f"Pregunta: {chat['question']}")
            st.write(f"Respuesta: {chat['answer']}")

    if uploaded_files:
        docs = []
        for file in uploaded_files:
            file_docs = load_document(file)
            if file_docs:
                docs.extend(file_docs)

        if docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory='./BD')
            chroma_local = Chroma(persist_directory="./BD", embedding_function=OpenAIEmbeddings())
            retriever = chroma_local.as_retriever()

            if pregunta:
                answer = get_answer(session_id, retriever, pregunta, texto)
                st.write(f"Respuesta: {answer}")

if __name__ == "__main__":
    main()
