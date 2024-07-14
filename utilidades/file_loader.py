import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import streamlit as st

def load_document(file):
    if file.type == "application/pdf":
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(file.name)
    elif file.type == "text/plain":
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        loader = TextLoader(file.name)
    else:
        st.error("Unsupported file type")
        return None
    return loader.load()
