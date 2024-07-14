from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from utilidades.prompt_creator import crear_prompt
from data.db_config import get_database
import os
from dotenv import load_dotenv

load_dotenv()

db = get_database()
llm = ChatGroq(model='mixtral-8x7b-32768', temperature=0)

def save_chat(session_id, question, answer):
    chats_collection = db['chats']
    chat = {"session_id": session_id, "question": question, "answer": answer}
    chats_collection.insert_one(chat)

def load_chat(session_id):
    chats_collection = db['chats']
    chat_history = chats_collection.find({"session_id": session_id})
    return [{"question": chat["question"], "answer": chat["answer"]} for chat in chat_history]

def get_answer(session_id, retriever, pregunta, texto):
    chain = create_stuff_documents_chain(llm, crear_prompt(texto))
    rag = create_retrieval_chain(retriever, chain)
    results = rag.invoke({"input": pregunta})
    save_chat(session_id, pregunta, results['answer'])
    return results['answer']
