from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def get_database():
    client = MongoClient(os.getenv("MONGODB_URI"))
    return client['iso27001_db']
