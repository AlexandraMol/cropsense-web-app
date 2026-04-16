from pymongo import MongoClient

def get_mongo_folders():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["cropsense_db"]
    collection = db["capture_events"]

    folders = collection.distinct("sample_id")
    return folders