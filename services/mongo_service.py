from pymongo import MongoClient
import numpy as np

client = MongoClient("mongodb://localhost:27017/")
db = client["cropsense_db"]

def get_mongo_folders():
    collection = db["capture_events"]
    return collection.distinct("sample_id")

def get_first_capture_event(sample_id):
    return db["capture_events"].find_one(
        {"sample_id": sample_id},
        sort=[("timestamp", 1)]
    )
def get_latest_capture_event(sample_id):
    return db["capture_events"].find_one(
        {"sample_id": sample_id},
        sort=[("timestamp", -1)]
    )

def get_hyperspectral_by_capture(capture_id):
    return db["sensors_hyperspectral"].find_one(
        {"capture_event_id": capture_id}
    )

def build_hyperspectral_data(sample_id):
    capture = get_latest_capture_event(sample_id)

    if not capture:
        return None

    hs = get_hyperspectral_by_capture(capture["_id"])

    if not hs:
        return None

    return {
        "signature": np.array(hs["mean_signature"], dtype=np.float32),
        "wavelengths": np.linspace(400, 1000, len(hs["mean_signature"]))
    }