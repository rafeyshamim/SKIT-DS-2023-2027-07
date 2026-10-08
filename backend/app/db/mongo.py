import os
import json
from typing import Dict, Any, Optional, List
from app.core.config import settings
from app.core.logging import logger

class MongoManager:
    """
    Manages MongoDB connections for unstructured/semi-structured DICOM metadata,
    3D volumetric coordinate sets, slice-level heatmaps, and inference output documents.
    Provides graceful fallback to local JSON document store when MongoDB service is offline.
    """
    def __init__(self):
        self.client = None
        self.db = None
        self.is_connected = False
        self._fallback_store_dir = os.path.join(settings.BASE_DIR, "data", "mongo_store")
        os.makedirs(self._fallback_store_dir, exist_ok=True)

    def connect(self):
        try:
            from pymongo import MongoClient
            self.client = MongoClient(settings.MONGODB_URL, serverSelectionTimeoutMS=2000)
            # Verify connection
            self.client.admin.command('ping')
            self.db = self.client[settings.MONGODB_DATABASE]
            self.is_connected = True
            logger.info("Connected to MongoDB cluster at %s", settings.MONGODB_URL)
        except Exception as e:
            self.is_connected = False
            logger.warning(
                "MongoDB not reachable (%s). Utilizing file-backed NoSQL document store at %s",
                e, self._fallback_store_dir
            )

    def close(self):
        if self.client:
            self.client.close()
            self.is_connected = False

    def insert_document(self, collection_name: str, doc: Dict[str, Any]) -> str:
        """Inserts a document into collection or local document store."""
        doc_id = str(doc.get("_id") or doc.get("id") or os.urandom(8).hex())
        doc["_id"] = doc_id

        if self.is_connected and self.db is not None:
            try:
                self.db[collection_name].replace_one({"_id": doc_id}, doc, upsert=True)
                return doc_id
            except Exception as e:
                logger.error("Failed to write to MongoDB: %s. Using fallback store.", e)

        # Fallback file persistence
        col_dir = os.path.join(self._fallback_store_dir, collection_name)
        os.makedirs(col_dir, exist_ok=True)
        file_path = os.path.join(col_dir, f"{doc_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, default=str)
        return doc_id

    def get_document(self, collection_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a document by _id from collection or local document store."""
        if self.is_connected and self.db is not None:
            try:
                doc = self.db[collection_name].find_one({"_id": doc_id})
                if doc:
                    return doc
            except Exception as e:
                logger.error("Failed to read from MongoDB: %s", e)

        file_path = os.path.join(self._fallback_store_dir, collection_name, f"{doc_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def query_documents(self, collection_name: str, query_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Queries documents matching filter."""
        if self.is_connected and self.db is not None:
            try:
                return list(self.db[collection_name].find(query_filter))
            except Exception as e:
                logger.error("Failed to query MongoDB: %s", e)

        # Fallback simple search
        results = []
        col_dir = os.path.join(self._fallback_store_dir, collection_name)
        if os.path.exists(col_dir):
            for fname in os.listdir(col_dir):
                if fname.endswith(".json"):
                    with open(os.path.join(col_dir, fname), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        match = all(data.get(k) == v for k, v in query_filter.items())
                        if match:
                            results.append(data)
        return results


mongo_manager = MongoManager()


def get_mongo():
    return mongo_manager
