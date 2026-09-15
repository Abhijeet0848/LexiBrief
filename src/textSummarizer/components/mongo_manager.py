import os
import json
import time
import uuid
import urllib.parse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from textSummarizer.logging import logger

def _load_env_file():
    """Lightweight fallback .env loader."""
    env_path = ".env"
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and not os.getenv(k):
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file()


class MongoDBManager:
    """Manages persistence for Summaries and Documents in MongoDB with a local fallback engine."""

    def __init__(self, db_name: str = "lexibrief_db"):
        self.db_name = os.getenv("MONGODB_DB", db_name)
        self.mongo_url = self._resolve_mongo_url()
        self.client = None
        self.db = None
        self.use_mongo = False

        self._init_connection()

    def _resolve_mongo_url(self) -> str:
        """Resolves MongoDB connection URL from individual credentials or full connection string."""
        explicit_url = os.getenv("MONGODB_URL") or os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or os.getenv("DATABASE_URL")
        if explicit_url:
            return explicit_url.strip().strip("'\"")

        user = os.getenv("MONGODB_USER")
        pwd = os.getenv("MONGODB_PASSWORD")
        cluster = os.getenv("MONGODB_CLUSTER")

        if user and pwd and cluster:
            encoded_pwd = urllib.parse.quote_plus(pwd)
            encoded_user = urllib.parse.quote_plus(user)
            cluster_host = cluster if ".mongodb.net" in cluster else f"{cluster}.mongodb.net"
            return f"mongodb+srv://{encoded_user}:{encoded_pwd}@{cluster_host}/{self.db_name}?retryWrites=true&w=majority"

        return "mongodb://localhost:27017"

    def _init_connection(self):
        """Attempts to initialize connection to MongoDB."""
        is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("NOW_REGION"))
        if is_serverless and ("localhost" in self.mongo_url or "127.0.0.1" in self.mongo_url):
            logger.info("Serverless environment detected without external MongoDB URL. Using resilient /tmp storage.")
            self.use_mongo = False
            self._ensure_local_dirs()
            return

        try:
            import pymongo
            self.client = pymongo.MongoClient(self.mongo_url, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
            # Test connection with ping
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.use_mongo = True
            safe_url = self.mongo_url.split("@")[-1] if "@" in self.mongo_url else self.mongo_url
            logger.info(f"Connected to MongoDB server at {safe_url} (Database: {self.db_name})")
        except Exception as e:
            logger.info(f"MongoDB cloud/local connection notice ({e}). Operating in resilient local storage mode.")
            self.use_mongo = False
            self._ensure_local_dirs()

    def _ensure_local_dirs(self):
        """Creates directory structure for local persistence (using /tmp on serverless or read-only filesystem)."""
        is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("NOW_REGION"))
        import tempfile
        
        if is_serverless:
            tmp_dir = os.path.join(tempfile.gettempdir(), "lexibrief_db")
            os.makedirs(tmp_dir, exist_ok=True)
            self.summaries_file = os.path.join(tmp_dir, "summaries.json")
            self.documents_file = os.path.join(tmp_dir, "documents.json")
        else:
            try:
                local_dir = os.path.join("artifacts", "database")
                os.makedirs(local_dir, exist_ok=True)
                test_file = os.path.join(local_dir, ".test_write")
                with open(test_file, "w") as tf:
                    tf.write("ok")
                os.remove(test_file)
                self.summaries_file = os.path.join(local_dir, "summaries.json")
                self.documents_file = os.path.join(local_dir, "documents.json")
            except (OSError, PermissionError):
                tmp_dir = os.path.join(tempfile.gettempdir(), "lexibrief_db")
                os.makedirs(tmp_dir, exist_ok=True)
                self.summaries_file = os.path.join(tmp_dir, "summaries.json")
                self.documents_file = os.path.join(tmp_dir, "documents.json")

        for f in [self.summaries_file, self.documents_file]:
            try:
                if not os.path.exists(f):
                    with open(f, "w", encoding="utf-8") as fp:
                        json.dump([], fp)
            except Exception:
                pass

    # ------------------ SUMMARIES COLLECTION ------------------

    def save_summary(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Saves a generated summary to the 'summaries' collection."""
        title = summary_data.get("title")
        if not title:
            raw_text = summary_data.get("text", "").strip()
            first_line = raw_text.split('\n')[0].strip() if raw_text else ""
            title = first_line[:60] if first_line else "Document Summary"

        orig_words = summary_data.get("analytics", {}).get("original_words") or summary_data.get("word_count_original") or len(summary_data.get("text", "").split())
        sum_words = summary_data.get("analytics", {}).get("summary_words") or summary_data.get("word_count_summary") or len(summary_data.get("summary", "").split())

        record = {
            "_id": str(uuid.uuid4()),
            "title": title,
            "text_snippet": summary_data.get("text", "")[:150] + ("..." if len(summary_data.get("text", "")) > 150 else ""),
            "full_text": summary_data.get("text", ""),
            "summary": summary_data.get("summary", ""),
            "key_points": summary_data.get("key_points", []),
            "keywords": summary_data.get("keywords", []),
            "mode": summary_data.get("mode", "balanced"),
            "method": summary_data.get("method", summary_data.get("method_used", "Auto")),
            "method_used": summary_data.get("method_used", "Auto"),
            "model_source": summary_data.get("model_source", "hybrid"),
            "rouge": summary_data.get("rouge", {}),
            "word_count_original": orig_words,
            "word_count_summary": sum_words,
            "compression_ratio": summary_data.get("analytics", {}).get("compression_ratio", "0%"),
            "latency_ms": summary_data.get("analytics", {}).get("latency_ms", 0),
            "created_at": datetime.now(timezone.utc).strftime("%B %d, %Y")
        }

        if self.use_mongo and self.db is not None:
            try:
                self.db.summaries.insert_one(record)
                return record
            except Exception as e:
                logger.warning(f"MongoDB write failed ({e}), saving to local storage.")

        # Local fallback write
        try:
            self._ensure_local_dirs()
            with open(self.summaries_file, "r", encoding="utf-8") as fp:
                items = json.load(fp)
            items.insert(0, record)
            with open(self.summaries_file, "w", encoding="utf-8") as fp:
                json.dump(items[:200], fp, indent=2)
        except Exception as err:
            logger.error(f"Local summary save error: {err}")

        return record

    def get_summaries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent summaries from the 'summaries' collection."""
        if self.use_mongo and self.db is not None:
            try:
                cursor = self.db.summaries.find({}, {"full_text": 0}).sort("created_at", -1).limit(limit)
                return list(cursor)
            except Exception as e:
                logger.warning(f"MongoDB read failed ({e}), reading from local storage.")

        # Local fallback read
        try:
            self._ensure_local_dirs()
            with open(self.summaries_file, "r", encoding="utf-8") as fp:
                items = json.load(fp)
            return items[:limit]
        except Exception:
            return []

    def get_summary_by_id(self, summary_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific summary by ID."""
        if self.use_mongo and self.db is not None:
            try:
                return self.db.summaries.find_one({"_id": summary_id})
            except Exception:
                pass
        
        try:
            self._ensure_local_dirs()
            with open(self.summaries_file, "r", encoding="utf-8") as fp:
                items = json.load(fp)
            for item in items:
                if item.get("_id") == summary_id:
                    return item
        except Exception:
            pass
        return None

    def delete_summary(self, summary_id: str) -> bool:
        """Deletes a summary by ID."""
        deleted = False
        if self.use_mongo and self.db is not None:
            try:
                res = self.db.summaries.delete_one({"_id": summary_id})
                deleted = res.deleted_count > 0
            except Exception:
                pass

        try:
            self._ensure_local_dirs()
            with open(self.summaries_file, "r", encoding="utf-8") as fp:
                items = json.load(fp)
            new_items = [it for it in items if it.get("_id") != summary_id]
            with open(self.summaries_file, "w", encoding="utf-8") as fp:
                json.dump(new_items, fp, indent=2)
            deleted = True
        except Exception:
            pass
        return deleted

    # ------------------ DOCUMENTS COLLECTION ------------------

    def save_document(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Saves an uploaded document to the 'documents' collection."""
        record = {
            "_id": str(uuid.uuid4()),
            "filename": doc_data.get("filename", "untitled.txt"),
            "format": doc_data.get("format", "TXT"),
            "text_snippet": (doc_data.get("text", "")[:220] + ("..." if len(doc_data.get("text", "")) > 220 else "")),
            "text_content": doc_data.get("text", ""),
            "words": doc_data.get("stats", {}).get("words", 0),
            "characters": doc_data.get("stats", {}).get("characters", 0),
            "sentences": doc_data.get("stats", {}).get("sentences", 0),
            "readability_score": doc_data.get("stats", {}).get("readability_score", 0),
            "keywords": [k.get("keyword", "") if isinstance(k, dict) else str(k) for k in doc_data.get("keywords", [])],
            "uploaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        }

        if self.use_mongo and self.db is not None:
            try:
                self.db.documents.insert_one(record)
                return record
            except Exception as e:
                logger.warning(f"MongoDB document save failed: {e}")

        # Local fallback write
        try:
            self._ensure_local_dirs()
            with open(self.documents_file, "r", encoding="utf-8") as fp:
                items = json.load(fp)
            items.insert(0, record)
            with open(self.documents_file, "w", encoding="utf-8") as fp:
                json.dump(items[:200], fp, indent=2)
        except Exception as err:
            logger.error(f"Local document save error: {err}")

        return record

    def get_documents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves uploaded documents from the 'documents' collection."""
        if self.use_mongo and self.db is not None:
            try:
                cursor = self.db.documents.find({}, {"text_content": 0}).sort("uploaded_at", -1).limit(limit)
                docs = list(cursor)
                for d in docs:
                    if not d.get("text_snippet") and d.get("_id"):
                        try:
                            full = self.db.documents.find_one({"_id": d["_id"]}, {"text_content": 1})
                            if full and full.get("text_content"):
                                snippet = full["text_content"][:220] + ("..." if len(full["text_content"]) > 220 else "")
                                d["text_snippet"] = snippet
                                self.db.documents.update_one({"_id": d["_id"]}, {"$set": {"text_snippet": snippet}})
                        except Exception:
                            pass
                return docs
            except Exception as e:
                logger.warning(f"MongoDB document read failed: {e}")

        # Local fallback read
        try:
            self._ensure_local_dirs()
            with open(self.documents_file, "r", encoding="utf-8") as fp:
                items = json.load(fp)
            for it in items:
                if not it.get("text_snippet") and it.get("text_content"):
                    it["text_snippet"] = it["text_content"][:220] + ("..." if len(it["text_content"]) > 220 else "")
            return items[:limit]
        except Exception:
            return []

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves document including full text by ID."""
        if self.use_mongo and self.db is not None:
            try:
                return self.db.documents.find_one({"_id": doc_id})
            except Exception:
                pass

        try:
            self._ensure_local_dirs()
            with open(self.documents_file, "r", encoding="utf-8") as fp:
                items = json.load(fp)
            for it in items:
                if it.get("_id") == doc_id:
                    return it
        except Exception:
            pass
        return None

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document by ID."""
        deleted = False
        if self.use_mongo and self.db is not None:
            try:
                res = self.db.documents.delete_one({"_id": doc_id})
                deleted = res.deleted_count > 0
            except Exception:
                pass

        try:
            self._ensure_local_dirs()
            with open(self.documents_file, "r", encoding="utf-8") as fp:
                items = json.load(fp)
            new_items = [it for it in items if it.get("_id") != doc_id]
            with open(self.documents_file, "w", encoding="utf-8") as fp:
                json.dump(new_items, fp, indent=2)
            deleted = True
        except Exception:
            pass
        return deleted

    def get_database_status(self) -> Dict[str, Any]:
        """Returns current database connectivity and collection counts."""
        summaries = self.get_summaries()
        documents = self.get_documents()
        return {
            "engine": "MongoDB (Atlas Cloud / Local)" if self.use_mongo else "MongoDB (Resilient Local Engine)",
            "status": "connected" if self.use_mongo else "active",
            "database": self.db_name,
            "collections": {
                "summaries": len(summaries),
                "documents": len(documents)
            }
        }
