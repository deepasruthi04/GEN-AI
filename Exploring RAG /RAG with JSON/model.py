import os
import json
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents import Document

persist_directory = "./db/chroma"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def fetch_and_deduplicate():
    mongo_uri = "mongodb://localhost:27017/"
    client = MongoClient(mongo_uri)
    db = client["IOT"]

    collections = db.list_collection_names()
    print("Collections found:", collections)
    

    all_docs = []
    for cname in collections:
        coll = db[cname]
        for doc in coll.find():
            doc = dict(doc)
            doc["_id"] = str(doc.get("_id"))
            doc["_collection"] = cname
            all_docs.append(doc)

    print(f"Total documents fetched: {len(all_docs)}")
    return deduplicate_by_timeseries(all_docs)


def deduplicate_by_timeseries(docs):
    seen = {}
    for doc in docs:
        ts = doc.get("timeseries")
        if ts is None:
            continue

        existing = seen.get(ts)
        has_cycle_times = "cycleStartTime" in doc and "cycleEndTime" in doc

        if existing is None:
            seen[ts] = doc
        else:
            existing_has_cycle_times = "cycleStartTime" in existing and "cycleEndTime" in existing
            if has_cycle_times and not existing_has_cycle_times:
                seen[ts] = doc

    return list(seen.values())


def format_documents(docs):
    formatted = []
    for doc in docs:
        formatted_doc = {
            "id": doc.get("_id"),
            "collection": doc.get("_collection"),
            "timeseries": doc.get("timeseries"),
            "cycleStartTime": doc.get("cycleStartTime"),
            "cycleEndTime": doc.get("cycleEndTime"),
            "data": doc
        }
        formatted.append(formatted_doc)
    return formatted


def flatten_document(doc, parent_key='', sep='.'):
    items = {}
    for k, v in doc.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_document(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def normalize_documents(formatted_docs, datetime_fields=None):
    if datetime_fields is None:
        datetime_fields = ["timeseries", "cycleStartTime", "cycleEndTime"]

    normalized = []
    for doc in formatted_docs:
        flat_doc = flatten_document(doc["data"])
        flat_doc["id"] = doc.get("id")
        flat_doc["collection"] = doc.get("collection")
        flat_doc["timeseries"] = doc.get("timeseries")
        flat_doc["cycleStartTime"] = doc.get("cycleStartTime")
        flat_doc["cycleEndTime"] = doc.get("cycleEndTime")

        for time_field in datetime_fields:
            if time_field in flat_doc:
                try:
                    flat_doc[time_field] = datetime.fromisoformat(str(flat_doc[time_field])).isoformat()
                except Exception:
                    pass

        normalized.append(flat_doc)
    return normalized


def doc_to_text(normalized_doc, exclude_keys=None):
    exclude = set(exclude_keys or ["id", "collection"])
    parts = []
    for k, v in normalized_doc.items():
        if k in exclude or v is None:
            continue
        parts.append(f"{k}: {v}")
    return " | ".join(parts)

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

def create_vector_db(normalized_docs, formatted_docs):
    print("\n💾 [STEP] Creating / updating vector DB...")

    texts = []
    metadata_list = []


    for norm_doc, formatted_doc in zip(normalized_docs, formatted_docs):
        text = doc_to_text(norm_doc)
        if not text.strip():
            continue
        texts.append(text)

        metadata = {
            "doc_id": formatted_doc.get("id"),
            "collection": formatted_doc.get("collection"),
            "timeseries": formatted_doc.get("timeseries"),
            "cycleStartTime": formatted_doc.get("cycleStartTime"),
            "cycleEndTime": formatted_doc.get("cycleEndTime"),
        }

        for key, value in metadata.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                metadata[key] = str(value)

        metadata_list.append(metadata)

    documents = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(texts, metadata_list)
    ]

    print(f" Number of documents to index in Chroma: {len(documents)}")

    documents = filter_complex_metadata(documents)

    os.makedirs(persist_directory, exist_ok=True)
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    vectordb.persist()

    print("✅ Vector DB successfully created and saved to disk.")

    docs_found = vectordb.similarity_search("test query", k=2)
    print(f"🔍 Test search found {len(docs_found)} documents.")
    for i, doc in enumerate(docs_found, 1):
        print(f"--- Document {i} ---")
        print("Metadata:", doc.metadata)
        print("Preview:", doc.page_content[:150])

    return vectordb



if __name__ == "__main__":
    print(" Starting full pipeline execution...\n")

    deduped = fetch_and_deduplicate()
    formatted = format_documents(deduped)
    normalized = normalize_documents(formatted)
    vectordb = create_vector_db(normalized, formatted)

    print("\n Pipeline complete!")



 