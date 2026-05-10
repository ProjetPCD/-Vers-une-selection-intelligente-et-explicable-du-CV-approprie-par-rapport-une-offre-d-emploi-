# backend/db/mongo.py
import os
from datetime import datetime
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME   = "match_db"

_client: Optional[AsyncIOMotorClient] = None

def get_db():
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URL)
    return _client[DB_NAME]

def jobs_col():         return get_db()["jobs"]
def results_col():      return get_db()["results"]
def phases_col():       return get_db()["phases"]        # ← CORRIGÉ (était "match_db")
def output_files_col(): return get_db()["output_files"]

# ══ JOBS ══════════════════════════════════════════════════════════════════════

async def save_job(jd_text: str, cv_texts: list[str]) -> str:
    doc = {
        "job_description": jd_text,
        "cvs": cv_texts,
        "created_at": datetime.utcnow(),
        "status": "pending",
        "current_phase": None,
        "progress": None
    }
    res = await jobs_col().insert_one(doc)
    return str(res.inserted_id)

async def get_job(job_id: str) -> Optional[dict]:
    doc = await jobs_col().find_one({"_id": ObjectId(job_id)})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc

async def update_job_status(
    job_id: str,
    status: str,
    current_phase: str = None,
    progress: str = None,
    summary: str = None
):
    update_fields = {"status": status, "updated_at": datetime.utcnow()}
    if current_phase is not None:
        update_fields["current_phase"] = current_phase
    if progress is not None:
        update_fields["progress"] = progress
    if summary is not None:
        update_fields["summary"] = summary
    await jobs_col().update_one(
        {"_id": ObjectId(job_id)},
        {"$set": update_fields}
    )

# ══ RESULTS ══════════════════════════════════════════════════════════════════

async def save_result(job_id: str, cv_index: int, report: dict):
    doc = {
        "job_id":    job_id,
        "cv_index":  cv_index,
        "pair_key":  f"{job_id}_cv_{cv_index + 1}",
        "report":    report,
        "created_at": datetime.utcnow()
    }
    await results_col().replace_one(
        {"job_id": job_id, "cv_index": cv_index},
        doc,
        upsert=True
    )

async def get_results_by_job(job_id: str) -> list[dict]:
    cursor = results_col().find({"job_id": job_id})
    docs   = await cursor.to_list(length=100)
    for d in docs:
        d["_id"] = str(d["_id"])
    return sorted(
        docs,
        key=lambda d: d.get("report", {}).get("pool_ranking", {}).get("rank", 9999)
    )

# ══ PHASES INTERMÉDIAIRES ════════════════════════════════════════════════════

async def save_phase_data(job_id: str, phase: str, pair_key: str, data: dict):
    """Sauvegarde les données intermédiaires (graph, phase2, phase3) dans MongoDB."""
    doc = {
        "job_id":     job_id,
        "phase":      phase,
        "pair_key":   pair_key,
        "data":       data,
        "created_at": datetime.utcnow()
    }
    await phases_col().replace_one(
        {"job_id": job_id, "phase": phase, "pair_key": pair_key},
        doc,
        upsert=True
    )

async def get_phase_data(job_id: str, phase: str, pair_key: str = None) -> Optional[dict]:
    """Récupère les données intermédiaires."""
    query = {"job_id": job_id, "phase": phase}
    if pair_key:
        query["pair_key"] = pair_key
    doc = await phases_col().find_one(query)
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc

# ══ OUTPUT FILES ══════════════════════════════════════════════════════════════

async def save_output_file_meta(
    job_id: str,
    category: str,
    filename: str,
    rel_path: str,
    size: int
):
    doc = {
        "job_id":     job_id,
        "category":   category,
        "filename":   filename,
        "file_path":  rel_path,
        "size_bytes": size,
        "created_at": datetime.utcnow()
    }
    await output_files_col().replace_one(
        {"job_id": job_id, "category": category, "filename": filename},
        doc,
        upsert=True
    )

async def get_output_files_for_job(job_id: str) -> list[dict]:
    cursor = output_files_col().find({"job_id": job_id})
    docs   = await cursor.to_list(length=100)
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs

# ══ SANTÉ ═════════════════════════════════════════════════════════════════════

async def ping_db() -> bool:
    try:
        await get_db().command("ping")
        return True
    except Exception:
        return False