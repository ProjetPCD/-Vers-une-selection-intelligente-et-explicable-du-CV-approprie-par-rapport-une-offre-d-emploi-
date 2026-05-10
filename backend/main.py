# backend/main.py
import asyncio
import json
import logging
import os
from typing import List
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from backend.db.mongo import (
    save_job, get_job, update_job_status,
    save_result, get_results_by_job,
    save_phase_data, get_phase_data,
    save_output_file_meta, get_output_files_for_job,
    ping_db
)
from backend.utils.pdf_extractor import extract_text_from_pdf
from backend.pipeline import run_full_pipeline, GRAPHS_STORE, PHASE2_STORE, PHASE3_STORE, BIAS_STORE, REPORTS_STORE

app = FastAPI(title="CV Matcher API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

BASE_DIR    = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"


# ══ DÉMARRAGE ════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup_event():
    ok = await ping_db()
    if not ok:
        logger.error("❌ ATTENTION : MongoDB inaccessible au démarrage !")
    else:
        logger.info("✅ MongoDB connecté au démarrage.")

    from backend.db.mongo import jobs_col
    try:
        res = await jobs_col().update_many(
            {"status": "running"},
            {"$set": {
                "status": "error",
                "current_phase": "Interrompu",
                "progress": "Le serveur a redémarré pendant le traitement. Veuillez soumettre à nouveau."
            }}
        )
        if res.modified_count > 0:
            logger.info(f"🧹 Nettoyage : {res.modified_count} job(s) interrompus marqués en erreur.")
    except Exception as e:
        logger.error(f"⚠ Erreur lors du nettoyage au démarrage : {e}")


# ══ SCHÉMAS ══════════════════════════════════════════════════════════════════
class SubmitResponse(BaseModel):
    job_id: str
    message: str = "Pipeline lancé en arrière-plan"

class StatusResponse(BaseModel):
    job_id: str
    status: str
    current_phase: Optional[str] = None
    progress:      Optional[str] = None
    summary:       Optional[str] = None

class CVRankItem(BaseModel):
    cv_index:  int
    pair_key:  str
    rank:      Optional[int]   = None
    tier:      Optional[str]   = None
    verdict:   Optional[str]   = None
    match_pct: Optional[float] = None
    combo_v6:  Optional[float] = None
    confidence:Optional[str]   = None

class ResultsResponse(BaseModel):
    job_id: str
    rankings: list[CVRankItem]


# ══ PIPELINE BACKGROUND ══════════════════════════════════════════════════════
#
# FIX DÉFINITIF — Architecture :
#
#   1.  run_full_pipeline() tourne dans asyncio.to_thread() → retourne (reports, summary)
#   2.  On reçoit `reports` directement — pas de dépendance aux STOREs mémoire
#   3.  On sauvegarde dans MongoDB ET sur disque
#   4.  On vérifie MongoDB (max 10 tentatives)
#   5.  On marque "done" SEULEMENT après confirmation
#   6.  Si MongoDB échoue, on sauvegarde dans un cache local IN-MEMORY indexé par job_id
#       → /api/results lit ce cache en fallback immédiat

# Cache local garanti disponible tant que le process tourne
_LOCAL_RESULTS_CACHE: dict[str, list[dict]] = {}


async def _run_pipeline_bg(job_id: str, jd_text: str, cv_texts: list[str]):
    try:
        await update_job_status(job_id, "running",
                                current_phase="Phase 1",
                                progress="Traitement en cours...")
        logger.info(f"[Pipeline] 🚀 Démarrage job={job_id} ({len(cv_texts)} CV(s))")

        # ── ÉTAPE 1 : Exécuter le pipeline (bloquant, dans un thread) ─────
        # On reçoit `reports` directement — c'est la source de vérité.
        reports, summary = await asyncio.to_thread(
            _run_pipeline_and_save_files, job_id, jd_text, cv_texts
        )
        logger.info(f"[Pipeline] ✅ {len(reports)} rapport(s) reçus du thread")

        # ── ÉTAPE 2 : Stocker dans le cache local IMMÉDIATEMENT ───────────
        # Ce cache est disponible dès maintenant, avant même MongoDB.
        cache_docs = []
        for pair_key, report in reports.items():
            try:
                cv_index = int(pair_key.split("_cv_")[-1]) - 1
                cache_docs.append({
                    "cv_index": cv_index,
                    "pair_key": pair_key,
                    "report":   report
                })
            except Exception as e:
                logger.error(f"[Cache] ❌ parse pair_key={pair_key} : {e}")
        _LOCAL_RESULTS_CACHE[job_id] = cache_docs
        logger.info(f"[Cache] ✅ {len(cache_docs)} résultat(s) stockés en cache local")

        # ── ÉTAPE 3 : Sauvegarder dans MongoDB ────────────────────────────
        mongo_ok = False
        for pair_key, report in reports.items():
            cv_index = int(pair_key.split("_cv_")[-1]) - 1
            try:
                await save_result(job_id, cv_index, report)
                mongo_ok = True
            except Exception as e:
                logger.error(f"[Pipeline] ❌ save_result cv={cv_index} : {e}")

        # Phases → MongoDB
        if job_id in GRAPHS_STORE:
            try:
                await save_phase_data(job_id, "phase1", job_id, GRAPHS_STORE[job_id])
            except Exception as e:
                logger.error(f"[Pipeline] ❌ save phase1 : {e}")

        for pair_key, p2 in PHASE2_STORE.items():
            if pair_key.startswith(job_id):
                try:
                    await save_phase_data(job_id, "phase2", pair_key, p2)
                except Exception as e:
                    logger.error(f"[Pipeline] ❌ save phase2 {pair_key} : {e}")

        for pair_key, p3 in PHASE3_STORE.items():
            if pair_key.startswith(job_id):
                try:
                    await save_phase_data(job_id, "phase3", pair_key, p3)
                except Exception as e:
                    logger.error(f"[Pipeline] ❌ save phase3 {pair_key} : {e}")

        # Métadonnées fichiers → MongoDB
        for subdir in ["inputs", "graphs", "phase2", "phase3", "reports", "bias", "results"]:
            dirpath = OUTPUTS_DIR / subdir
            if dirpath.exists():
                for fp in dirpath.iterdir():
                    if job_id in fp.name and fp.is_file():
                        try:
                            rel_path = f"/api/output-file/{subdir}/{fp.name}"
                            await save_output_file_meta(
                                job_id, subdir, fp.name, rel_path, fp.stat().st_size
                            )
                        except Exception as e:
                            logger.error(f"[Fichier] ❌ {fp.name} : {e}")

        # ── ÉTAPE 4 : Vérifier MongoDB (sans bloquer la progression) ──────
        if mongo_ok:
            confirmed = False
            for attempt in range(10):
                check = await get_results_by_job(job_id)
                if check:
                    logger.info(f"[Pipeline] ✅ MongoDB confirmé — {len(check)} résultat(s) (tentative {attempt+1})")
                    confirmed = True
                    break
                logger.warning(f"[Pipeline] ⏳ MongoDB pas encore flushé (tentative {attempt+1}/10)...")
                await asyncio.sleep(1.0)
            if not confirmed:
                logger.warning(f"[Pipeline] ⚠ MongoDB non confirmé — cache local disponible en fallback")

        # ── ÉTAPE 5 : Marquer DONE — le cache est déjà prêt ──────────────
        await update_job_status(job_id, "done",
                                current_phase="Terminé",
                                progress="Pipeline terminé",
                                summary=summary)
        logger.info(f"🏁 Job {job_id} marqué DONE — cache={len(cache_docs)} résultats")

    except Exception as e:
        logger.error(f"[Pipeline ERROR] job={job_id} : {e}")
        import traceback; traceback.print_exc()
        await update_job_status(job_id, "error",
                                current_phase="Erreur",
                                progress=str(e)[:200])


def _run_pipeline_and_save_files(job_id: str, jd_text: str, cv_texts: list[str]):
    """Exécuté dans un thread séparé. Retourne (reports, summary) directement."""
    reports, summary = run_full_pipeline(job_id, jd_text, cv_texts)

    # Graphe Phase 1
    if job_id in GRAPHS_STORE:
        p = OUTPUTS_DIR / "graphs" / f"{job_id}_graph.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(GRAPHS_STORE[job_id], f, ensure_ascii=False, indent=2)

    # Phase 2
    for pk, data in PHASE2_STORE.items():
        if pk.startswith(job_id):
            p = OUTPUTS_DIR / "phase2" / f"{pk}_phase2.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    # Phase 3
    for pk, data in PHASE3_STORE.items():
        if pk.startswith(job_id):
            p = OUTPUTS_DIR / "phase3" / f"{pk}_phase3.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    # Rapports
    for pk, rpt in reports.items():
        p = OUTPUTS_DIR / "reports" / f"{pk}_report.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rpt, f, ensure_ascii=False, indent=2, default=str)

    # Résumé résultats
    try:
        p = OUTPUTS_DIR / "results" / f"{job_id}_results_summary.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        lst = sorted([
            {
                "cv_index":  int(pk.split("_cv_")[-1]) - 1,
                "pair_key":  pk,
                "rank":      rpt.get("pool_ranking", {}).get("rank"),
                "tier":      rpt.get("pool_ranking", {}).get("tier"),
                "verdict":   rpt.get("overall_assessment", {}).get("verdict"),
                "match_pct": rpt.get("overall_assessment", {}).get("match_pct"),
            }
            for pk, rpt in reports.items()
        ], key=lambda x: x.get("rank") or 9999)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"job_id": job_id, "rankings": lst}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[Pipeline] ❌ Résumé résultats : {e}")

    return reports, summary


# ══ ENDPOINT 1 — Soumettre ════════════════════════════════════════════════════
@app.post("/api/match", response_model=SubmitResponse)
async def submit_match(
    background_tasks: BackgroundTasks,
    job_description: str = Form(...),
    cvs: List[UploadFile] = File(...),
):
    if not job_description.strip():
        raise HTTPException(400, "job_description vide")

    cv_texts = []
    for upload in cvs:
        raw  = await upload.read()
        text = extract_text_from_pdf(raw)
        if len(text.strip()) < 50:
            raise HTTPException(422,
                f"CV '{upload.filename}' illisible ou trop court ({len(text.strip())} caractères)")
        cv_texts.append(text)

    job_id = await save_job(job_description, cv_texts)
    logger.info(f"[Soumission] ✅ Job {job_id} créé dans MongoDB ({len(cv_texts)} CV(s))")

    inputs_dir = OUTPUTS_DIR / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    jd_filename = f"{job_id}_offre.txt"
    jd_path = inputs_dir / jd_filename
    try:
        jd_path.write_text(job_description, encoding="utf-8")
        await save_output_file_meta(
            job_id, "inputs", jd_filename,
            f"/api/output-file/inputs/{jd_filename}",
            jd_path.stat().st_size
        )
    except Exception as e:
        logger.warning(f"[Soumission] ⚠ Sauvegarde offre : {e}")

    for i, text in enumerate(cv_texts):
        cv_filename = f"{job_id}_cv_{i + 1}.txt"
        cv_path = inputs_dir / cv_filename
        try:
            cv_path.write_text(text, encoding="utf-8")
            await save_output_file_meta(
                job_id, "inputs", cv_filename,
                f"/api/output-file/inputs/{cv_filename}",
                cv_path.stat().st_size
            )
        except Exception as e:
            logger.warning(f"[Soumission] ⚠ Sauvegarde CV {i+1} : {e}")

    background_tasks.add_task(_run_pipeline_bg, job_id, job_description, cv_texts)
    return SubmitResponse(job_id=job_id)


# ══ ENDPOINT 2 — Statut ══════════════════════════════════════════════════════
@app.get("/api/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")
    return StatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        current_phase=job.get("current_phase"),
        progress=job.get("progress"),
        summary=job.get("summary")
    )


# ══ ENDPOINT 3 — Résultats ════════════════════════════════════════════════════
@app.get("/api/results/{job_id}", response_model=ResultsResponse)
async def get_results(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")
    if job.get("status") != "done":
        raise HTTPException(202, f"En cours ({job.get('status')})")

    docs = []

    # ── SOURCE 1 : MongoDB (le plus fiable) ──────────────────────────────
    docs = await get_results_by_job(job_id)
    if docs:
        logger.info(f"[Résultats] ✅ MongoDB → {len(docs)} résultat(s)")

    # ── SOURCE 2 : Cache local en mémoire (disponible immédiatement après pipeline) ──
    if not docs and job_id in _LOCAL_RESULTS_CACHE:
        docs = _LOCAL_RESULTS_CACHE[job_id]
        logger.info(f"[Résultats] ✅ Cache local → {len(docs)} résultat(s)")

    # ── SOURCE 3 : Disque (résiste aux redémarrages) ──────────────────────
    if not docs:
        reports_dir = OUTPUTS_DIR / "reports"
        if reports_dir.exists():
            logger.warning(f"[Résultats] MongoDB + cache vides → lecture disque")
            for fp in sorted(reports_dir.iterdir()):
                if fp.name.startswith(job_id) and fp.suffix == ".json" and "_report" in fp.name:
                    try:
                        stem = fp.stem.replace("_report", "")
                        cv_index = int(stem.split("_cv_")[-1]) - 1
                        pk = f"{job_id}_cv_{cv_index + 1}"
                        rpt = json.loads(fp.read_text(encoding="utf-8"))
                        docs.append({"cv_index": cv_index, "pair_key": pk, "report": rpt})
                        logger.info(f"[Résultats] ✅ Disque : {fp.name}")
                    except Exception as e:
                        logger.error(f"[Résultats] ⚠ Lecture {fp.name} : {e}")

    # ── SOURCE 4 : REPORTS_STORE pipeline (dernier recours) ───────────────
    if not docs and REPORTS_STORE:
        logger.warning(f"[Résultats] Disque vide → REPORTS_STORE")
        for pk, rpt in REPORTS_STORE.items():
            if pk.startswith(job_id):
                try:
                    cv_index = int(pk.split("_cv_")[-1]) - 1
                    docs.append({"cv_index": cv_index, "pair_key": pk, "report": rpt})
                except Exception:
                    pass

    if not docs:
        logger.error(f"[Résultats] ❌ Aucune source disponible pour job={job_id}")
        raise HTTPException(404, "Aucun résultat trouvé pour ce job")

    docs = sorted(docs, key=lambda d: d.get("report", {}).get("pool_ranking", {}).get("rank", 9999))
    rankings = []
    for doc in docs:
        r   = doc.get("report", {})
        pr  = r.get("pool_ranking", {})
        oa  = r.get("overall_assessment", {})
        cv6 = r.get("combo_scores_v6", {}).get("combo_v6_final", {})
        rankings.append(CVRankItem(
            cv_index=doc["cv_index"],
            pair_key=doc["pair_key"],
            rank=pr.get("rank"),
            tier=pr.get("tier"),
            verdict=oa.get("verdict"),
            match_pct=oa.get("match_pct"),
            combo_v6=float(cv6.get("value", 0)) if isinstance(cv6, dict) else float(cv6 or 0),
            confidence=oa.get("confidence", {}).get("confidence_label")
        ))
    return ResultsResponse(job_id=job_id, rankings=rankings)


# ══ ENDPOINT 4 — Graphe JD ═══════════════════════════════════════════════════
def _normalize_graph(data: dict | list) -> dict:
    if isinstance(data, list):
        data = data[0] if data else {}
    if isinstance(data, dict) and 'graph' in data:
        data = data['graph']
    if isinstance(data, dict):
        if 'node' in data and 'nodes' not in data:
            data['nodes'] = data.pop('node')
        if 'edge' in data and 'edges' not in data:
            data['edges'] = data.pop('edge')
    return data

@app.get("/api/graph/{job_id}")
async def get_graph(job_id: str):
    if job_id in GRAPHS_STORE:
        return JSONResponse(_normalize_graph(GRAPHS_STORE[job_id]))
    graph_path = OUTPUTS_DIR / "graphs" / f"{job_id}_graph.json"
    if graph_path.exists():
        with open(graph_path, "r", encoding="utf-8") as f:
            return JSONResponse(_normalize_graph(json.load(f)))
    db_data = await get_phase_data(job_id, "phase1", job_id)
    if db_data:
        return JSONResponse(_normalize_graph(db_data.get("data", {})))
    job = await get_job(job_id)
    if job:
        return JSONResponse({"status": "processing", "message": "Graphe en cours...", "job_id": job_id},
                            status_code=202)
    raise HTTPException(404, "Graphe JD non trouvé")


# ══ ENDPOINT 5 — Phase 2 ═════════════════════════════════════════════════════
@app.get("/api/phase2/{job_id}/{cv_index}")
async def get_phase2(job_id: str, cv_index: int):
    pk = f"{job_id}_cv_{cv_index + 1}"
    if pk in PHASE2_STORE:
        data = PHASE2_STORE[pk]
        safe = {k: v for k, v in data.items() if k not in ("cv_sections",)}
        safe["cv_sections"] = {k: v[:500] for k, v in data.get("cv_sections", {}).items()}
        return JSONResponse(safe)
    p = OUTPUTS_DIR / "phase2" / f"{pk}_phase2.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return JSONResponse(json.load(f))
    db_data = await get_phase_data(job_id, "phase2", pk)
    if db_data:
        return JSONResponse(db_data.get("data", {}))
    job = await get_job(job_id)
    if job:
        return JSONResponse({"status": "processing", "message": "Phase 2 en cours..."}, status_code=202)
    raise HTTPException(404, "Données Phase 2 non trouvées")


# ══ ENDPOINT 6 — Phase 3 ═════════════════════════════════════════════════════
@app.get("/api/phase3/{job_id}/{cv_index}")
async def get_phase3(job_id: str, cv_index: int):
    pk = f"{job_id}_cv_{cv_index + 1}"
    if pk in PHASE3_STORE:
        return JSONResponse(PHASE3_STORE[pk])
    p = OUTPUTS_DIR / "phase3" / f"{pk}_phase3.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return JSONResponse(json.load(f))
    db_data = await get_phase_data(job_id, "phase3", pk)
    if db_data:
        return JSONResponse(db_data.get("data", {}))
    job = await get_job(job_id)
    if job:
        return JSONResponse({"status": "processing", "message": "Phase 3 en cours..."}, status_code=202)
    raise HTTPException(404, "Données Phase 3 non trouvées")


# ══ ENDPOINT 7 — Rapport complet ══════════════════════════════════════════════
@app.get("/api/report/{job_id}/{cv_index}")
async def get_report(job_id: str, cv_index: int):
    pk = f"{job_id}_cv_{cv_index + 1}"
    if pk in REPORTS_STORE:
        return JSONResponse(REPORTS_STORE[pk])
    p = OUTPUTS_DIR / "reports" / f"{pk}_report.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return JSONResponse(json.load(f))
    # Chercher dans le cache local
    if job_id in _LOCAL_RESULTS_CACHE:
        for doc in _LOCAL_RESULTS_CACHE[job_id]:
            if doc["cv_index"] == cv_index:
                return JSONResponse(doc["report"])
    results = await get_results_by_job(job_id)
    for r in results:
        if r["cv_index"] == cv_index:
            return JSONResponse(r.get("report", {}))
    job = await get_job(job_id)
    if job:
        return JSONResponse({"status": "processing", "message": "Rapport en cours..."}, status_code=202)
    raise HTTPException(404, "Rapport non trouvé")


# ══ ENDPOINT 8 — Liste des fichiers générés ══════════════════════════════════
@app.get("/api/outputs/{job_id}")
async def list_outputs(job_id: str):
    files = []
    try:
        mongo_files = await get_output_files_for_job(job_id)
        if mongo_files:
            for mf in mongo_files:
                files.append({
                    "name":     mf["filename"],
                    "category": mf["category"],
                    "size":     mf["size_bytes"],
                    "path":     mf["file_path"],
                })
            return {"job_id": job_id, "files": files}
    except Exception as e:
        logger.warning(f"[Outputs] MongoDB échoué, fallback disque : {e}")

    for subdir in ["inputs", "graphs", "phase2", "phase3", "reports", "bias", "results"]:
        dirpath = OUTPUTS_DIR / subdir
        if dirpath.exists():
            for fp in dirpath.iterdir():
                if job_id in fp.name and fp.is_file():
                    files.append({
                        "name":     fp.name,
                        "category": subdir,
                        "size":     fp.stat().st_size,
                        "path":     f"/api/output-file/{subdir}/{fp.name}"
                    })
    return {"job_id": job_id, "files": files}


# ══ ENDPOINT 9 — Télécharger un fichier ══════════════════════════════════════
@app.get("/api/output-file/{category}/{filename}")
async def get_output_file(category: str, filename: str):
    safe_cats = {"inputs", "graphs", "phase2", "phase3", "reports", "bias", "results", "checkpoints"}
    if category not in safe_cats:
        raise HTTPException(400, "Catégorie invalide")
    fpath = OUTPUTS_DIR / category / filename
    if not fpath.exists() or not fpath.is_file():
        raise HTTPException(404, "Fichier non trouvé")
    media = "text/plain" if fpath.suffix == ".txt" else "application/json"
    return FileResponse(str(fpath), filename=filename, media_type=media)


# ══ ENDPOINT 10 — Disponibilité des données intermédiaires ══════════════════
@app.get("/api/available/{job_id}")
async def get_available_data(job_id: str):
    available = {
        "graph":         job_id in GRAPHS_STORE,
        "graph_on_disk": (OUTPUTS_DIR / "graphs" / f"{job_id}_graph.json").exists(),
        "cvs": []
    }
    seen_cvs: set[int] = set()
    for subdir in ["phase2", "phase3", "reports"]:
        dirpath = OUTPUTS_DIR / subdir
        if dirpath.exists():
            for f in dirpath.iterdir():
                if f.name.startswith(job_id + "_cv_"):
                    try:
                        cv_idx = int(f.name.split("_cv_")[-1].split("_")[0].replace(".json", "")) - 1
                        seen_cvs.add(cv_idx)
                    except Exception:
                        pass

    for cv_idx in sorted(seen_cvs):
        pk = f"{job_id}_cv_{cv_idx + 1}"
        available["cvs"].append({
            "cv_index":   cv_idx,
            "has_phase2": (pk in PHASE2_STORE) or (OUTPUTS_DIR / "phase2" / f"{pk}_phase2.json").exists(),
            "has_phase3": (pk in PHASE3_STORE) or (OUTPUTS_DIR / "phase3" / f"{pk}_phase3.json").exists(),
            "has_report": (pk in REPORTS_STORE) or (OUTPUTS_DIR / "reports" / f"{pk}_report.json").exists(),
        })
    return available


@app.get("/health")
async def health():
    return {"status": "ok"}