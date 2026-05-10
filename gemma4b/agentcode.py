
from __future__ import annotations

import os
import json
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from typing import Dict, List, Tuple, Optional

import pandas as pd

from pipeline_corrected import (
    # Config
    CSV_INPUT, JD_COLUMN_SCORING, ID_COLUMN, CV_PREFIX,
    TEST_MODE, TEST_MAX_JOBS, TEST_MAX_CVS,
    RESULTS_DIR, BIAS_DIR, REPORTS_DIR,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    _TOTAL_ROWS,

    # Stores globaux
    GRAPHS_STORE, PHASE2_STORE, PHASE3_STORE, BIAS_STORE, REPORTS_STORE,

  
    CvLLM, make_llm_client, OllamaClient,

    
    run_phase1_graph_generation, _make_safe_job_id,

    run_phase2_audit_strategy,
    run_phase3_scoring, apply_experience_and_cert_bonus,
    run_phase3_5_bias_audit,
    phase4_generate_report,

    
    extract_cv_sections, extract_experience_years,
    extract_action_verbs, extract_quantified_metrics, extract_ner_hints,
)



MAX_WORKERS = 2          

ENABLE_CHECKPOINT = True
CHECKPOINT_DIR    = "checkpoints_parallel"
VERBOSE           = True




START_ROW: int = 30   
END_ROW:   int = 34   

# Validation immédiate au chargement du module
assert 0 <= START_ROW < _TOTAL_ROWS, \
    f"START_ROW doit être entre 0 et {_TOTAL_ROWS - 1}, reçu {START_ROW}"
assert START_ROW < END_ROW <= _TOTAL_ROWS, \
    f"END_ROW doit être entre {START_ROW + 1} et {_TOTAL_ROWS}, reçu {END_ROW}"

print(f"✅ Plage configurée : lignes {START_ROW} → {END_ROW - 1}  "
      f"({END_ROW - START_ROW} offres)")




class SharedLLMClient:
    """
    UN SEUL client Ollama partagé entre tous les threads.
    Le verrou _lock garantit qu'un seul thread appelle chat_completion() à la fois.
    """

    def __init__(self):
        self._client     = OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        self._lock       = threading.Lock()
        self._call_count = 0
        self._total_time = 0.0
        print(f"  ✅ SharedLLMClient initialisé — {OLLAMA_MODEL}")

    def chat_completion(self, messages: list, temperature: float = 0.1,
                        max_tokens: int = 800) -> dict:
        with self._lock:
            t0     = time.time()
            result = self._client.chat_completion(
                messages=messages, temperature=temperature, max_tokens=max_tokens,
            )
            self._call_count += 1
            self._total_time  += time.time() - t0
            return result

    def stats(self) -> dict:
        return {
            "total_calls":  self._call_count,
            "total_time_s": round(self._total_time, 1),
            "avg_per_call": round(self._total_time / max(self._call_count, 1), 2),
        }

    @property
    def model(self):
        return self._client.model



class SharedCvLLM:
    def __init__(self, shared_client: SharedLLMClient):
        self._shared = shared_client
        self.model   = shared_client.model
        self.client  = shared_client  

    def _repair_json(self, text: str) -> str:
        import re
        text = re.sub(r"```json\s*|\s*```", "", text).strip()
        s = text.find("{");
        if s == -1: s = text.find("[")
        e = text.rfind("}");
        if e == -1: e = text.rfind("]")
        if s != -1 and e != -1:
            text = text[s:e + 1]
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        ob  = text.count("{") - text.count("}")
        ob2 = text.count("[") - text.count("]")
        if ob2 > 0: text += "]" * ob2
        if ob  > 0: text += "}" * ob
        return text

    def generate(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.1, max_tokens: int = 800,
                 is_json: bool = True, max_retries: int = 2) -> Optional[str]:
        import re
        JSON_RULES = "\nReturn ONLY valid JSON. No markdown. No preamble. Close all brackets.\n"
        sp     = (system_prompt + JSON_RULES) if is_json else system_prompt
        sp     = sp[:4000]
        budget = 14_000 - len(sp) - 200
        if len(prompt) > budget:
            prompt = prompt[:budget]

        messages = [
            {"role": "system", "content": sp},
            {"role": "user",   "content": prompt},
        ]

        for attempt in range(max_retries):
            try:
                resp    = self._shared.chat_completion(messages, temperature, max_tokens)
                content = resp["choices"][0]["message"]["content"]
                if not is_json:
                    return content
                cleaned = self._repair_json(content)
                try:
                    json.loads(cleaned)
                    return cleaned
                except json.JSONDecodeError:
                    if attempt < max_retries - 1:
                        messages += [
                            {"role": "assistant", "content": content},
                            {"role": "user",
                             "content": "Invalid JSON. Return ONLY the corrected JSON."},
                        ]
            except Exception as ex:
                print(f"    ⚠ LLM error (attempt {attempt+1}): {str(ex)[:80]}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        return None if is_json else ""



def _checkpoint_path(pair_key: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"{pair_key}.done")

def _is_done(pair_key: str) -> bool:
    if not ENABLE_CHECKPOINT:
        return False
    return os.path.exists(os.path.join(REPORTS_DIR, f"report_{pair_key}.json"))

def _mark_done(pair_key: str) -> None:
    if ENABLE_CHECKPOINT:
        open(_checkpoint_path(pair_key), "w").close()




def _save_json(data: dict, directory: str, filename: str) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


_LOCK_P2   = threading.Lock()
_LOCK_P3   = threading.Lock()
_LOCK_BIAS = threading.Lock()
_LOCK_REP  = threading.Lock()
_PROGRESS  = {"done": 0, "total": 0, "lock": threading.Lock()}



def process_single_cv(
    job_id:     str,
    cv_id:      str,
    cv_text:    str,
    jd_text:    str,
    graph_data: dict,
    shared_llm: SharedCvLLM,
) -> dict:
    pair_key = f"{job_id}_{cv_id}"
    t0       = time.time()
    result   = {"pair_key": pair_key, "success": False, "verdict": "N/A"}

   
    if _is_done(pair_key):
        print(f"  ⏩ {pair_key} — déjà traité (checkpoint)")
        try:
            with open(os.path.join(REPORTS_DIR, f"report_{pair_key}.json"),
                      encoding="utf-8") as f:
                report = json.load(f)
            with _LOCK_REP:
                REPORTS_STORE[pair_key] = report
            return {"pair_key": pair_key, "success": True,
                    "verdict": report.get("overall_assessment", {}).get("verdict", "?")}
        except Exception:
            pass

    if VERBOSE:
        print(f"\n  🔵 START  {pair_key}")

    try:
       
        if VERBOSE: print(f"    [{pair_key}] ▶ Phase 2 …")
        audit = run_phase2_audit_strategy(
            job_data=graph_data, cv_text=cv_text,
            llm=shared_llm, job_id=job_id, cv_id=cv_id,
        )
        if not audit or not audit.get("skill_instructions"):
            print(f"    [{pair_key}] ❌ Phase 2 vide — abandon")
            return result

        with _LOCK_P2: PHASE2_STORE[pair_key] = audit
        _save_json(audit, RESULTS_DIR, f"{pair_key}_phase2.json")
        if VERBOSE:
            print(f"    [{pair_key}] ✅ Phase 2 OK "
                  f"({len(audit.get('skill_instructions', {}))} compétences)")

      
        if VERBOSE: print(f"    [{pair_key}] ▶ Phase 3 …")
        scored = run_phase3_scoring(audit, shared_llm)
        scored["scored_nodes"] = apply_experience_and_cert_bonus(
            scored_nodes=scored["scored_nodes"],
            skill_instructions=audit.get("skill_instructions", {}),
            exp_years=audit.get("exp_years", {}),
        )
        with _LOCK_P3: PHASE3_STORE[pair_key] = scored
        _save_json(scored, RESULTS_DIR, f"{pair_key}_phase3.json")
        if VERBOSE:
            print(f"    [{pair_key}] ✅ Phase 3 OK "
                  f"({len(scored.get('scored_nodes', {}))} scores | "
                  f"moy={scored.get('metadata', {}).get('mean_score', '?')})")

     
        if VERBOSE: print(f"    [{pair_key}] ▶ Phase 3.5 …")
        audited = run_phase3_5_bias_audit(
            phase3_output=scored, graph_data=graph_data,
            job_desc=jd_text, cv_text=cv_text,
            label=pair_key, llm=shared_llm._shared,
        )
        with _LOCK_BIAS: BIAS_STORE[pair_key] = audited
        _save_json(audited, BIAS_DIR, f"{pair_key}_audited.json")
        if VERBOSE:
            b = audited.get("bias_summary", {})
            print(f"    [{pair_key}] ✅ Phase 3.5 OK "
                  f"(biais={b.get('nodes_with_bias', 0)} / "
                  f"audités={b.get('nodes_audited', 0)})")

    
        if VERBOSE: print(f"    [{pair_key}] ▶ Phase 4 …")
        report = phase4_generate_report(
            job_id=job_id, cv_id=cv_id,
            graph_data=graph_data, phase3_data=audited,
        )
        with _LOCK_REP: REPORTS_STORE[pair_key] = report
        _save_json(report, REPORTS_DIR, f"report_{pair_key}.json")
        _mark_done(pair_key)

        verdict = report.get("overall_assessment", {}).get("verdict", "?")
        match   = report.get("overall_assessment", {}).get("match_pct", "?")
        elapsed = round(time.time() - t0, 1)
        result.update({"success": True, "verdict": verdict})

        with _PROGRESS["lock"]:
            _PROGRESS["done"] += 1
            done, total = _PROGRESS["done"], _PROGRESS["total"]

        print(f"\n  ✅ DONE  {pair_key} | {verdict} ({match}%) "
              f"| ⏱ {elapsed}s | 📊 {done}/{total}")

    except Exception as exc:
        import traceback
        elapsed = round(time.time() - t0, 1)
        print(f"\n  ❌ FAIL  {pair_key} | {exc} | ⏱ {elapsed}s")
        if VERBOSE:
            traceback.print_exc()

    return result



def _slice_df(df: pd.DataFrame, test_mode: bool,
              max_jobs: int, start_row: int, end_row: int) -> pd.DataFrame:
    """
    Applique la plage start_row/end_row puis, si test_mode, limite à max_jobs.

    Ordre intentionnel :
      1. iloc[start:end]   → sélectionne la plage voulue
      2. head(max_jobs)    → en mode test, réduit encore si nécessaire

    Cela permet de tester une portion précise du dataset sans traiter
    depuis le début (ex: start=100, end=200, max_jobs=10 → lignes 100-109).
    """
    sliced = df.iloc[start_row:end_row]
    if test_mode:
        sliced = sliced.head(max_jobs)
    return sliced




def run_full_pipeline_parallel(
    csv_path:    str  = CSV_INPUT,
    test_mode:   bool = TEST_MODE,
    max_jobs:    int  = TEST_MAX_JOBS,
    max_cvs:     int  = TEST_MAX_CVS,
    max_workers: int  = MAX_WORKERS,
    run_graph:   bool = True,
    skip_bias:   bool = False,
    start_row:   int  = START_ROW,   
    end_row:     int  = END_ROW,    
) -> None:
 
   
    assert 0 <= start_row < _TOTAL_ROWS, \
        f"start_row={start_row} hors bornes [0, {_TOTAL_ROWS - 1}]"
    assert start_row < end_row <= _TOTAL_ROWS, \
        f"end_row={end_row} invalide (doit être dans ]{start_row}, {_TOTAL_ROWS}])"

    n_rows_in_range = end_row - start_row
    n_effective     = min(max_jobs, n_rows_in_range) if test_mode else n_rows_in_range

    t_start = time.time()

    
    print("\n" + "═" * 70)
    print(f"  PIPELINE PARALLÈLE — {'🧪 TEST' if test_mode else '🚀 PROD'}")
    print(f"  CSV         : {csv_path}")
    print(f"  Plage CSV   : lignes {start_row} → {end_row - 1}  "
          f"({n_rows_in_range} offres dans la plage)")
    print(f"  Offres eff. : {n_effective}"
          + (f"  (limité par max_jobs={max_jobs})" if test_mode else ""))
    print(f"  CVs/job max : {max_cvs}")
    print(f"  Workers     : {max_workers}")
    print(f"  Ollama      : {OLLAMA_BASE_URL} | Modèle : {OLLAMA_MODEL}")
    print(f"  Checkpoint  : {'✅ ON' if ENABLE_CHECKPOINT else '❌ OFF'}")
    print(f"  Skip biais  : {'✅ OUI' if skip_bias else '❌ NON'}")
    print("═" * 70)

   
    if run_graph:
        run_phase1_graph_generation(
            csv_path=csv_path,
            test_mode=test_mode,
            max_jobs=max_jobs,
            start_row=start_row,
            end_row=end_row,
        )

    if not GRAPHS_STORE:
        print("  ❌ GRAPHS_STORE vide — Arrêt.")
        return

   
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"  ❌ Erreur CSV : {e}")
        return

    df = _slice_df(df, test_mode, max_jobs, start_row, end_row)
    print(f"  📋 {len(df)} lignes après slice "
          f"(start={start_row}, end={end_row}"
          + (f", max_jobs={max_jobs}" if test_mode else "") + ")")

    cv_columns = sorted([c for c in df.columns if c.startswith(CV_PREFIX)])

    
    tasks: List[Tuple] = []
    skipped_checkpoint = 0

    for idx, row in df.iterrows():
        job_id     = _make_safe_job_id(row, ID_COLUMN, idx)
        graph_data = GRAPHS_STORE.get(job_id)
        if not graph_data:
            continue

        jd_text = str(row.get(JD_COLUMN_SCORING, ""))
        cols    = cv_columns[:max_cvs] if test_mode else cv_columns

        for cv_col in cols:
            cv_text  = str(row.get(cv_col, ""))
            if len(cv_text.strip()) < 50 or cv_text == "nan":
                continue

            pair_key = f"{job_id}_{cv_col}"
            if _is_done(pair_key):
                skipped_checkpoint += 1
                try:
                    rp = os.path.join(REPORTS_DIR, f"report_{pair_key}.json")
                    with open(rp, encoding="utf-8") as f:
                        REPORTS_STORE[pair_key] = json.load(f)
                except Exception:
                    pass
                continue

            tasks.append((job_id, cv_col, cv_text, jd_text, graph_data))

    total = len(tasks)
    print(f"\n  📋 {total} paire(s) à traiter | "
          f"{skipped_checkpoint} déjà faites (checkpoint)")

    if total == 0:
        print("  ✅ Tout est déjà traité !")
        _print_final_stats(t_start, 0, 0, total, max_workers,
                           start_row, end_row)
        return

    with _PROGRESS["lock"]:
        _PROGRESS["total"] = total
        _PROGRESS["done"]  = 0

   
    shared_raw = SharedLLMClient()
    shared_llm = SharedCvLLM(shared_raw)

   
    success_count = 0
    fail_count    = 0
    verdicts: List[str] = []

    print(f"\n  ┌{'─' * 68}")

    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="CVAgent"
    ) as executor:
        future_map = {
            executor.submit(
                process_single_cv,
                job_id, cv_id, cv_text, jd_text, graph_data, shared_llm,
            ): (job_id, cv_id)
            for job_id, cv_id, cv_text, jd_text, graph_data in tasks
        }

        for future in as_completed(future_map):
            job_id, cv_id = future_map[future]
            try:
                res = future.result()
                if res.get("success"):
                    success_count += 1
                    verdicts.append(res.get("verdict", "?"))
                else:
                    fail_count += 1
            except Exception as exc:
                print(f"  ❌ Future error ({job_id}/{cv_id}): {exc}")
                fail_count += 1

    print(f"  └{'─' * 68}")

    
    llm_stats = shared_raw.stats()
    print(f"\n  📡 Stats LLM : {llm_stats['total_calls']} appels | "
          f"{llm_stats['total_time_s']}s total | "
          f"{llm_stats['avg_per_call']}s/appel en moy.")

    _print_final_stats(t_start, success_count, fail_count, total,
                       max_workers, start_row, end_row, verdicts)



def _print_final_stats(t_start, success, fail, total, workers,
                       start_row, end_row, verdicts=None):
    elapsed    = round(time.time() - t_start, 1)
    avg_per_cv = round(elapsed / max(total, 1), 1)

    print("\n" + "═" * 70)
    print(f"  ✅ PIPELINE TERMINÉ")
    print(f"  📍 Plage traitée   : lignes {start_row} → {end_row - 1}  "
          f"({end_row - start_row} offres)")
    print(f"  ⏱  Temps total     : {elapsed}s (~{avg_per_cv}s / CV)")
    print(f"  ✅  Succès          : {success} / {total}")
    print(f"  ❌  Échecs          : {fail} / {total}")
    print(f"  👷  Workers         : {workers}")

    print(f"\n  Stores finaux :")
    print(f"    GRAPHS_STORE  : {len(GRAPHS_STORE)}")
    print(f"    PHASE2_STORE  : {len(PHASE2_STORE)}")
    print(f"    PHASE3_STORE  : {len(PHASE3_STORE)}")
    print(f"    BIAS_STORE    : {len(BIAS_STORE)}")
    print(f"    REPORTS_STORE : {len(REPORTS_STORE)}")

    if verdicts:
        vc = Counter(verdicts)
        print(f"\n  Verdicts :")
        for v, c in vc.most_common():
            print(f"    {v:40s} × {c}")

    print("═" * 70)



if __name__ == "__main__":

    run_full_pipeline_parallel(
        csv_path    = CSV_INPUT,
        test_mode   = True,   
        max_jobs    = 30,     
        max_cvs     = 5,
        max_workers = 2,      
        run_graph   = True,
        skip_bias   = False,
        start_row   = START_ROW,  
        end_row     = END_ROW,
    )