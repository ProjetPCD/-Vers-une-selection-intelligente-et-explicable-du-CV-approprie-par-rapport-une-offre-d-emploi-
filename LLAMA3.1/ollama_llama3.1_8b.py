from __future__ import annotations


import json
import re
import os
import time
import math
import zipfile
import requests
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import networkx as nx
import numpy as np
import numpy as np
from difflib import SequenceMatcher

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    print("[WARN] spacy non disponible — NER CV désactivé")



_COVERAGE_THRESH  = 0.50   
_ELITE_THRESH     = 0.80   
_CRITICAL_THRESH  = 0.70
_MAX_CV_SCORE     = 2.0   




TEST_MODE      = True      
TEST_MAX_JOBS  = 2         
TEST_MAX_CVS   = 2       

# ── Fichiers d'entrée ────────────────────────────────────────────────────────
CSV_INPUT           = "C:\\Users\\USER\\Desktop\\sou\\job-1 (1).csv"  
JD_COLUMN           = "job_description_adequate"
JD_COLUMN_SCORING   = "job_description_adequate"
ID_COLUMN           = None     
CV_PREFIX           = "cv_"


GRAPHS_DIR   = "C:\\Users\\USER\\Desktop\\sou\\oll11"
RESULTS_DIR  = "C:\\Users\\USER\\Desktop\\sou\\oll22"
BIAS_DIR     = "C:\\Users\\USER\\Desktop\\sou\\oll33"
REPORTS_DIR  = "C:\\Users\\USER\\Desktop\\sou\\oll44"


NVIDIA_API_KEY  = ""
NVIDIA_MODEL    = "meta/llama-3.1-8b-instruct"
USE_NVIDIA_API  = False  


OLLAMA_BASE_URL = "http://localhost:11434"  
OLLAMA_MODEL    = "llama3.1:8b"  




CONF_THRESHOLD      = 0.35    
JD_MAX_CHARS        = 6_000
MAX_PROMPT_CHARS    = 14_000
MAX_SYSTEM_CHARS    = 5_500

NODE_PRUNE_THRESHOLD  = 0.20
BIAS_AUDIT_DELTA_THRESHOLD = 0.15

CHECKPOINT_DIR = "checkpoints"
VOCAB_PATH     = "C:\\Users\\USER\\Desktop\\sou\\vocab.json"
REGISTRY_PATH  = "C:\\Users\\USER\\Desktop\\sou\\skill_registry.json"

GRAPHS_STORE:    Dict[str, dict] = {}
PHASE2_STORE:    Dict[str, dict] = {}
PHASE3_STORE:    Dict[str, dict] = {}
BIAS_STORE:      Dict[str, dict] = {}
REPORTS_STORE:   Dict[str, dict] = {}

print("✅ Config chargée (OLLAMA LOCAL UNIQUEMENT).")
print(f"   Mode : {'🧪 TEST' if TEST_MODE else '🚀 COMPLET'} "
      f"(max {TEST_MAX_JOBS} jobs × {TEST_MAX_CVS} CVs)")
if USE_NVIDIA_API:
    print(f"   API : NVIDIA | Model : {NVIDIA_MODEL}")
else:
    print(f"   ✅ API : Ollama Local | Model : {OLLAMA_MODEL}")
    print(f"   📍 URL : {OLLAMA_BASE_URL}")
    print(f"   💡 Assurez-vous que Ollama est lancé : ollama serve")

for directory in [GRAPHS_DIR, RESULTS_DIR, BIAS_DIR, REPORTS_DIR, CHECKPOINT_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)
print(f"✅ Dossiers créés : {GRAPHS_DIR}, {RESULTS_DIR}, {REPORTS_DIR}")




class NvidiaClient:
    """Client HTTP pour NVIDIA API (gratuit avec clé API)."""

    def __init__(self, api_key: str, model: str = NVIDIA_MODEL):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self._check_connection()

    def _check_connection(self):
        """Vérifie que la clé API est valide."""
        if not self.api_key:
            raise ValueError("❌ NVIDIA_API_KEY manquante ! Récupérez-la sur https://build.nvidia.com/")
        print(f"  ✅ NVIDIA API connectée")
        print(f"     Modèle: {self.model}")

    def chat_completion(self, messages: List[dict], temperature: float = 0.05,
                       max_tokens: int = 3000) -> dict:
        """Appelle NVIDIA API avec messages (retry automatique + backoff 429)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    self.api_url, json=payload, headers=headers,
                    timeout=180
                )
                resp.raise_for_status()
                data = resp.json()
                return {
                    "choices": [{
                        "message": {
                            "content": data["choices"][0]["message"]["content"]
                        }
                    }]
                }
            except requests.exceptions.HTTPError as e:
                # ✅ Gestion spéciale pour 429 (Too Many Requests)
                if e.response.status_code == 429:
                    wait = 5 * (2 ** attempt)  # Backoff exponentiel : 10, 20, 40, 80s
                    print(f"  ⚠️ Rate limit (429) — attente {wait}s... (tentative {attempt}/{max_retries})")
                    if attempt < max_retries:
                        time.sleep(wait)
                        continue
                    else:
                        print(f"  ❌ Rate limit dépassé après {max_retries} tentatives.")
                        raise
                else:
                    print(f"  ❌ Erreur HTTP {e.response.status_code}: {e}")
                    raise
            except requests.exceptions.Timeout:
                wait = 10 * attempt
                print(f"  ⏳ Timeout NVIDIA API (tentative {attempt}/{max_retries}) — attente {wait}s...")
                if attempt < max_retries:
                    time.sleep(wait)
                else:
                    print(f"  ❌ Abandon après {max_retries} tentatives.")
                    raise
            except requests.exceptions.RequestException as e:
                print(f"  ❌ Erreur NVIDIA API: {e}")
                raise


class OllamaClient:
    """Client HTTP pour Ollama compatible avec l'API OpenAI."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api/chat"
        self._check_connection()

    def _check_connection(self):
        """Vérifie que Ollama est accessible."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                print(f"  ✅ Ollama connecté ({self.base_url})")
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                print(f"     Modèles disponibles: {', '.join(model_names[:5])}")
            else:
                print(f"  ⚠️ Ollama répond mais status {resp.status_code}")
        except Exception as e:
            print(f"  ❌ Impossible de connecter Ollama: {e}")
            print(f"     Assurez-vous que Ollama tourne sur {self.base_url}")
            print(f"     Lancez: ollama serve")
            raise

    def chat_completion(self, messages: List[dict], temperature: float = 0.05,
                       max_tokens: int = 3000) -> dict:
        """Appelle Ollama avec messages (compatible OpenAI)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        try:
            resp = requests.post(self.api_url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()

           
            print(f"[DEBUG] Ollama response keys: {data.keys()}")
            print(f"[DEBUG] Message structure: {data.get('message', {})}")

      
            return {
                "choices": [{
                    "message": {
                        "content": data.get("message", {}).get("content", "")
                    }
                }]
            }
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Erreur Ollama: {e}")
            raise
        except Exception as e:
            print(f"  ❌ Erreur traitement réponse: {e}")
            raise




_FALLBACK_VOCAB: dict = {
    "version": "2.0",
    "hard_skills": [
        "python","java","javascript","typescript","c++","c#","go","rust","sql","nosql",
        "machine learning","deep learning","nlp","computer vision","data science",
        "data engineering","data analysis","statistics","etl","feature engineering",
        "mlops","llm","reinforcement learning","time series","anomaly detection",
        "recommendation systems","a/b testing","data modeling","data visualization",
        "business intelligence","neural networks","transformer","fine-tuning","rag",
        "embeddings","vector search","knowledge graph","cybersecurity","aws","azure",
        "gcp","docker","kubernetes","terraform","ci/cd","linux","git","api","rest",
        "graphql","microservices","devops","spark","kafka","airflow","dbt","flink",
        "postgresql","mysql","mongodb","redis","elasticsearch","bigquery","snowflake",
    ],
    "tools": [
        "tensorflow","pytorch","keras","scikit-learn","xgboost","lightgbm",
        "hugging face","langchain","llamaindex","mlflow","wandb","pandas","numpy",
        "matplotlib","tableau","power bi","looker","databricks","github","gitlab",
        "jenkins","github actions","grafana","prometheus","datadog","jupyter","colab",
    ],
    "soft_skills": [
        "communication","leadership","teamwork","problem solving","critical thinking",
        "adaptability","time management","collaboration","attention to detail",
        "analytical thinking","mentoring","stakeholder management","autonomy",
        "curiosity","rigour","resilience",
    ],
    "domains": [
        "fintech","healthcare","e-commerce","cybersecurity","blockchain","iot",
        "saas","banking","insurance","logistics","retail","telecom",
        "artificial intelligence","cloud computing","big data","generative ai",
    ],
    "education": [
        "bachelor","master","phd","mba","computer science","mathematics",
        "statistics","engineering","bac+5","bac+3",
    ],
    "certifications": [
        "aws certified","azure certified","gcp certified","pmp","cissp","cism",
        "ckad","cka","terraform associate","databricks certified","scrum master",
    ],
    "aliases": {
        "ml": "machine learning", "dl": "deep learning", "k8s": "kubernetes",
        "nlp": "natural language processing", "llms": "llm", "genai": "generative ai",
        "bi": "business intelligence", "tf": "tensorflow", "ts": "typescript",
        "js": "javascript", "py": "python", "pg": "postgresql",
        "es": "elasticsearch", "rag": "rag", "cv": "computer vision",
    },
    "llm_discovered": [],
}

_CAT_TO_TYPE = {
    "hard_skills": "Hard Skill", "tools": "Tool", "soft_skills": "Soft Skill",
    "domains": "Domain", "education": "Education", "certifications": "Certification",
    "llm_discovered": "Hard Skill",
}


class VocabManager:
    def __init__(self, path: str = VOCAB_PATH):
        self.path = Path(path)
        self._data = self._load()
        self._build_index()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return data
        self._save(_FALLBACK_VOCAB)
        return dict(_FALLBACK_VOCAB)

    def _save(self, data: dict | None = None) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data or self._data, f, ensure_ascii=False, indent=2)

    def _build_index(self) -> None:
        self.aliases: dict[str, str] = {
            k.lower(): v.lower() for k, v in self._data.get("aliases", {}).items()
        }
        self.term_type: dict[str, str] = {}
        for cat, ptype in _CAT_TO_TYPE.items():
            for t in self._data.get(cat, []):
                self.term_type[t.lower()] = ptype
        self.all_terms: set[str] = set(self.term_type)

    def resolve_alias(self, term: str) -> str:
        return self.aliases.get(term.lower(), term)

    def get_type(self, term: str) -> str:
        return self.term_type.get(term.lower(), "Hard Skill")

    def update_from_llm_nodes(self, nodes: list[dict]) -> int:
        discovered: list[str] = self._data.setdefault("llm_discovered", [])
        pending: dict = self._data.setdefault("pending_review", {})
        known_lower = {t.lower() for t in discovered} | self.all_terms
        added = 0
        for n in nodes:
            label = n.get("label", "").strip()
            if not label or len(label) < 3 or len(label.split()) > 4:
                continue
            if n.get("type") not in ("Hard Skill", "Tool", "Soft Skill", "Domain"):
                continue
            key = label.lower()
            if key in known_lower:
                continue
            pending[key] = pending.get(key, 0) + 1
            if pending[key] >= 2:
                discovered.append(label)
                known_lower.add(key)
                self.term_type[key] = n["type"]
                self.all_terms.add(key)
                del pending[key]
                added += 1
        if added:
            self._save()
            self._build_index()
        return added



def _canonical_key(label: str) -> str:
    key = re.sub(r"[^a-z0-9\s]", " ", label.lower().strip())
    return re.sub(r"\s+", "_", key).strip("_")


class SkillRegistry:
    def __init__(self, path: str = REGISTRY_PATH):
        self.path = Path(path)
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def register_nodes(self, nodes: list[dict], job_id: str) -> None:
        for n in nodes:
            key = _canonical_key(n.get("label", n.get("id", "")))
            if not key:
                continue
            if key not in self._data:
                self._data[key] = {
                    "canonical_label": n.get("label", key),
                    "type": n.get("type", "N/A"),
                    "freq": 0, "w_0_sum": 0.0, "jobs": []
                }
            e = self._data[key]
            e["freq"] += 1
            e["w_0_sum"] += float(n.get("w_0", 0.5))
            if job_id not in e["jobs"]:
                e["jobs"].append(job_id)
            e["w_global"] = round((e["w_0_sum"] / e["freq"]) * math.log1p(e["freq"]), 4)
        self._save()

    def get_global_weight(self, label: str) -> float | None:
        key = _canonical_key(label)
        entry = self._data.get(key)
        return entry["w_global"] if entry else None



_SECTION_PRIORITIES = [
    (1.00, [r"responsabilit", r"mission", r"what you.ll do", r"key responsibilities"]),
    (0.95, [r"required skills?", r"must.have", r"technical skills?", r"stack"]),
    (0.80, [r"profil", r"candidate profile", r"about you", r"who you are"]),
    (0.65, [r"nice.to.have", r"preferred", r"bonus", r"atout"]),
    (0.50, [r"contexte", r"about the role"]),
    (0.25, [r"about (?:us|the company)", r"qui sommes.nous"]),
    (0.10, [r"benefits?", r"perks?", r"avantages?"]),
]

_GLINER_LABELS = [
    "programming language", "framework", "library", "tool", "cloud service",
    "database", "soft skill", "certification", "degree", "job title",
    "technology domain", "methodology",
]
_GLINER_LABEL_TO_TYPE = {
    "programming language": "Hard Skill", "framework": "Tool",
    "library": "Tool", "tool": "Tool", "cloud service": "Tool",
    "database": "Tool", "soft skill": "Soft Skill",
    "certification": "Certification", "degree": "Education",
    "job title": "Domain", "technology domain": "Domain",
    "methodology": "Hard Skill",
}

_REGEX_RULES = [
    (r"\b(?:AWS|Azure|GCP|Google Cloud)\b(?:\s+\w+){0,2}", "Tool", 0.95),
    (r"\b(?:AWS|Azure|GCP)\s+Certified\s+\w+", "Certification", 1.0),
    (r"\b(?:\d+\+?\s+years?)\s+(?:of\s+)?(?:experience\s+(?:with|in))?\s*\w+", "Experience", 0.90),
    (r"\b(?:B\.?S\.?c?|M\.?S\.?c?|Ph\.?D\.?|MBA|Bachelor|Master|Licence|Ingénieur)\b", "Education", 0.90),
    (r"\b(?:PMP|CISSP|CKA|CKAD|Scrum Master|AWS Certified|Azure Certified|GCP Professional)\b", "Certification", 0.98),
]
_REGEX_AUTHORITY_TYPES = {"Certification", "Education", "Experience"}


class RegexExtractor:
    WEIGHT = 1.0

    def extract(self, text: str) -> list[dict]:
        seen, candidates = set(), []
        for pattern, ptype, conf in _REGEX_RULES:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                raw = m.group(0).strip()
                key = raw.lower()
                if key in seen or len(key) < 3:
                    continue
                seen.add(key)
                ctx = text[max(0, m.start() - 60):m.end() + 60].strip()[:120]
                candidates.append({
                    "text": raw, "provisional_type": ptype,
                    "context": ctx, "source": "regex", "confidence": conf
                })
        return candidates


class GlinerExtractor:
    WEIGHT = 0.75
    SEGMENT_SIZE = 2_000
    OVERLAP = 200

    def __init__(self, model_name="urchade/gliner_multi-v2.1"):
        try:
            from gliner import GLiNER
            self._model = GLiNER.from_pretrained(model_name)
            self._available = True
            print(f"  [GLiNER] {model_name.split('/')[-1]} ✓")
        except Exception as e:
            self._available = False
            print(f"  [GLiNER] non disponible : {e}")

    @property
    def available(self): return self._available

    def _segments(self, text):
        segs, i = [], 0
        while i < len(text):
            end = min(i + self.SEGMENT_SIZE, len(text))
            segs.append((i, text[i:end]))
            if end == len(text):
                break
            i += self.SEGMENT_SIZE - self.OVERLAP
        return segs

    def extract(self, text):
        if not self._available:
            return []
        seen, candidates = set(), []
        for offset, segment in self._segments(text):
            try:
                entities = self._model.predict_entities(segment, _GLINER_LABELS)
            except Exception:
                continue
            for ent in entities:
                raw = ent["text"].strip()
                key = raw.lower()
                if key in seen or len(key) < 2:
                    continue
                seen.add(key)
                ptype = _GLINER_LABEL_TO_TYPE.get(ent["label"].lower(), "Hard Skill")
                abs_start = offset + segment.find(raw)
                ctx = text[max(0, abs_start - 60):abs_start + len(raw) + 60].strip()
                candidates.append({
                    "text": raw, "provisional_type": ptype, "context": ctx[:120],
                    "source": "gliner", "confidence": round(ent.get("score", 0.7), 3)
                })
        return candidates


class GlinerV2Extractor(GlinerExtractor):
    WEIGHT = 0.85
    SEGMENT_SIZE = 1_500

    def __init__(self):
        super().__init__("urchade/gliner_large-v2.1")


class JobBertExtractor:
    WEIGHT_EN = 0.90
    WEIGHT_FR = 0.50
    WEIGHT_OTHER = 0.70
    CHUNK_WORDS = 400
    _LABEL_MAP = {
        "B-SKILL": "Hard Skill", "I-SKILL": "Hard Skill",
        "B-SOFT": "Soft Skill", "I-SOFT": "Soft Skill",
        "B-TOOL": "Tool", "I-TOOL": "Tool",
        "B-CERT": "Certification", "B-EDU": "Education",
    }

    def __init__(self, lang="other"):
        self._lang = lang
        try:
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline("ner", model="jjzha/jobbert-base-cased",
                                     aggregation_strategy="simple")
            self._available = True
            print(f"  [JobBERT] ✓")
        except Exception as e:
            self._available = False
            print(f"  [JobBERT] non disponible : {e}")

    def set_language(self, lang): self._lang = lang

    @property
    def weight(self):
        return {"en": self.WEIGHT_EN, "fr": self.WEIGHT_FR,
                "other": self.WEIGHT_OTHER}.get(self._lang, self.WEIGHT_OTHER)

    @property
    def available(self): return self._available

    def extract(self, text):
        if not self._available:
            return []
        seen, candidates = set(), []
        words = text.split()
        for i in range(0, len(words), self.CHUNK_WORDS):
            overlap = " ".join(words[max(0, i - 20):i]) if i > 0 else ""
            chunk = (overlap + " " + " ".join(words[i:i + self.CHUNK_WORDS])).strip()
            try:
                entities = self._pipe(chunk)
            except Exception:
                continue
            for ent in entities:
                raw = ent["word"].strip()
                key = raw.lower()
                if key in seen or len(key) < 2:
                    continue
                ptype = self._LABEL_MAP.get(ent.get("entity_group", ""))
                if not ptype:
                    continue
                seen.add(key)
                candidates.append({
                    "text": raw, "provisional_type": ptype, "context": "",
                    "source": "jobbert", "confidence": round(ent.get("score", 0.8), 3)
                })
        return candidates


class FusedExtractor:
    def __init__(self, regex_ext, gliner_ext=None, gliner_v2_ext=None,
                 jobbert_ext=None, vocab=None):
        self._regex = regex_ext
        self._gliner = gliner_ext
        self._glinerv2 = gliner_v2_ext
        self._jobbert = jobbert_ext
        self._vocab = vocab

    def _normalize(self, text):
        key = text.strip().lower()
        return self._vocab.resolve_alias(key) if self._vocab else key

    def extract_all_nodes(self, text) -> list[dict]:
        layers = [(self._regex.extract(text), RegexExtractor.WEIGHT)]
        if self._gliner and self._gliner.available:
            layers.append((self._gliner.extract(text), GlinerExtractor.WEIGHT))
        if self._glinerv2 and self._glinerv2.available:
            layers.append((self._glinerv2.extract(text), GlinerV2Extractor.WEIGHT))
        if self._jobbert and self._jobbert.available:
            layers.append((self._jobbert.extract(text), self._jobbert.weight))

        vote_map: dict = {}
        for layer_candidates, layer_weight in layers:
            for c in layer_candidates:
                norm = self._normalize(c["text"])
                if not norm or len(norm) < 2:
                    continue
                if norm not in vote_map:
                    vote_map[norm] = {
                        "best_raw": c["text"], "context": c.get("context", ""), "votes": []
                    }
                vote_map[norm]["votes"].append(
                    (c["confidence"], layer_weight, c["provisional_type"], c["source"])
                )

        result = []
        for norm, entry in vote_map.items():
            votes = entry["votes"]
            nb = len(votes)
            w_conf = sum(conf * wt for conf, wt, _, _ in votes) / nb
            if nb >= 2:
                w_conf = min(w_conf * 1.5, 1.0)
            if w_conf < CONF_THRESHOLD:
                continue
            regex_votes = [
                (conf, wt, t, src) for conf, wt, t, src in votes
                if src == "regex" and t in _REGEX_AUTHORITY_TYPES
            ]
            final_type = (
                regex_votes[0][2] if regex_votes
                else max(votes, key=lambda v: v[0] * v[1])[2]
            )
            result.append({
                "text": entry["best_raw"], "provisional_type": final_type,
                "context": entry["context"], "source": "+".join(sorted({s for _, _, _, s in votes})),
                "confidence": round(w_conf, 3), "vote_count": nb
            })
        result.sort(key=lambda x: x["confidence"], reverse=True)
        print(f"  [FusedExtractor] {len(result)} candidats | seuil={CONF_THRESHOLD}")
        return result



def _similarity_ratio(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def semantic_dedup_nodes(nodes: list[dict], sim_threshold: float = 0.75) -> list[dict]:
    kept: list[dict] = []
    for node in sorted(nodes, key=lambda n: n.get("w_0", 0.5), reverse=True):
        label = node.get("label", node.get("id", ""))
        is_dup = False
        for existing in kept:
            ex_label = existing.get("label", existing.get("id", ""))
            if _similarity_ratio(label, ex_label) >= sim_threshold:
                existing["w_0"] = max(existing.get("w_0", 0.5), node.get("w_0", 0.5))
                is_dup = True
                break
        if not is_dup:
            kept.append(node)
    removed = len(nodes) - len(kept)
    if removed:
        print(f"  [Dedup] {removed} nœud(s) dupliqué(s) fusionné(s)")
    return kept



def prune_low_weight_nodes(nodes: list[dict], edges: list[dict],
                           threshold: float = NODE_PRUNE_THRESHOLD) -> list[dict]:
    important_ids = {n["id"] for n in nodes if n.get("w_0", 0.5) >= 0.5}
    protected_targets = {
        e["target"] for e in edges
        if e.get("source") in important_ids
    }

    pruned, kept = [], []
    for n in nodes:
        w = n.get("w_0", 0.5)
        ntype = n.get("type", "")
        if ntype in ("Certification", "Education"):
            kept.append(n)
        elif w >= threshold or n["id"] in protected_targets:
            kept.append(n)
        else:
            pruned.append(n)

    if pruned:
        print(f"  [Pruning] {len(pruned)} nœud(s) bruit retiré(s)")
    return kept



def registry_boost(nodes: list[dict], registry: SkillRegistry) -> list[dict]:
    for n in nodes:
        w_global = registry.get_global_weight(n.get("label", ""))
        if w_global is not None:
            w_orig = n.get("w_0", 0.5)
            w_new = round(0.85 * w_orig + 0.15 * min(w_global, 1.0), 3)
            w_new = max(w_orig - 0.10, min(w_orig + 0.10, w_new))
            n["w_0"] = w_new
    return nodes



def edge_quality_filter(edges: list[dict], nodes: list[dict],
                        min_node_weight: float = 0.15) -> list[dict]:
    weight_map = {n["id"]: n.get("w_0", 0.5) for n in nodes}
    filtered = [
        e for e in edges
        if weight_map.get(e.get("source", ""), 0) >= min_node_weight
        and weight_map.get(e.get("target", ""), 0) >= min_node_weight
    ]
    removed = len(edges) - len(filtered)
    if removed:
        print(f"  [EdgeFilter] {removed} arête(s) bruit retirée(s)")
    return filtered


def detect_language(text: str) -> str:
    fr_markers = len(re.findall(r"\b(et|de|le|la|les|un|une|vous|nous|pour|avec|dans)\b", text, re.I))
    en_markers = len(re.findall(r"\b(and|the|of|to|in|you|we|for|with|your|our)\b", text, re.I))
    if fr_markers > en_markers * 1.5:
        return "fr"
    if en_markers > fr_markers * 1.5:
        return "en"
    return "other"


def make_llm_client(model: str = None):
    """Crée un client LLM (NVIDIA ou Ollama selon config)."""
    if USE_NVIDIA_API and NVIDIA_API_KEY:
        return NvidiaClient(api_key=NVIDIA_API_KEY, model=NVIDIA_MODEL)
    else:
        return OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)


PASS_A_SYSTEM = """You are a Senior Job Requirements Analyst. Your task: extract a structured graph from a job description.
RULES:
1. Extract ONLY skills/requirements explicitly or strongly implicitly present.
3. Types: Hard Skill | Tool | Soft Skill | Domain | Education | Certification | Responsibility
4. Focus on SUBSTANCE over NOISE — prefer fewer, high-quality nodes.
### STEP 2 — CLASSIFY
Assign each kept node its FINAL type:
- "Hard Skill"    : Technical competency (Python, SQL, Machine Learning...)
- "Soft Skill"    : Behavioral/interpersonal trait (Leadership, Communication...)
- "Tool"          : Specific software, framework, library or platform (PyTorch, Docker...)
  NOTE: Named tools like PyTorch, Airflow, Tableau MUST be "Tool", never "Hard Skill"
- "Responsibility": Concrete task or outcome expected in the role
- "Domain"        : Industry or field context (FinTech, Healthcare, AI...)
- "Experience"    : Required years or type of prior professional exposure
- "Education"     : Degree, diploma, or academic background
- "Certification" : Formal credential or professional certification

### STEP 3 — SCORE w_0 (read the FULL JD, not just the context snippet)
- 0.9 – 1.0 : In job title OR core responsibility with obligation language
- 0.7 – 0.8 : Strongly tied to primary day-to-day tasks (own, lead, build, design)
- 0.5 – 0.7 : Responsibility-linked soft skill OR important secondary skill
- 0.4 – 0.6 : Supporting tool or secondary requirement, mentioned once
- 0.1 – 0.3 : Explicitly "nice-to-have", "plus", "optional", or "bonus"
After processing the candidate list, RE-READ the full JD and ask yourself:
### HOW TO INFER — mapping patterns to skills:

| JD sentence pattern | Implied skill(s) |
|---|---|
| "animer / organiser des réunions" | Teamwork, Communication, Facilitation |
| "gérer les priorités / deadlines" | Gestion du temps, Organisation |
| "travailler en autonomie sur..." | Autonomie, Self-starter |
This list is ILLUSTRATIVE, not exhaustive. Use your semantic understanding of the full JD.

### INFERENCE RULES:

1. **Any node type can be inferred** — Hard Skills, Tools, Soft Skills, and Responsibilities
   alike — as long as the reasoning is grounded in the JD context or in a domain-universal truth.

2. **Grade every inference by confidence level** and reflect it in w_0:

   | Confidence | Condition | w_0 penalty vs explicit |
   |---|---|---
   | HIGH (0.85+) | Domain-universal technical truth — no reasonable alternative exists | −0.05 |
   | MED  (0.60–0.84) | Strong contextual implication — likely but not 100% certain | −0.15 |
   | LOW  (0.30–0.59) | Plausible but speculative — cite your reasoning carefully | −0.25 |
    Examples by confidence:
   - HIGH: "frameworks deep learning" → Python (PyTorch and TensorFlow are Python-only APIs)
   - MED:  "développe des modèles ML" → Python (dominant but R/Julia are possible)
   - LOW:  "travaille sur des données"→ SQL (too vague — could be NoSQL, Spark, etc.)
3. **Write your reasoning chain** in `inferred_from` — not just the source sentence,
   but WHY you made the inference:
   "PyTorch and TensorFlow are Python-only → deep learning frameworks imply Python"
4. **Do NOT infer when**:
   - The JD context is too vague (e.g. "travaille avec des outils modernes")
   - Multiple equally likely alternatives exist with no contextual tiebreaker
5. **Inferred nodes are marked** with "inferred": true.
   Explicit nodes get "inferred": false (omit inferred_from).


OUTPUT: strict JSON only.
{
   "nodes": [
    {{
      "id": "snake_case_unique_slug",
      "label": "Human Readable Label",
      "type": "Hard Skill | Soft Skill | Tool | Responsibility | Domain | Experience | Education | Certification",
      "w_0": <float 0.0-1.0>,
      "seniority_required": "Junior | Mid | Senior | Expert | N/A",
      "context": "One sentence: how this node appears or is implied in this specific JD",
      "inferred": true | false,
      "inferred_from": "<exact JD sentence that implies this node, or omit if inferred=false>"
    }}
  ]
}"""

PASS_B_SYSTEM = """You are a Knowledge Graph Architect. Create semantic edges between job requirement nodes.

RELATION TYPES (ordered by strength):
- requires |enables|validates| part_of |supports|  implements| feeds|complements
- "requires"
A CANNOT be effectively used without B in this role's context.
✓ "Spark requires Python" — Spark DAGs are written in Python
✗ "Docker requires Kubernetes" — Docker works standalone; that's "supports" at most

- "supports"
A makes B easier or more effective, but B can exist without A.
✓ "Docker supports CI/CD" — containers improve pipelines but aren't mandatory
✗ "Python supports Pandas" — Pandas CANNOT run without Python; use "requires"

- "part_of"
  A is a subset, instance, or specific activity within B.
  ✓ "Pandas is part_of Data Analysis"
  ✗ "Leadership is part_of Communication" — they're peers; use "complements"
- "implements"
    A (Tool or Responsibility) is the concrete execution of B (skill or broader task).
    ✓ "Airflow implements Pipeline Orchestration"
    ✓ "Tableau implements Data Visualization"
    ✗ "Python implements Machine Learning" — Python is too general; use "requires" instead

- "complements"
    A and B are peer skills that reinforce each other with no hierarchy.
    ✓ "Communication complements Leadership"
    ✗ "Python complements Pandas" — Pandas requires Python; use "requires"

- "validates"
    A (Certification or Education) formally certifies mastery of B.
    ✓ "AWS Certified validates Cloud Computing skills"

- "enables"
    A (Experience or Education) opens access to a Responsibility or advanced skill.
    ✓ "5 years ML experience enables Architecture Design responsibility"

- "feeds"
   A (Responsibility) produces output that is consumed or required by B (Responsibility).
  Use ONLY between two Responsibility nodes to express data/workflow lineage.
  ✓ "Data Analysis feeds Data Visualization" — analysis results are the input to viz
  ✗ "Python feeds Data Analysis" — Python is a skill, not a responsibility; use "part_of"
## CRITICAL RULE — CONDITIONAL EDGES ONLY

Create an edge ONLY IF at least one condition holds:
  (a) The relationship is EXPLICITLY stated in the JD (e.g. "using Airflow to orchestrate pipelines")
  (b) The relationship is STRONGLY IMPLIED by the JD context
      (e.g. Python-heavy role + Pandas listed → "Pandas requires Python" is implied)
  (c) The relationship is a domain-universal technical truth
      (e.g. Spark always requires Python or Scala)

  (d) Only create edges between nodes that have a REAL semantic dependency.
  (e) Each edge needs a specific, verifiable rationale.
  (f) Prefer quality over quantity

DO NOT create an edge merely to fill a connectivity quota.
If you cannot cite a JD signal or a technical truth, OMIT the edge.

A sparse, accurate graph is better than a dense, hallucinated one.
## EDGE WEIGHT SCORING
- 0.85 – 1.0 : Explicitly stated in the JD with strong language
- 0.65 – 0.85: Strongly implied by the JD context
- 0.40 – 0.65: Domain-universal technical truth not specifically emphasized in this JD
- 0.10 – 0.40: Weak or speculative — add ONLY if you can cite a JD signal

## FEW-SHOT EDGES (compact)

CORRECT:
  python  → requires   → airflow  (DAGs are Python-only)
  data_analysis → feeds → data_visualization

WRONG:
  python → part_of → data_science_domain  (general-purpose, not a sub-domain)
## OUTPUT — strict JSON only, no markdown, no preamble, no trailing commas
{{
  "edges": [
    {{
      "source": "node_id",
      "target": "node_id",
      "relation": "requires | supports | part_of | implements | complements | validates | enables | feeds",
      "weight": <float 0.0-1.0>,
      "rationale": "One sentence citing the JD signal or technical truth behind this edge"
    }}
  ]
}}
OUTPUT: strict JSON only.
{"edges": [{"source":"id_a","target":"id_b","relation":"requires","weight":0.9,"rationale":"<why>"}]}
"""


def _clean_json_response(raw: str) -> str:
    """Nettoie et tente de réparer le JSON du LLM."""
    if not raw:
        return "{}"
    
   
    raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
    
    s = raw.find("{")
    if s == -1:
        s = raw.find("[")
    e = raw.rfind("}")
    if e == -1:
        e = raw.rfind("]")
    
    if s != -1 and e != -1:
        raw = raw[s:e + 1]
    
   
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
  
    raw = re.sub(r'\}\s*\{', '}, {', raw)

    raw = re.sub(r'"\s*\n?\s*"', '", "', raw)
    

    try:
        json.loads(raw)
        return raw  
    except json.JSONDecodeError as e:
     
        cut = raw[:e.pos]
       
        last_complete = max(cut.rfind("},"), cut.rfind("}]"), cut.rfind("}\n"))
        if last_complete != -1:
            raw = cut[:last_complete + 1]
           
            open_braces = raw.count("{") - raw.count("}")
            open_brackets = raw.count("[") - raw.count("]")
            if open_brackets > 0:
                raw += "]" * open_brackets
            if open_braces > 0:
                raw += "}" * open_braces
        else:
          
            return "{}"
    
    return raw


def llm_pass_a(jd_text: str, client: OllamaClient, candidates: list[dict], job_id: str) -> list[dict]:
    """Extrait les nœuds du JD en utilisant les candidats par lots de 10."""
    all_nodes = []
    batch_size = 10
    
   
    target_candidates = candidates[:30]
    
    for i in range(0, len(target_candidates), batch_size):
        batch = target_candidates[i:i+batch_size]
        print(f"    [Pass A] Traitement lot {i//batch_size + 1}/{(len(target_candidates)-1)//batch_size + 1}")
        
        hints_str = "\n".join(
            f"  - [{c['provisional_type']}] {c['text']} (conf={c['confidence']})"
            for c in batch
        )
        
        prompt = (
            f"JOB DESCRIPTION:\n{jd_text[:JD_MAX_CHARS]}\n\n"
            f"EXTRACTION HINTS (BATCH):\n{hints_str}\n\n"
            f"Extract all meaningful job requirements as graph nodes for these {len(batch)} hints."
        )
        
        try:
            resp = client.chat_completion(
                messages=[
                    {"role": "system", "content": PASS_A_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.05,
                max_tokens=2500
            )
            raw = resp["choices"][0]["message"]["content"]
            data = json.loads(_clean_json_response(raw))
            nodes = data.get("nodes", [])
            all_nodes.extend(nodes)
        except Exception as e:
            print(f"    [Pass A] Erreur sur lot {i//batch_size + 1} : {e}")

    
    unique_nodes = {n["id"]: n for n in all_nodes if "id" in n}.values()
    print(f"  [Pass A] {len(unique_nodes)} nœuds uniques extraits au total")
    return list(unique_nodes)



def llm_pass_b(nodes: list[dict], jd_text: str, client: OllamaClient, job_id: str) -> list[dict]:
    if not nodes:
        print(f"  [Pass B] ⚠️ Aucun nœud — pas d'arêtes")
        return []

  
    top_nodes = sorted(nodes, key=lambda n: n.get("w_0", 0), reverse=True)[:20]

    nodes_summary = json.dumps(
        [{"id": n["id"], "label": n.get("label", n["id"]),
          "type": n.get("type"), "w_0": n.get("w_0")} for n in top_nodes],
        ensure_ascii=False
    )
    prompt = (
        f"NODES:\n{nodes_summary}\n\n"
        f"JOB CONTEXT:\n{jd_text[:1500]}\n\n"
        "Generate semantic edges between these nodes."
    )

    raw = None
    edges = []

   
    try:
        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": PASS_B_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=0.05,
            max_tokens=4096
        )
        raw = resp["choices"][0]["message"]["content"]
        cleaned = _clean_json_response(raw)
        data = json.loads(cleaned)
        edges = data.get("edges", [])
        print(f"  [Pass B] ✅ {len(edges)} arêtes générées (client principal)")
        return edges

    except json.JSONDecodeError:
       
        print(f"  [Pass B] ⚠️ JSON invalide, tentative d'extraction partielle...")
        if raw:
            partial = re.findall(
                r'\{[^{}]*"source"\s*:\s*"[^"]+"\s*,[^{}]*"target"\s*:\s*"[^"]+"\s*,[^{}]*\}',
                raw
            )
            for p in partial:
                try:
                    edges.append(json.loads(p))
                except json.JSONDecodeError:
                    continue
        if edges:
            print(f"  [Pass B] ✅ {len(edges)} arêtes partielles extraites")
            return edges

    except Exception as e:
        print(f"  [Pass B] ❌ Erreur client : {str(e)[:80]}")

  
    if not edges and USE_NVIDIA_API:
        print(f"  [Pass B] ↪️ Fallback vers Ollama local...")
        try:
            ollama_client = OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
            resp = ollama_client.chat_completion(
                messages=[
                    {"role": "system", "content": PASS_B_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.05,
                max_tokens=3000
            )
            raw = resp["choices"][0]["message"]["content"]
            cleaned = _clean_json_response(raw)
            data = json.loads(cleaned)
            edges = data.get("edges", [])
            print(f"  [Pass B] ✅ {len(edges)} arêtes via Ollama fallback")
            return edges
        except Exception as e2:
            print(f"  [Pass B] ⚠️ Ollama fallback échoué : {str(e2)[:60]}")

  
    print(f"  [Pass B] 🔧 Génération auto d'arêtes...")
    for i, n1 in enumerate(top_nodes):
        for n2 in top_nodes[i+1:]:
            
            if n1.get("type") != n2.get("type"):
                edges.append({
                    "source": n1["id"],
                    "target": n2["id"],
                    "relation": "supports",
                    "weight": 0.5,
                    "rationale": "Relation auto-générée (arête de fallback)"
                })

    print(f"  [Pass B] ✅ {len(edges)} arêtes auto-générées (fallback)")
    return edges




def build_extractor(vocab: VocabManager) -> FusedExtractor:
    regex_ext   = RegexExtractor()
    gliner_ext  = GlinerExtractor() if True else None
    gliner_v2   = GlinerV2Extractor() if True else None
    jobbert_ext = JobBertExtractor() if True else None
    return FusedExtractor(regex_ext, gliner_ext, gliner_v2, jobbert_ext, vocab)


def phase1_generate_graph(
    jd_text: str,
    extractor: FusedExtractor,
    llm_client: OllamaClient,
    vocab: VocabManager,
    registry: SkillRegistry,
    job_id: str = "unknown",
) -> tuple[nx.DiGraph, dict | None]:
    print(f"\n{'─'*50}")
    print(f"  [Phase 1] Génération graphe — {job_id}")

    lang = detect_language(jd_text)
    if extractor._jobbert and extractor._jobbert.available:
        extractor._jobbert.set_language(lang)
    print(f"  Langue : {lang}")

    fused_candidates = extractor.extract_all_nodes(jd_text)
    nodes = llm_pass_a(jd_text, llm_client, fused_candidates, job_id)
    if not nodes:
        print("  [Phase 1] Aucun nœud — abandon.")
        return nx.DiGraph(), None

    nodes = semantic_dedup_nodes(nodes)
    nodes = registry_boost(nodes, registry)
    registry.register_nodes(nodes, job_id)

    edges = llm_pass_b(nodes, jd_text, llm_client, job_id)

    nodes = prune_low_weight_nodes(nodes, edges)
    edges = edge_quality_filter(edges, nodes)

    vocab.update_from_llm_nodes(nodes)

    G = nx.DiGraph()
    node_ids = set()
    for n in nodes:
        G.add_node(
            n["id"],
            label=n.get("label", n["id"]),
            type=n.get("type", "Hard Skill"),
            w_0=n.get("w_0", 0.5),
            seniority=n.get("seniority_required", "any"),
            context=n.get("context", ""),
            inferred=n.get("inferred", False),
        )
        node_ids.add(n["id"])

    skipped = 0
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s in node_ids and t in node_ids:
            G.add_edge(s, t,
                       relation=e.get("relation", "related"),
                       weight=e.get("weight", 0.5),
                       rationale=e.get("rationale", ""))
        else:
            skipped += 1

    if skipped:
        print(f"  {skipped} arête(s) ignorée(s)")

    graph_data = {"nodes": nodes, "edges": edges, "lang": lang, "job_id": job_id}
    print(f"  ✅ Graphe : {G.number_of_nodes()} nœuds | {G.number_of_edges()} arêtes")
    return G, graph_data


def _make_safe_job_id(row: pd.Series, id_column, idx: int) -> str:
    if id_column and id_column in row.index:
        val = str(row[id_column]).strip()
        if val and val != "nan" and len(val) <= 60:
            return re.sub(r'[^\w\-]', '_', val)[:50]
    return f"job_{idx}"

def run_phase1_graph_generation(
    csv_path: str = CSV_INPUT,
    jd_column: str = JD_COLUMN,
    id_column: str = ID_COLUMN,
    output_dir: str = GRAPHS_DIR,
    test_mode: bool = TEST_MODE,
    max_jobs: int = TEST_MAX_JOBS,
) -> Dict[str, dict]:
    print("\n" + "█" * 70)
    print("█  PHASE 1 — GÉNÉRATION DES GRAPHES IDÉAUX")
    print(f"█  Mode : {'🧪 TEST' if test_mode else '🚀 COMPLET'}")
    print("█" * 70)

    os.makedirs(output_dir, exist_ok=True)
    vocab     = VocabManager(VOCAB_PATH)
    registry  = SkillRegistry(REGISTRY_PATH)
    extractor = build_extractor(vocab)
    llm       = make_llm_client()

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"  ❌ Erreur CSV : {e}")
        return {}

    if test_mode:
        df = df.head(max_jobs)
        print(f"  🧪 TEST : {len(df)} jobs seulement")

    results: Dict[str, dict] = {}
    stats = {"success": 0, "skipped": 0, "failed": 0}

    for idx, row in df.iterrows():
        job_id  = _make_safe_job_id(row, id_column, idx)
        jd_text = row.get(jd_column, "")
        if not jd_text or pd.isna(jd_text):
            stats["skipped"] += 1
            continue

        out_path = os.path.join(output_dir, f"{job_id}.json")
        if os.path.exists(out_path):
            print(f"  ⏩ {job_id} — graphe existant")
            with open(out_path, encoding="utf-8") as f:
                results[job_id] = json.load(f)
            GRAPHS_STORE[job_id] = results[job_id]
            stats["success"] += 1
            continue

        try:
            G, data = phase1_generate_graph(
                str(jd_text), extractor, llm, vocab, registry, job_id
            )
            if data:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                results[job_id] = data
                GRAPHS_STORE[job_id] = data
                stats["success"] += 1
            else:
                stats["failed"] += 1
        except Exception as e:
            print(f"  ❌ Exception {job_id} : {e}")
            stats["failed"] += 1

    print(f"\n  Bilan Phase 1 : {stats['success']} ✅ | {stats['failed']} ❌ | {stats['skipped']} ⏩")
    return results




_NLP_CV = None

def _get_nlp_cv():
    global _NLP_CV
    if _NLP_CV is None and _SPACY_AVAILABLE:
        try:
            _NLP_CV = spacy.load("en_core_web_sm")
        except Exception:
            try:
                _NLP_CV = spacy.load("fr_core_news_md")
            except Exception:
                pass
    return _NLP_CV


def extract_cv_sections(cv_text: str) -> Dict[str, str]:
    SECTION_HEADERS = {
        "summary":    r"(?i)^(summary|profile|about\s*me|objective|overview)\s*:?\s*$",
        "experience": r"(?i)^(experience|work\s*experience|employment)\s*:?\s*$",
        "projects":   r"(?i)^(projects?|personal\s*projects?)\s*:?\s*$",
        "education":  r"(?i)^(education|academic|degrees?)\s*:?\s*$",
        "skills":     r"(?i)^(skills?|technical\s*skills?|competencies)\s*:?\s*$",
        "certs":      r"(?i)^(certifications?|certificates?)\s*:?\s*$",
    }
    lines = cv_text.split("\n")
    sections = defaultdict(list)
    current = "preamble"
    for line in lines:
        matched = False
        for name, pattern in SECTION_HEADERS.items():
            if re.match(pattern, line.strip()):
                current = name
                matched = True
                break
        if not matched:
            sections[current].append(line)
    result = {name: "\n".join(ls).strip()[:2500]
              for name, ls in sections.items() if "\n".join(ls).strip()}
    if len(result) <= 1:
        chunk = len(cv_text) // 3
        result = {
            "part_1": cv_text[:chunk],
            "part_2": cv_text[chunk:2 * chunk],
            "part_3": cv_text[2 * chunk:]
        }
    return result


def extract_experience_years(cv_text: str) -> Dict[str, int]:
    patterns = [
        r"(\d+)\+?\s*(?:years?|ans?)\s+(?:of\s+)?(?:experience\s+(?:with|in|using))?\s*([A-Za-z][A-Za-z0-9\+\#\.\-\s]{2,25})",
        r"([A-Za-z][A-Za-z0-9\+\#\.\-\s]{2,20})\s*[\(\-]\s*(\d+)\+?\s*(?:years?|ans?)\s*[\)\-]?",
    ]
    exp_map: Dict[str, int] = {}
    for pattern in patterns:
        for m in re.finditer(pattern, cv_text, re.IGNORECASE):
            groups = m.groups()
            if groups[0].isdigit():
                years, skill = int(groups[0]), groups[1].strip().lower()
            else:
                skill, years = groups[0].strip().lower(), int(groups[1])
            if 0 < years <= 30 and len(skill) > 2:
                existing = exp_map.get(skill, 0)
                exp_map[skill] = max(existing, years)
    return exp_map


def extract_ner_hints(cv_text: str) -> Dict[str, list]:
    nlp = _get_nlp_cv()
    if nlp is None:
        return {}
    doc = nlp(cv_text[:8000])
    hints: Dict[str, list] = {"ORG": [], "DATE": [], "PRODUCT": [], "GPE": []}
    seen = set()
    for ent in doc.ents:
        if ent.label_ not in hints:
            continue
        val = ent.text.strip()
        if len(val) < 3 or val.isdigit():
            continue
        key = (ent.label_, val.lower())
        if key in seen:
            continue
        seen.add(key)
        ctx = cv_text[max(0, ent.start_char - 50):ent.end_char + 50].replace("\n", " ").strip()
        hints[ent.label_].append({"entity": val, "context": ctx[:120]})
    return {k: v[:8] for k, v in hints.items() if v}


def extract_action_verbs(cv_text: str) -> Dict[str, list]:
    PATTERNS = {
        "Leadership":      r"\b(led|managed|directed|supervised|mentored)\b",
        "Technical_Build": r"\b(built|developed|architected|designed|implemented)\b",
        "Technical_Ops":   r"\b(automated|optimized|migrated|scaled|refactored)\b",
        "Collaborative":   r"\b(collaborated|partnered|coordinated|facilitated)\b",
        "Problem_Solving": r"\b(solved|resolved|improved|reduced|eliminated)\b",
        "Data_Analytics":  r"\b(analyzed|modeled|visualized|queried|processed)\b",
    }
    sentences = [s.strip() for s in re.split(r"[.\n]", cv_text) if s.strip()]
    result = {}
    for category, pattern in PATTERNS.items():
        found = []
        for sent in sentences:
            m = re.search(pattern, sent, re.IGNORECASE)
            if m:
                found.append({"verb": m.group(0).lower(), "sentence": sent[:130]})
        if found:
            result[category] = found[:5]
    return result


def extract_quantified_metrics(cv_text: str) -> list:
    METRIC_PATTERNS = [
        r"\d+[\+]?\s*(?:years?|yrs?)",
        r"\d+\s*%\s*(?:increase|decrease|improvement)",
        r"\$\s*\d+[.,]?\d*\s*(?:million|thousand|k|M)",
        r"(?:reduced|improved|increased)\s+\w+\s+by\s+\d+",
    ]
    metrics, seen = [], set()
    for pattern in METRIC_PATTERNS:
        for m in re.finditer(pattern, cv_text, re.IGNORECASE):
            val = m.group(0).strip()
            if val.lower() in seen:
                continue
            seen.add(val.lower())
            ctx = cv_text[max(0, m.start() - 60):m.end() + 60].strip()
            metrics.append({"metric": val, "context": ctx[:120]})
    return metrics[:15]




class CvLLM:
    """Client LLM (NVIDIA ou Ollama selon config)."""

    def __init__(self, model: str = None):
        if USE_NVIDIA_API and NVIDIA_API_KEY:
            self.client = NvidiaClient(api_key=NVIDIA_API_KEY, model=NVIDIA_MODEL)
            self.model = NVIDIA_MODEL
        else:
            self.client = OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
            self.model = OLLAMA_MODEL
        print(f"  ✅ CvLLM : {self.model}")

    def _repair_json(self, text: str) -> str:
        text = re.sub(r"```json\s*|\s*```", "", text).strip()
        s = text.find("{")
        if s == -1: s = text.find("[")
        e = text.rfind("}")
        if e == -1: e = text.rfind("]")
        
        if s != -1 and e != -1:
            text = text[s:e + 1]
            
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        
        # Fermer les blocs tronqués
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        if open_brackets > 0: text += "]" * open_brackets
        if open_braces > 0: text += "}" * open_braces
        
        return text

    def generate(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.05, max_tokens: int = 3000,
                 is_json: bool = True, max_retries: int = 3) -> str:
        JSON_RULES = (
            "\nOUTPUT RULES: Return ONLY valid JSON. No markdown. "
            "No preamble. Double quotes everywhere. "
            "Close all brackets.\n"
        )
        sp = system_prompt + (JSON_RULES if is_json else "")
        if len(sp) > MAX_SYSTEM_CHARS:
            sp = sp[:MAX_SYSTEM_CHARS]
        budget = MAX_PROMPT_CHARS - len(sp) - 200
        if len(prompt) > budget:
            prompt = prompt[:budget]

        messages = [
            {"role": "system", "content": sp},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(max_retries):
            try:
                resp = self.client.chat_completion(
                    messages=messages, temperature=temperature
                )
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
                            {"role": "user", "content": "Invalid JSON. Return ONLY the corrected JSON."}
                        ]
            except Exception as ex:
                wait = 2 ** attempt
                print(f"    ⚠ LLM error (attempt {attempt + 1}): {str(ex)[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(wait)

        return '{"skill_audit_instructions": []}' if is_json else ""




PHASE2_SYSTEM_PROMPT = """You are a Principal Talent Strategist. Analyze the CV against the job requirements.
Goal: generate AUDIT INSTRUCTIONS only — no scores, no summaries.

AUDIT DIMENSIONS (apply per skill, hard or soft):
1. PROJECT PROOF     – Used in a real project? Estimate project complexity: [low|mid|high|critical].
2. EXPERIENCE        – Total active years with this skill. Recency: date of last use.
3. CERTIFICATION     – Any cert listed? Flag if unverifiable or expired.
4. INTENSITY         – Was this skill central or peripheral in each role/project?
5. SENIORITY         – Infer level [junior|mid|senior|expert] from scope, autonomy, outcomes.
6. EXTERNAL PROOF    – Publications, talks, OSS reviews, patents, or expert-validated work?
7. OUTCOMES          – Any quantified result tied to this skill (%, $, scale, perf gain)?
8. INFERABLE SCORE   – If skill is a known standalone tool/library (e.g. Pandas, Docker),
                       signal can exist even with no explicit project mention → flag as [inferred_standalone].

RED FLAGS (auto-detect):
- recency_gap: last use > 5 years ago
- ghost_skill: mentioned ≥2 times but never anchored to a project or outcome
- cert_without_application: certification present but zero project evidence

SOFT SKILL SPECIAL RULES (apply if skill_type = soft):
- Validate via: hackathons (weight international > local), competitions, exchange programs
- Communication skills → check language level stated or implied
- Autonomy → count self-initiated or personal projects
- Leadership/teamwork → look for event/program-based proof, not just job titles

OUTPUT: strict JSON only — no markdown, no preamble.
{
  "skill_audit_instructions": [
    {
      "skill_id": "<id>",
      "skill_type": "hard|tool|soft",
      "evidence_verdict": "strong|moderate|weak|absent|inferred|inferred_standalone",
      "project_complexity": "low|mid|high|critical|none",
      "experience_years_detected": 0,
      "recency_last_used": "<year or null>",
      "intensity": "central|supporting|peripheral|unknown",
      "seniority_signal": "junior|mid|senior|expert|unknown",
      "certification": { "found": false, "verifiable": false },
      "external_proof": false,
      "quantified_outcome": "<outcome string or null>",
      "red_flags": [],
      "verification_question": "<specific, falsifiable question>",
      "evidence_anchor": "<CV fragment, max 130 chars>"
    }
  ]
}"""




PHASE2_EDGES_SYSTEM_PROMPT = """You are a Principal Talent Strategist. Analyze skill synergies in the CV.
Goal: Find evidence the candidate used BOTH skills TOGETHER in the same role or project.

EDGE ANALYSIS RULES:
1. SYNERGY DETECTION:
   - Same sentence or bullet point = strong synergy
   - Same project/role but different bullets = moderate synergy
   - Same time period but different projects = weak synergy (temporal overlap only)

2. EVIDENCE EXTRACTION:
   - Look for explicit mentions (e.g. "used Python AND SQL to build...")
   - Look for implicit connections (complementary skills in same context)
   - Extract exact CV phrase showing both skills together

3. TEMPORAL OVERLAP:
   - Check if skills were used in overlapping time periods
   - Same role date range = true
   - Consecutive roles = true only if explicitly mentioned together

4. SAME PROJECT/ROLE:
   - Both mentioned in same role/project description = true
   - Both mentioned in same accomplishment = true
   - Only temporal overlap (different projects) = false

OUTPUT: strict JSON only — no markdown, no preamble.
{
  "edges": [
    {
      "source": "<source skill id>",
      "target": "<target skill id>",
      "synergy_found": true | false,
      "synergy_evidence": {
        "found": true | false,
        "cv_excerpt": "<exact phrase from CV or null>",
        "temporal_overlap": true | false,
        "same_project_or_role": true | false
      },
      "instruction": {
        "rationale": "<1-3 sentences explaining the synergy or absence>"
      }
    }
  ]
}"""


def build_graph_from_data(job_data: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for n in job_data.get("nodes", []):
        G.add_node(n["id"],
                   label=n.get("label", n["id"]),
                   type=n.get("type", "Hard Skill"),
                   w_0=n.get("w_0", 0.5),
                   seniority=n.get("seniority_required", "any"))
    for e in job_data.get("edges", []):
        G.add_edge(e["source"], e["target"],
                   weight=e.get("weight", 0.5),
                   relation=e.get("relation", "related"))
    return G


def extract_edges_instructions(edges: list[dict], node_info: dict, cv_text: str, llm: CvLLM) -> dict:
    """Extrait les instructions pour les edges (synergies de compétences)."""
    if not edges:
        return {}

    all_edge_instructions = {}
    batch_size = 5
    total_edges = len(edges)
    batch_count = (total_edges + batch_size - 1) // batch_size

    for i in range(0, total_edges, batch_size):
        batch = edges[i:i+batch_size]
        batch_num = i // batch_size + 1
        print(f"    [Phase 2 - Edges] Lot {batch_num}/{batch_count} ({len(batch)} synergies)...", end=" ", flush=True)

       
        edges_summary = "\n".join(
            f'  - {e["source"]} ({node_info.get(e["source"], {}).get("label", e["source"])}) '
            f'--[{e.get("relation", "related")}]--> '
            f'{e["target"]} ({node_info.get(e["target"], {}).get("label", e["target"])})'
            for e in batch
        )

        expected_pairs = json.dumps([f'{e["source"]}|{e["target"]}' for e in batch])

        prompt = f"""CV TEXT:
{cv_text}

SKILL PAIRS TO ANALYZE (keys must match: {expected_pairs}):
{edges_summary}

For each pair, find evidence showing BOTH skills used TOGETHER in the same role or project.

Generate audit instructions for skill synergies.
"""

        raw = llm.generate(prompt, PHASE2_EDGES_SYSTEM_PROMPT, temperature=0.1)
        try:
            data = json.loads(_clean_json_response(raw))
            edge_instr = data.get("edges", [])
            for item in edge_instr:
                if isinstance(item, dict):
                    key = f"{item.get('source')}|{item.get('target')}"
                    all_edge_instructions[key] = item
            print(f"✓ {len(edge_instr)}/{len(batch)}")
        except Exception as e:
            print(f"✗ Erreur: {str(e)[:50]}")
           
            for edge in batch:
                key = f"{edge['source']}|{edge['target']}"
                all_edge_instructions[key] = {
                    "source": edge["source"],
                    "target": edge["target"],
                    "synergy_found": False,
                    "synergy_evidence": {
                        "found": False,
                        "cv_excerpt": None,
                        "temporal_overlap": False,
                        "same_project_or_role": False
                    },
                    "instruction": {"rationale": "Analysis skipped due to error"}
                }

    return all_edge_instructions


def run_phase2_audit_strategy(
    job_data: dict,
    cv_text: str,
    llm: CvLLM,
    job_id: str = "unknown",
    cv_id: str = "cv_1",
) -> dict:
    sections       = extract_cv_sections(cv_text)
    ner_hints      = extract_ner_hints(cv_text)
    action_verbs   = extract_action_verbs(cv_text)
    metrics        = extract_quantified_metrics(cv_text)
    exp_years      = extract_experience_years(cv_text)

    nodes = job_data.get("nodes", [])
    edges = job_data.get("edges", [])
    all_instructions = []

  
    print(f"  ── PASS 1: Analyse des nodes ({len(nodes)} compétences)")
    batch_size = 10
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i+batch_size]
        batch_num = i // batch_size + 1
        batch_total = (len(nodes) + batch_size - 1) // batch_size
        print(f"    [Phase 2-A] Lot {batch_num}/{batch_total} ({len(batch)} compétences)...", end=" ", flush=True)

        skills_summary = json.dumps(
            [{"id": n["id"], "label": n.get("label"), "type": n.get("type"), "w_0": n.get("w_0")}
             for n in batch],
            ensure_ascii=False
        )

        prompt = f"""
JOB SKILLS (BATCH): {skills_summary}

CV SECTIONS:
{json.dumps({k: v[:1200] for k, v in sections.items()}, ensure_ascii=False, indent=2)}

ACTION VERBS: {json.dumps(action_verbs, ensure_ascii=False)}

METRICS: {json.dumps(metrics, ensure_ascii=False)}

Generate audit instructions for these {len(batch)} skills.
"""
        raw = llm.generate(prompt, PHASE2_SYSTEM_PROMPT, temperature=0.1)
        try:
            data = json.loads(_clean_json_response(raw))
            batch_instr = data.get("skill_audit_instructions", [])
            all_instructions.extend(batch_instr)
            print(f"✓ {len(batch_instr)}/{len(batch)}")
        except Exception as e:
            print(f"✗ Erreur: {str(e)[:50]}")

    print(f"  ✓ {len(all_instructions)} instructions de nodes générées")

    
    print(f"\n  ── PASS 2: Analyse des synergies ({len(edges)} edges)")
    node_info = {
        n["id"]: {"label": n.get("label", n["id"]), "type": n.get("type", "?")}
        for n in nodes
    }
    edge_instructions = extract_edges_instructions(edges, node_info, cv_text, llm)
    print(f"  ✓ {len(edge_instructions)} instructions d'edges générées")

    return {
        "job_id": job_id, "cv_id": cv_id,
        "skill_instructions": {i.get("skill_id"): i for i in all_instructions},
        "edge_instructions": edge_instructions,
        "cv_sections": sections,
        "ner_hints": ner_hints,
        "metrics": metrics,
        "exp_years": exp_years,
        "edges_list": edges,
    }


def run_phase2(
    csv_path: str = CSV_INPUT,
    jd_column: str = JD_COLUMN_SCORING,
    id_column: str = ID_COLUMN,
    cv_prefix: str = CV_PREFIX,
    test_mode: bool = TEST_MODE,
    max_jobs: int = TEST_MAX_JOBS,
    max_cvs: int = TEST_MAX_CVS,
) -> Dict[str, dict]:
    print("\n" + "█" * 70)
    print("█  PHASE 2 — AGENT DE RÉFLEXION")
    print(f"█  Mode : {'🧪 TEST' if test_mode else '🚀 COMPLET'}")
    print("█" * 70)

    if not GRAPHS_STORE:
        print("  ❌ GRAPHS_STORE vide")
        return {}

    llm = CvLLM()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"  ❌ Erreur CSV : {e}")
        return {}

    if test_mode:
        df = df.head(max_jobs)

    results: Dict[str, dict] = {}
    cv_columns = sorted([c for c in df.columns if c.startswith(CV_PREFIX)])

    for idx, row in df.iterrows():
        job_id  = _make_safe_job_id(row, id_column, idx)
        job_data = GRAPHS_STORE.get(job_id)
        if not job_data:
            print(f"  ⏩ {job_id} — graphe manquant")
            continue

        cv_cols_to_process = cv_columns[:max_cvs] if test_mode else cv_columns
        for cv_col in cv_cols_to_process:
            cv_text = str(row.get(cv_col, ""))
            if len(cv_text.strip()) < 50 or cv_text == "nan":
                continue
            pair_key = f"{job_id}_{cv_col}"
            print(f"\n  ── Paire : {pair_key}")
            audit = run_phase2_audit_strategy(job_data, cv_text, llm, job_id, cv_col)
            if audit:
                results[pair_key] = audit
                PHASE2_STORE[pair_key] = audit

            
                out_path = os.path.join(RESULTS_DIR, f"{pair_key}_phase2.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(audit, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Phase 2 : {len(results)} paires")
    return results



PHASE3_SYSTEM_PROMPT = """You are a Senior Technical Recruiter. You receive per-skill audit instructions from Phase 2.
Your goal: convert each audit into a calibrated score using ONLY the evidence signals already extracted.

  0.00 = Absent — no mention anywhere
  0.25 = Listed in skills only, zero project anchor
  0.50 = Contextual mention, no concrete project evidence
  0.75 = Used in ≥1 project, basic outcome mentioned
  1.00 = Multi-project use, at least one quantified outcome
  1.25 = Deep expertise, architectural or design-level decisions
  1.50 = Lead-level — team ownership, cross-functional impact
  1.75 = Domain expert — published, presented, or taught externally
  2.00 = Core specialty — world-class mastery evidenced

DIMENSION BONUSES (additive, cap: 2.00):
  +0.25 : certification.verifiable = true
  +0.05 : experience_years_detected (cap +0.30)
  +0.15 : external_proof = true (publication, talk, OSS review)
  +0.10 : quantified_outcome is non-null
  +0.10 : intensity = central (not peripheral)
  +0.10 : project_complexity = high OR critical

SOFT SKILL OVERRIDE (skill_type = soft):
  Base score from: hackathon/competition proof > exchange program > language level > own projects
  Apply same bonuses where applicable. No project_complexity bonus.

INFERRED STANDALONE (evidence_verdict = inferred_standalone):
  Score as 0.50 base max — tool is known but project application unconfirmed.

RED FLAG PENALTIES (cumulative):
  recency_gap         → −0.25
  ghost_skill         → −0.20
  cert_without_application → −0.10

RULES:
- Drive every score from the audit fields, not from re-reading the CV.
- Rationale must cite the evidence_anchor and active audit signals.
- Flag each active bonus and penalty explicitly in score_breakdown.

OUTPUT: strict JSON only — no markdown, no preamble.
{
  "scored_nodes": {
    "<skill_id>": {
      "w_cv": 0.85,
      "evidence_quality": "strong|moderate|weak|absent|inferred|inferred_standalone",
      "score_breakdown": {
        "base_score": 0.75,
        "bonuses": {
          "cert_bonus": 0.0,
          "experience_bonus": 0.0,
          "external_proof_bonus": 0.0,
          "outcome_bonus": 0.10,
          "intensity_bonus": 0.0,
          "complexity_bonus": 0.0
        },
        "penalties": {
          "recency_gap": 0.0,
          "ghost_skill": 0.0,
          "cert_without_application": 0.0
        },
        "final_score": 0.85
      },
      "active_red_flags": [],
      "rationale": "<cite evidence_anchor + which signals drove the score>"
    }
  }
}"""



PHASE3_EDGES_SYSTEM_PROMPT = """You are a Senior Technical Recruiter. You receive skill synergy audit instructions from Phase 2.
Your goal: score each edge (skill pair synergy) based on evidence of collaboration and mutual reinforcement.

EDGE SYNERGY SCORING SCALE:
  0.00 = No synergy — skills never mentioned together or one skill is missed
  0.25 = Weak synergy — temporal overlap only, different projects
  0.75 = Strong synergy — same sentence, same accomplishment


SYNERGY SIGNAL WEIGHTS (additive, cap: 1.00):
  +0.20  synergy_found = true (base synergy evidence)
  +0.15  same_project_or_role = true (confirmed collaboration)
  +0.15  temporal_overlap = true (time period alignment)
  +0.25  output/outcome demonstrates synergy (e.g., "built X using Python AND SQL")

RULES:
- Every synergy score must cite the cv_excerpt or reason for absence
- Rationale must explain how the two skills interact or fail to interact
- Flag if this is a "ghost synergy" (claimed but not evidenced)

OUTPUT: strict JSON only — no markdown, no preamble.
{
  "scored_edges": {
    "<source>|<target>": {
      "synergy_score": 0.65,
      "evidence_quality": "strong|moderate|weak|absent",
      "score_breakdown": {
        "base_synergy": 0.20,
        "evidence_bonus": 0.10,
        "penalties": 0.0,
        "final_score": 0.65
      },
      "is_ghost_synergy": false,
      "rationale": "<explain how skills work together and cite cv_excerpt>"
    }
  }
}"""


def run_phase3_scoring(
    phase2_output: dict,
    llm: CvLLM,
) -> dict:
    skill_instructions = phase2_output.get("skill_instructions", {})
    edge_instructions =phase2_output.get("edge_instructions", {})
    cv_sections = phase2_output.get("cv_sections", {})
    exp_years = phase2_output.get("exp_years", {})
    job_id = phase2_output.get("job_id", "?")
    cv_id  = phase2_output.get("cv_id", "?")

    instructions_list = list(skill_instructions.values())
    nb = len(instructions_list)
    BATCH_SIZE = 6 if nb > 20 else 10

    all_scored_nodes: Dict[str, dict] = {}
    ok_batches = 0

    for i in range(0, nb, BATCH_SIZE):
        batch = instructions_list[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        prompt = f"""
EXPERIENCE YEARS:
{json.dumps(exp_years, ensure_ascii=False)}

CV SECTIONS:
{json.dumps({k: v[:1000] for k, v in cv_sections.items()}, ensure_ascii=False, indent=2)}

SKILLS TO SCORE (batch {batch_num}):
{json.dumps(batch, ensure_ascii=False, indent=2)}

Score each skill based on CV evidence.
"""

        raw = llm.generate(prompt, PHASE3_SYSTEM_PROMPT, temperature=0.05, max_tokens=3500)
        try:
            data = json.loads(raw)
            scored = data.get("scored_nodes", {})
            # ── Dériver p3_confidence depuis evidence_quality (déjà renvoyé par le LLM) ──
            for nid, nd in scored.items():
                eq = str(nd.get("evidence_quality", "")).lower()
                nd["p3_confidence"] = (
                    "HIGH"   if eq == "strong"   else
                    "MEDIUM" if eq == "moderate" else
                    "LOW"    if eq in ("weak", "inferred", "inferred_standalone") else
                    "NONE"   # absent ou champ manquant
                )
                nd["p3_source"] = "llm"
            all_scored_nodes.update(scored)
            print(f"    Batch {batch_num} : ✅ {len(scored)} scores")
            ok_batches += 1
        except json.JSONDecodeError as e:
            # ── Fallback : créer des entrées nulles plutôt que de perdre ces skills ──
            print(f"    Batch {batch_num} : ❌ JSON invalide — fallback sur {len(batch)} skills")
            for skill in batch:
                sid = skill.get("skill_id") or skill.get("id") or f"unknown_{batch_num}"
                if sid not in all_scored_nodes:
                    all_scored_nodes[sid] = {
                        "w_cv": 0.0,
                        "evidence_quality": "absent",
                        "score_breakdown": {"base_score": 0.0, "final_score": 0.0,
                                            "bonuses": {}, "penalties": {}},
                        "active_red_flags": [],
                        "rationale": "Fallback — batch JSON invalide",
                        "p3_confidence": "NONE",
                        "p3_source": "fallback",
                    }

        if i + BATCH_SIZE < nb:
            time.sleep(1)

    scores = [v.get("w_cv", 0) for v in all_scored_nodes.values()]
    print(f"\n  [Phase 3] {job_id}×{cv_id} : {len(all_scored_nodes)} skills scorés")

    fail_batches = (nb + BATCH_SIZE - 1) // BATCH_SIZE - ok_batches

 
    all_scored_edges = {}
    edge_scores = []

    if edge_instructions:
        all_scored_edges = score_edges_phase3(edge_instructions, all_scored_nodes, cv_sections, llm)
        edge_scores = [v.get("synergy_score", 0) for v in all_scored_edges.values()]

  
    n_fallback_nodes = sum(1 for v in all_scored_nodes.values() if v.get("p3_source") == "fallback")
    n_total_nodes    = max(len(all_scored_nodes), 1)
    fallback_ratio   = round(n_fallback_nodes / n_total_nodes, 4)

    return {
        "job_id": job_id, "cv_id": cv_id,
        "scored_nodes": all_scored_nodes,
        "scored_edges": all_scored_edges,
        "metadata": {
            "skills_scored": len(all_scored_nodes),
            "mean_score": round(sum(scores) / len(scores), 3) if scores else 0,
            "edges_scored": len(all_scored_edges),
            "mean_synergy": round(sum(edge_scores) / len(edge_scores), 3) if edge_scores else 0,
            "fallback_ratio": fallback_ratio,
            "n_fallback_nodes": n_fallback_nodes,
            "synergy_distribution": {
                "mean": round(sum(edge_scores) / len(edge_scores), 3) if edge_scores else 0,
                "max": round(max(edge_scores), 3) if edge_scores else 0,
                "min": round(min(edge_scores), 3) if edge_scores else 0,
            },
            "batches_ok": ok_batches, "batches_fail": fail_batches,
            "score_distribution": {
                "mean": round(sum(scores) / len(scores), 3) if scores else 0,
                "max": round(max(scores), 3) if scores else 0,
                "min": round(min(scores), 3) if scores else 0,
                "above_1": sum(1 for s in scores if s >= 1.0),
                "below_05": sum(1 for s in scores if s < 0.5),
            }
        }
    }


def score_edges_phase3(
    edge_instructions: dict,
    scored_nodes: dict,
    cv_sections: dict,
    llm: CvLLM,
) -> dict:
    """Score edges (skill synergies) based on Phase 2 edge instructions and Phase 3 node scores."""
    if not edge_instructions:
        return {}

    edges_list = list(edge_instructions.values())
    all_scored_edges: Dict[str, dict] = {}
    batch_size = 4
    total_edges = len(edges_list)
    batch_count = (total_edges + batch_size - 1) // batch_size

    print(f"\n  ── PASS 2: Scoring synergies ({total_edges} edges)")

    for i in range(0, total_edges, batch_size):
        batch = edges_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"    [Phase 3-B] Lot {batch_num}/{batch_count}...", end=" ", flush=True)

        enriched_batch = []
        for edge in batch:
            src = edge.get("source")
            tgt = edge.get("target")
            edge_copy = edge.copy()
            edge_copy["parent_source_score"] = scored_nodes.get(src, {}).get("w_cv", 0.0)
            edge_copy["parent_target_score"] = scored_nodes.get(tgt, {}).get("w_cv", 0.0)
            enriched_batch.append(edge_copy)

        prompt = f"""
PARENT NODE SCORES (for context):
{json.dumps({k: v.get("w_cv", 0) for k, v in list(scored_nodes.items())[:20]}, ensure_ascii=False)}

CV SECTIONS:
{json.dumps({k: v[:800] for k, v in cv_sections.items()}, ensure_ascii=False, indent=2)}

EDGES TO SCORE (batch {batch_num}):
{json.dumps(enriched_batch, ensure_ascii=False, indent=2)}

Score each edge synergy based on evidence of collaboration.
"""

        raw = llm.generate(prompt, PHASE3_EDGES_SYSTEM_PROMPT, temperature=0.05, max_tokens=2500)
        try:
            data = json.loads(_clean_json_response(raw))
            scored = data.get("scored_edges", {})
           
            for ekey, ed in scored.items():
                eq = str(ed.get("evidence_quality", "")).lower()
                ed["p3_confidence"] = (
                    "HIGH"   if eq == "strong"   else
                    "MEDIUM" if eq == "moderate" else
                    "LOW"    if eq == "weak"     else
                    "NONE"
                )
                ed["p3_source"] = "llm"
               
                if "w_cv" not in ed:
                    ed["w_cv"] = ed.get("synergy_score", 0.0)
            all_scored_edges.update(scored)
            print(f"✓ {len(scored)}/{len(batch)}")
        except Exception as e:
            print(f"✗ Erreur: {str(e)[:40]}")
          
            for edge in batch:
                key = f"{edge.get('source')}|{edge.get('target')}"
                all_scored_edges[key] = {
                    "synergy_score": 0.0,
                    "w_cv": 0.0,
                    "evidence_quality": "absent",
                    "score_breakdown": {
                        "base_synergy": 0.0,
                        "collaboration_bonus": 0.0,
                        "temporal_bonus": 0.0,
                        "evidence_bonus": 0.0,
                        "parent_context_bonus": 0.0,
                        "penalties": 0.0,
                        "final_score": 0.0
                    },
                    "is_ghost_synergy": False,
                    "rationale": "Scoring error - fallback to 0",
                    "p3_confidence": "NONE",
                    "p3_source": "fallback",
                }

        if i + batch_size < total_edges:
            time.sleep(0.5)

    print(f"  ✓ {len(all_scored_edges)} synergies scorées")
    return all_scored_edges


def apply_experience_and_cert_bonus(
    scored_nodes: Dict[str, dict],
    skill_instructions: Dict[str, dict],
    exp_years: Dict[str, int],
) -> Dict[str, dict]:
   
    for skill_id, node_score in scored_nodes.items():
        instruction = skill_instructions.get(skill_id, {})
        base = node_score.get("w_cv", 0.0)
        cert_bonus = 0.0
        exp_bonus  = 0.0

   
        cert_obj = instruction.get("certification", {})
        if isinstance(cert_obj, dict) and cert_obj.get("verifiable", False):
            cert_bonus = 0.25

        skill_label = instruction.get("skill_id", skill_id).lower().replace("_", " ")
        exp_bonus_from_regex = 0.0
        for exp_skill, years in exp_years.items():
            if exp_skill in skill_label or skill_label in exp_skill:
                exp_bonus_from_regex = min(years * 0.05, 0.30)
                break

        if exp_bonus_from_regex > 0:
            exp_bonus = exp_bonus_from_regex
        else:
         
            llm_years = instruction.get("experience_years_detected", 0)
            if isinstance(llm_years, (int, float)) and llm_years > 0:
                exp_bonus = min(llm_years * 0.05, 0.30)

        new_score = round(min(2.0, base + cert_bonus + exp_bonus), 3)
        node_score["w_cv"] = new_score

        sb = node_score.setdefault("score_breakdown", {})
        sb["base_score"]  = base
        sb["final_score"] = new_score
        bonuses = sb.setdefault("bonuses", {})
        bonuses["cert_bonus"]       = cert_bonus
        bonuses["experience_bonus"] = round(exp_bonus, 3)

    return scored_nodes



def run_phase3(
    test_mode: bool = TEST_MODE,
    output_dir: str = RESULTS_DIR,
) -> Dict[str, dict]:
    print("\n" + "█" * 70)
    print("█  PHASE 3 — SCORING CONTEXTUEL")
    print(f"█  Mode : {'🧪 TEST' if test_mode else '🚀 COMPLET'}")
    print("█" * 70)

    if not PHASE2_STORE:
        print("  ❌ PHASE2_STORE vide")
        return {}

    llm = CvLLM()
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, dict] = {}

    for pair_key, phase2_output in PHASE2_STORE.items():
        if not phase2_output:
            continue
        print(f"\n  ── Paire : {pair_key}")
        result = run_phase3_scoring(phase2_output, llm)
        results[pair_key] = result
        PHASE3_STORE[pair_key] = result

    print(f"\n  ✅ Phase 3 : {len(results)} paires scorées")
    return results




AUDIT_SYSTEM_PROMPT = """
You are a Senior Ethical AI Auditor specializing in algorithmic fairness in recruitment.

STEP 1 — IDENTIFY BIAS CATEGORY:
  A. PRESTIGE BIAS: Weight changed because of university/company NAME (not skills demonstrated)
  B. GENDER/DIVERSITY BIAS: Penalizing soft skills, collaboration, caregiving gaps
  C. AGEISM: Using graduation year, career length without context
  D. PROXY BIAS: Socioeconomic signals (location, language of CV)
  E. HALO BIAS: One strong signal used to judge entire profile

STEP 2 — SEVERITY: LOW | MEDIUM | HIGH | N/A

STEP 3 — CORRECTION FACTOR:
  - Unfairly penalized → correction_factor > 1.0 (max 1.50)
  - Unfairly favored   → correction_factor < 1.0 (min 0.60)
  - No bias            → correction_factor = 1.0

STEP 4 — JUSTIFICATION: 2-4 sentences covering evidence, category, factor, fair alternative.

OUTPUT: strict JSON only.
{
  "step1_bias_category": "PRESTIGE|GENDER_DIVERSITY|AGEISM|PROXY|HALO|NONE",
  "step2_severity": "LOW|MEDIUM|HIGH|N/A",
  "step3_correction_direction": "INCREASE|DECREASE|NONE",
  "is_biased": true,
  "correction_factor": 1.0,
  "reason": "<justification>"
}
"""

def _audit_node_bias(
    node_id: str,
    node_data: dict,
    w0: float,
    w_cv: float,
    context: str,
    llm,
) -> dict | None:
    """Appelle le LLM (NVIDIA ou Ollama) pour auditer un nœud."""
    delta = round(w_cv - w0, 3)
    direction_label = "INCREASED" if delta > 0 else "DECREASED"

    prompt = f"""NODE UNDER AUDIT:
  ID    : {node_id}
  Label : {node_data.get('label', node_id)}
  Type  : {node_data.get('type', 'N/A')}

WEIGHT CHANGE:
  w_0 (ideal)   : {w0}
  w_cv (adapted): {w_cv}
  Delta         : {delta:+.3f} ({direction_label})

RATIONALE PROVIDED:
{node_data.get('rationale', 'No rationale.')}

CONTEXT (JD + CV first 600 chars each):
{context}

Determine if this weight change is merit-based or biased. Follow all 4 steps."""

    messages = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    try:
        resp = llm.chat_completion(messages, temperature=0.0, max_tokens=512)
        raw  = resp["choices"][0]["message"]["content"]
        return json.loads(_clean_json_response(raw))
    except Exception as e:
        print(f"    [Audit] Erreur nœud '{node_id}': {e}")
        return None


def audit_edge_bias(
    scored_edges: Dict[str, dict],
    ideal_edges: list,
    context: str,
    llm,
) -> Dict[str, dict]:
    """Audite les arêtes dont le poids a dévié de plus de 0.30,
    et seulement celles avec une trace réelle (w_cv >= 0.2) — pas les synergies absentes."""
    audited = {}
    ideal_map = {f"{e['source']}->{e['target']}": e for e in ideal_edges}

    for edge_key, edge_score in scored_edges.items():
        ideal_edge = ideal_map.get(edge_key)
        if not ideal_edge:
            continue
        w_ideal = ideal_edge.get("weight", 0.5)
        w_cv    = edge_score.get("w_cv", edge_score.get("synergy_score", 0.0))
        delta   = abs(w_cv - w_ideal)

     
        if w_cv < 0.2 or delta < 0.30:
            continue

        prompt = f"""EDGE AUDIT:
  Relation : {edge_key}
  Type     : {ideal_edge.get('relation', 'related')}
  w_ideal  : {w_ideal}
  w_cv     : {w_cv}
  Rationale: {edge_score.get('rationale', 'None')}
  Context  : {context[:1500]}

Is this edge weight change merit-based or biased?"""

        messages = [
            {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        try:
            resp  = llm.chat_completion(messages, temperature=0.0, max_tokens=512)
            raw   = resp["choices"][0]["message"]["content"]
            audit = json.loads(_clean_json_response(raw))
            audited[edge_key] = audit

            cf = float(audit.get("correction_factor", 1.0))
            cf = max(0.60, min(1.50, cf))
            edge_score["p35_audited"]          = True
            edge_score["p35_bias_category"]    = audit.get("step1_bias_category", "NONE")
            edge_score["p35_severity"]         = audit.get("step2_severity", "N/A")
            edge_score["p35_correction_factor"] = cf
            edge_score["p35_reason"]            = audit.get("reason", "")
            if audit.get("is_biased"):
                old_w = edge_score.get("w_cv", w_cv)
                new_w = round(min(1.0, old_w * cf), 3)
                edge_score["w_cv"]            = new_w
                edge_score["w_cv_corrected"]  = new_w
                edge_score["p35_is_biased"]   = True
                print(f"    ⚠ Arête biaisée : {edge_key} — {audit.get('step1_bias_category')}")
            else:
                edge_score["p35_is_biased"]   = False
        except Exception as e:
            print(f"    [EdgeAudit] Erreur {edge_key}: {e}")
            edge_score["p35_audited"]   = True
            edge_score["p35_is_biased"] = False
            edge_score["p35_reason"]    = "Audit failed (LLM error)"

    return audited


def run_phase3_5_bias_audit(
    phase3_output: dict,
    graph_data: dict,
    job_desc: str,
    cv_text: str,
    label: str = "",
    llm=None,
) -> dict:
    """Audite les biais sur un résultat Phase 3 (utilise NVIDIA ou Ollama)."""
    if llm is None:
        llm = make_llm_client()

    print(f"\n  [Phase 3.5] Audit biais — {label}")

    ideal_nodes  = {n["id"]: n for n in graph_data.get("nodes", [])}
    scored_nodes = phase3_output.get("scored_nodes", {})

    context = (
        f"JD (first 2000):\n{job_desc[:2000]}\n\n"
        f"CV (first 2000):\n{cv_text[:2000]}"
    )

    nodes_to_audit = [
        n_id for n_id, data in scored_nodes.items()
        if n_id in ideal_nodes
        and data.get("w_cv", 0) >= 0.2
        and abs(
            data.get("w_cv", 0) / 2.0 -
            ideal_nodes[n_id].get("w_0", 0)
        ) > BIAS_AUDIT_DELTA_THRESHOLD
    ]

    print(f"    Nœuds à auditer : {len(nodes_to_audit)} (seuil delta={BIAS_AUDIT_DELTA_THRESHOLD})")

    biased = 0
    corrected = 0

    for n_id in nodes_to_audit:
        node_data = scored_nodes[n_id]
        w0   = ideal_nodes[n_id].get("w_0", 0)
        w_cv = node_data.get("w_cv", 0)

        audit = _audit_node_bias(n_id, node_data, w0, w_cv, context, llm)
        if audit is None:
           
            scored_nodes[n_id]["p35_audited"]   = True
            scored_nodes[n_id]["p35_is_biased"] = False
            scored_nodes[n_id]["p35_reason"]    = "Audit failed (LLM error)"
            continue

        cf = float(audit.get("correction_factor", 1.0))
        cf = max(0.60, min(1.50, cf))

        scored_nodes[n_id]["p35_audited"]          = True
        scored_nodes[n_id]["p35_bias_category"]    = audit.get("step1_bias_category", "NONE")
        scored_nodes[n_id]["p35_severity"]         = audit.get("step2_severity", "N/A")
        scored_nodes[n_id]["p35_correction_direction"] = audit.get("step3_correction_direction", "NONE")

        if audit.get("is_biased"):
            old_w = node_data["w_cv"]
            new_w = round(min(2.0, old_w * cf), 3)
            scored_nodes[n_id]["w_cv"]                  = new_w
            scored_nodes[n_id]["w_cv_corrected"]        = new_w
            scored_nodes[n_id]["p35_is_biased"]         = True
            scored_nodes[n_id]["p35_correction_factor"] = cf
            scored_nodes[n_id]["p35_reason"]            = audit.get("reason", "")
            biased += 1
            if cf != 1.0:
                corrected += 1
            print(f"    ⚠ {n_id} — BIAIS {audit.get('step1_bias_category')} "
                  f"({audit.get('step2_severity')}) : w_cv {old_w}→{new_w}")
        else:
            scored_nodes[n_id]["p35_is_biased"]         = False
            scored_nodes[n_id]["p35_correction_factor"] = 1.0

 
    scored_edges = phase3_output.get("scored_edges", {})
    ideal_edges  = graph_data.get("edges", [])
    edge_audits  = audit_edge_bias(scored_edges, ideal_edges, context, llm)
    phase3_output["edge_bias_audits"] = edge_audits

    phase3_output["bias_summary"] = {
        "nodes_audited":   len(nodes_to_audit),
        "nodes_with_bias": biased,
        "nodes_corrected": corrected,
        "edges_audited":   len(edge_audits),
    }

    print(f"    Bilan : {biased} biais détectés, {corrected} corrigés")
    return phase3_output


def run_phase3_5(
    csv_path: str  = CSV_INPUT,
    jd_column: str = JD_COLUMN_SCORING,
    id_column: str = ID_COLUMN,
    cv_prefix: str = CV_PREFIX,
    test_mode: bool = TEST_MODE,
    max_jobs: int   = TEST_MAX_JOBS,
    max_cvs: int    = TEST_MAX_CVS,
) -> Dict[str, dict]:
    """Point d'entrée Phase 3.5 — Audit éthique des scores Phase 3.

    Lit  : PHASE3_STORE, GRAPHS_STORE
    Écrit: BIAS_STORE (+ fichiers JSON dans BIAS_DIR)
    """
    print("\n" + "█" * 70)
    print("█  PHASE 3.5 — AUDIT DE BIAIS ÉTHIQUE")
    print(f"█  Mode : {'🧪 TEST' if test_mode else '🚀 COMPLET'}")
    print("█" * 70)

    if not PHASE3_STORE:
        print("  ❌ PHASE3_STORE vide — lancez d'abord run_phase3()")
        return {}

    llm = make_llm_client()

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"  ❌ Erreur CSV : {e}")
        return {}

    if test_mode:
        df = df.head(max_jobs)

    results: Dict[str, dict] = {}
    os.makedirs(BIAS_DIR, exist_ok=True)

    for pair_key, phase3_output in PHASE3_STORE.items():
        job_id = phase3_output.get("job_id", "")
        cv_id  = phase3_output.get("cv_id", "")

        graph_data = GRAPHS_STORE.get(job_id, {})

        try:
            row_idx = int(job_id.split("_")[-1])
            if row_idx >= len(df):
                print(f"  ⏩ {pair_key} — index {row_idx} hors bornes")
                continue
            row = df.iloc[row_idx]
        except (ValueError, IndexError):
            print(f"  ⚠ {pair_key} — impossible d'extraire l'index depuis '{job_id}'")
            continue

        jd_text = str(row.get(jd_column, ""))
        cv_text = str(row.get(cv_id, ""))

        if not jd_text or jd_text == "nan":
            print(f"  ⚠ {pair_key} — JD vide (col='{jd_column}')")
            continue

        if not cv_text or cv_text == "nan" or len(cv_text.strip()) < 10:
            print(f"  ⚠ {pair_key} — CV vide ou invalide (col='{cv_id}')")
            continue

        audited = run_phase3_5_bias_audit(
            phase3_output, graph_data, jd_text, cv_text,
            label=pair_key, llm=llm
        )
        results[pair_key] = audited
        BIAS_STORE[pair_key] = audited

        out_path = os.path.join(BIAS_DIR, f"{pair_key}_audited.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(audited, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Phase 3.5 : {len(results)} paires auditées")
    return results



def _w0(node: dict) -> float:
    """Importance JD du nœud/arête."""
    return float(node.get("w_0", node.get("weight", 0.5)))
 
 
def _wcv_raw(node_id: str, cv_scores: Dict[str, dict]) -> float:
    """Score brut CV pour un nœud."""
    return float(cv_scores.get(node_id, {}).get("w_cv", 0.0))
 
 
def _wcv_norm(node_id: str, cv_scores: Dict[str, dict]) -> float:
    """Score CV normalisé [0,1]."""
    return _wcv_raw(node_id, cv_scores) / _MAX_CV_SCORE
 
 
def _edge_wcv_norm(edge: dict, cv_scores: Dict[str, dict]) -> float:
    """Score CV normalisé d'une arête (cherche dans cv_scores sinon dans edge)."""
    eid = edge.get("id", f"{edge.get('source','')}|{edge.get('target','')}")
    raw = cv_scores.get(eid, {}).get("w_cv", edge.get("w_cv", 0.0))
    return float(raw) / _MAX_CV_SCORE
 
 

 
def _cosine_vec(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))
 
 
def _jaccard_sets(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)
 
 
def _dcg_at_k(relevances: List[float], k: int) -> float:
    relevances = relevances[:k]
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))
 
 
def _ndcg_at_k(relevances: List[float], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    dcg   = _dcg_at_k(relevances, k)
    idcg  = _dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0
 
 
def _average_precision(relevances: List[float], threshold: float = 0.5) -> float:
    binary = [1 if r >= threshold else 0 for r in relevances]
    precisions, n_relevant = [], 0
    for i, b in enumerate(binary):
        if b:
            n_relevant += 1
            precisions.append(n_relevant / (i + 1))
    return float(np.mean(precisions)) if precisions else 0.0
 
 
def _reciprocal_rank(relevances: List[float], threshold: float = 0.5) -> float:
    for i, r in enumerate(relevances):
        if r >= threshold:
            return 1.0 / (i + 1)
    return 0.0
 
 
 
def compute_ntcw(ideal_nodes: List[dict], cv_scores: Dict[str, dict]) -> float:
    """
    [PF-7] NTCW = Σ(w_cv_norm_i × w_0_i) / Σ(w_0_i²)
 
    Interprétation :
      NTCW > 1.05  → candidat sur-qualifié
      NTCW ≈ 1.0   → alignement parfait
      NTCW < 0.50  → candidat sous-qualifié
    """
    numerator   = 0.0
    denominator = 0.0
    for node in ideal_nodes:
        w0   = _w0(node)
        wcvn = _wcv_norm(node["id"], cv_scores)
        numerator   += wcvn * w0
        denominator += w0 * w0
    return round(numerator / denominator, 4) if denominator else 0.0
 

 
def compute_ctm(ideal_nodes: List[dict], cv_scores: Dict[str, dict]) -> float:
    """
    [PF-7] CTM = sqrt( Σ(w_cv_norm_i - w_0_i)² ) / sqrt(n)
 
    Interprétation :
      CTM ≤ 0.25  → profil très proche de l'idéal
      CTM ≤ 0.50  → écart modéré
      CTM > 0.50  → profil significativement différent
    """
    n      = len(ideal_nodes)
    sum_sq = 0.0
    for node in ideal_nodes:
        diff    = _wcv_norm(node["id"], cv_scores) - _w0(node)
        sum_sq += diff * diff
    return round(math.sqrt(sum_sq) / math.sqrt(n), 4) if n else 0.0
 

 
def compute_weighted_gap(ideal_nodes: List[dict], cv_scores: Dict[str, dict]) -> dict:
    """
    [PF-6] Gap pondéré corrigé.
    avg_weighted_gap = Σ(gap_i × w_0_i) / Σ(w_0_i)
    """
    node_gaps:    List[dict] = []
    critical_gaps: List[dict] = []
    weighted_sum = 0.0
    weight_total = 0.0
 
    for node in ideal_nodes:
        node_id = node["id"]
        w0   = _w0(node)
        wcvn = _wcv_norm(node_id, cv_scores)
        gap  = max(0.0, w0 - wcvn)
 
        severity = (
            "critical" if gap >= 0.6 else
            "high"     if gap >= 0.4 else
            "medium"   if gap >= 0.2 else
            "low"
        )
        match_pct = round(max(0.0, (1.0 - gap) * 100), 1)
 
        weighted_sum += gap * w0
        weight_total += w0
 
        entry = {
            "node_id":         node_id,
            "label":           node.get("label", node_id),
            "type":            node.get("type", node.get("p3_node_type", "")),
            "w_0":             round(w0, 3),
            "w_cv_normalized": round(wcvn, 3),
            "gap":             round(gap, 3),
            "severity":        severity,
            "match_pct":       match_pct,
        }
        node_gaps.append(entry)
        if severity == "critical":
            critical_gaps.append(entry)
 
    avg_weighted_gap = round(weighted_sum / weight_total, 4) if weight_total else 0.0
    overall_match    = round(max(0.0, (1.0 - avg_weighted_gap) * 100), 2)
    matched_nodes    = sum(1 for g in node_gaps if g["match_pct"] >= 70)
 
    return {
        "overall_match_pct": overall_match,
        "avg_weighted_gap":  avg_weighted_gap,
        "matched_nodes":     matched_nodes,
        "total_nodes":       len(node_gaps),
        "critical_gaps":     sorted(critical_gaps, key=lambda x: x["w_0"], reverse=True),
        "all_node_gaps":     sorted(node_gaps,     key=lambda x: x["gap"],  reverse=True),
    }
 
 

 
def compute_cluster_scores(
    ideal_nodes: List[dict],
    cv_scores: Dict[str, dict],
) -> Dict[str, dict]:
    """[PF-9] Décompose le match par type de nœud."""
    clusters: Dict[str, list] = defaultdict(list)
    for node in ideal_nodes:
        ntype = node.get("type", node.get("p3_node_type", "hard_skill"))
        clusters[ntype].append((_w0(node), _wcv_norm(node["id"], cv_scores)))
 
    result = {}
    for ntype, pairs in clusters.items():
        w0_sum  = sum(p[0] for p in pairs)
        wcv_sum = sum(p[0] * p[1] for p in pairs)
        avg_match = round((wcv_sum / w0_sum) * 100, 1) if w0_sum else 0.0
        result[ntype] = {
            "count":      len(pairs),
            "match_pct":  avg_match,
            "weight_sum": round(w0_sum, 3),
        }
    return result
 
 

 
def compute_edge_metrics(edges: List[dict], cv_scores: Dict[str, dict]) -> dict:
    """Agrège les scores d'arêtes (synergies)."""
    edge_scores: List[dict] = []
 
    for edge in edges:
        source  = edge.get("source", "")
        target  = edge.get("target", "")
        w0      = _w0(edge)
        wcvn    = _edge_wcv_norm(edge, cv_scores)
        wcv_raw = wcvn * _MAX_CV_SCORE
 
        edge_scores.append({
            "edge_key":      f"{source}|{target}",
            "source":        source,
            "target":        target,
            "w_0":           round(w0, 3),
            "w_cv":          round(wcv_raw, 3),
            "w_cv_norm":     round(wcvn, 3),
            "match_pct":     round(min(wcvn / w0, 1.0) * 100, 1) if w0 else 0.0,
            "p3_confidence": edge.get("p3_confidence", "NONE"),
            "p35_is_biased": edge.get("p35_is_biased", False),
        })
 
    if not edge_scores:
        return {"avg_edge_match_pct": 0.0, "top_synergies": [], "total_edges": 0}
 
    w0_total  = sum(e["w_0"] for e in edge_scores)
    wcv_total = sum(e["w_cv_norm"] * e["w_0"] for e in edge_scores)
    avg_match = round((wcv_total / w0_total) * 100, 2) if w0_total else 0.0
 
    return {
        "avg_edge_match_pct": avg_match,
        "total_edges":        len(edge_scores),
        "top_synergies":      sorted(edge_scores, key=lambda x: -x["w_cv"])[:5],
    }
 
 

 
def compute_verdict(ntcw: float, ctm: float, match_pct: float) -> str:
    """[PF-8] Verdict basé sur combinaison NTCW + CTM + match_pct."""
    if ntcw >= 0.85 and ctm <= 0.30:
        return "Strong Match"
    if ntcw >= 0.85 and ctm > 0.45:
        return "Over-qualified (Structural Mismatch)"
    if ntcw >= 0.70 and ctm <= 0.45:
        return "Good Fit"
    if ntcw >= 0.55 and ctm <= 0.55:
        return "Moderate Match"
    if match_pct >= 50:
        return "Weak Match"
    return "Poor Match"
 

 
def compute_confidence(
    ideal_nodes: List[dict],
    cv_scores: Dict[str, dict],
    gap_data: dict,
) -> dict:
    """[PF-10] Score de confiance basé sur biais, gaps critiques, couverture."""
    n = len(ideal_nodes)
    if n == 0:
        return {"confidence_score": 0.5, "confidence_label": "Low", "factors": {}}
 
    nodes_audited = sum(1 for nd in ideal_nodes if nd.get("p35_audited", False))
    nodes_biased  = sum(1 for nd in ideal_nodes if nd.get("p35_is_biased", False))
    bias_rate     = nodes_biased / max(nodes_audited, 1)
    bias_factor   = 1.0 - (bias_rate * 0.40)
 
    critical_count = len(gap_data.get("critical_gaps", []))
    critical_rate  = critical_count / max(n, 1)
    gap_factor     = 1.0 - (critical_rate * 0.25)
 
    confident_nodes = sum(
        1 for nd in ideal_nodes
        if nd.get("p3_confidence", "NONE") not in ("NONE", "")
    )

    confident_from_cv = sum(
        1 for nid, sc in cv_scores.items()
        if sc.get("p3_confidence", "NONE") not in ("NONE", "")
    )
    coverage_factor = max(confident_nodes, confident_from_cv) / n
 
    confidence = round(
        max(0.40, min(1.0, bias_factor * gap_factor * coverage_factor)),
        2,
    )
    return {
        "confidence_score": confidence,
        "confidence_label": (
            "High"   if confidence >= 0.80 else
            "Medium" if confidence >= 0.60 else
            "Low"
        ),
        "factors": {
            "bias_rate":     round(bias_rate, 3),
            "critical_rate": round(critical_rate, 3),
            "coverage_rate": round(coverage_factor, 3),
            "nodes_audited": nodes_audited,
            "nodes_biased":  nodes_biased,
        },
    }
 
 

 
def compute_profile_summary(
    ideal_nodes: List[dict],
    cv_scores: Dict[str, dict],
    gap_data: dict,
) -> dict:
    """Extrait forces (w_cv_norm ≥ 0.75, w_0 ≥ 0.6) et faiblesses critiques."""
    strengths = []
    for node in ideal_nodes:
        nid  = node["id"]
        wcvn = _wcv_norm(nid, cv_scores)
        w0   = _w0(node)
        if wcvn >= 0.75 and w0 >= 0.6:
            strengths.append({
                "id":            nid,
                "label":         node.get("label", nid),
                "type":          node.get("type", node.get("p3_node_type", "")),
                "w_cv":          round(_wcv_raw(nid, cv_scores), 3),
                "p3_confidence": cv_scores.get(nid, {}).get("p3_confidence",
                                 node.get("p3_confidence", "NONE")),
            })
 
    weaknesses = [
        {
            "id":       g["node_id"],
            "label":    g["label"],
            "type":     g["type"],
            "gap":      g["gap"],
            "severity": g["severity"],
            "w_0":      g["w_0"],
        }
        for g in gap_data["critical_gaps"]
        if g["w_0"] >= 0.6
    ]
 
    return {
        "strengths":           sorted(strengths, key=lambda x: -x["w_cv"])[:8],
        "critical_weaknesses": weaknesses[:8],
    }
 
 
 
def extract_bias_summary(
    ideal_nodes: List[dict],
    edges: List[dict],
    cv_scores: Dict[str, dict],
    scored_edges: Optional[Dict[str, dict]] = None,
) -> dict:
    """Reconstruit le résumé biais depuis les attributs p35_* écrits par Phase 3.5.
    Les flags sont posés sur scored_nodes (cv_scores) et scored_edges, pas sur les
    objets ideal_* du graphe. On les y cherche en priorité, en gardant un fallback
    sur ideal_* pour rétro-compatibilité.
    """
    biased_nodes = []
    biased_edges = []
    scored_edges = scored_edges or {}

   
    for node in ideal_nodes:
        nid = node["id"]
        cv  = cv_scores.get(nid, {})
      
        is_biased = cv.get("p35_is_biased", node.get("p35_is_biased", False))
        if is_biased:
            biased_nodes.append({
                "id":                nid,
                "label":             node.get("label", nid),
                "bias_category":     cv.get("p35_bias_category",
                                            node.get("p35_bias_category", "")),
                "severity":          cv.get("p35_severity",
                                            node.get("p35_severity", "")),
                "correction_factor": cv.get("p35_correction_factor",
                                            node.get("p35_correction_factor", 1.0)),
                "w_cv_before":       cv.get("w_cv_before", 0.0),
                "w_cv_corrected":    cv.get("w_cv_corrected", cv.get("w_cv", 0.0)),
                "reason":            cv.get("p35_reason", node.get("p35_reason", "")),
            })

    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        ekey   = f"{source}->{target}"
        ekey_alt = f"{source}|{target}"
        sc = scored_edges.get(ekey, scored_edges.get(ekey_alt, {}))
        is_biased = sc.get("p35_is_biased", edge.get("p35_is_biased", False))
        if is_biased:
            biased_edges.append({
                "edge":              f"{source}|{target}",
                "bias_category":     sc.get("p35_bias_category",
                                            edge.get("p35_bias_category", "")),
                "severity":          sc.get("p35_severity",
                                            edge.get("p35_severity", "")),
                "correction_factor": sc.get("p35_correction_factor",
                                            edge.get("p35_correction_factor", 1.0)),
                "w_cv_before":       sc.get("w_cv_before", 0.0),
                "w_cv_corrected":    sc.get("w_cv_corrected", sc.get("w_cv", 0.0)),
                "reason":            sc.get("p35_reason", edge.get("p35_reason", "")),
            })

    cats = Counter(
        n["bias_category"] for n in biased_nodes + biased_edges
        if n.get("bias_category") not in ("NONE", "")
    )

   
    nodes_audited = sum(
        1 for nd in ideal_nodes
        if cv_scores.get(nd["id"], {}).get("p35_audited", False)
        or nd.get("p35_audited", False)
    )
    edges_audited = sum(
        1 for e in edges
        if scored_edges.get(f"{e.get('source','')}->{e.get('target','')}", {}).get("p35_audited", False)
        or scored_edges.get(f"{e.get('source','')}|{e.get('target','')}", {}).get("p35_audited", False)
        or e.get("p35_audited", False)
    )

    return {
        "nodes_audited":   nodes_audited,
        "nodes_with_bias": len(biased_nodes),
        "edges_audited":   edges_audited,
        "edges_with_bias": len(biased_edges),
        "bias_categories": dict(cats.most_common()),
        "biased_nodes":    biased_nodes,
        "biased_edges":    biased_edges,
    }
 

 
def compute_final_candidate_score(
    ideal_nodes: List[dict],
    edges: List[dict],
    cv_scores: Dict[str, dict],
) -> dict:
    """Score final pondéré. Pondération : 60% nodes, 40% edges."""
    node_scores, node_weights = [], []
    for node in ideal_nodes:
        w0 = _w0(node)
        node_scores.append(_wcv_raw(node["id"], cv_scores) * w0)
        node_weights.append(w0)
 
    edge_scores, edge_weights = [], []
    for edge in edges:
        w0 = _w0(edge)
        wcv = _edge_wcv_norm(edge, cv_scores) * _MAX_CV_SCORE
        edge_scores.append(wcv * w0)
        edge_weights.append(w0)
 
    node_score = sum(node_scores) / sum(node_weights) if node_weights else 0.0
    edge_score = sum(edge_scores) / sum(edge_weights) if edge_weights else 0.0
    combined   = 0.6 * node_score + 0.4 * edge_score
 
    return {
        "node_score": round(node_score, 4),
        "edge_score": round(edge_score, 4),
        "combined":   round(combined,   4),
    }
 

 
def compute_advanced_metrics_v6(
    ideal_nodes: List[dict],
    edges: List[dict],
    cv_scores: Dict[str, dict],
) -> dict:
    """
    Calcule toutes les métriques V6.
 
    Métriques calculées :
      node_matching_cosine    — matching nœuds pondéré par w_0
      edge_matching_score     — matching des relations JD→CV
      gap_score               — 1 - distance L2 normalisée (1=parfait)
      jaccard_nodes           — Jaccard skills couverts vs skills JD
      weighted_precision      — précision pondérée par w_0
      weighted_recall         — rappel pondéré par w_0
      weighted_f1             — F1 harmonique pondéré
      skill_depth_score       — profondeur sur skills couverts
      missing_critical_penalty— pénalité skills critiques absents
      redundancy_penalty      — pénalité surqualification inutile
      ndcg_5 / ndcg_10        — NDCG@k sur le ranking
      map_score               — Mean Average Precision
      mrr_score               — Mean Reciprocal Rank
      skill_balance_score     — équilibre inter-compétences
      top_skill_alignment     — alignement top-3 CV vs JD
      semantic_fit_score      — fit sémantique composite
      experience_density      — densité Σ(w_cv×w_0)/n
    """
    n = len(ideal_nodes)
    if n == 0:
        return _empty_v6_metrics()
 
    w0_arr  = np.array([_w0(nd)                        for nd in ideal_nodes])
    wcv_arr = np.array([_wcv_norm(nd["id"], cv_scores) for nd in ideal_nodes])
 

    node_ids    = [nd["id"] for nd in ideal_nodes]
    covered     = set(nid for nid, cv in zip(node_ids, wcv_arr) if cv >= _COVERAGE_THRESH)
    critical    = set(nid for nid, w0 in zip(node_ids, w0_arr)  if w0 >= _CRITICAL_THRESH)
    required_jd = set(nid for nid, w0 in zip(node_ids, w0_arr)  if w0 >= 0.5)
 
    total_w0 = float(np.sum(w0_arr))
    if total_w0 > 0:
        per_node = np.array([
            max(0.0, 1.0 - abs(cv - w0) / (w0 + 1e-9))
            for cv, w0 in zip(wcv_arr, w0_arr)
        ])
        node_matching_cosine = float(np.dot(w0_arr, per_node) / total_w0)
    else:
        node_matching_cosine = 0.0
 
  
    if edges:
        total_ew, matched_ew = 0.0, 0.0
        for edge in edges:
            w_e   = _w0(edge)
            src   = edge.get("source", "")
            tgt   = edge.get("target", "")
            wcv_u = _wcv_norm(src, cv_scores) if src else 0.0
            wcv_v = _wcv_norm(tgt, cv_scores) if tgt else 0.0
            match = (
                (1.0 if wcv_u >= _COVERAGE_THRESH else wcv_u / _COVERAGE_THRESH) +
                (1.0 if wcv_v >= _COVERAGE_THRESH else wcv_v / _COVERAGE_THRESH)
            ) / 2.0
            total_ew   += w_e
            matched_ew += w_e * match
        edge_matching_score = float(matched_ew / total_ew) if total_ew > 0 else 0.0
    else:
        edge_matching_score = len(covered) / n if n else 0.0
 
   
    norm_ideal = float(np.linalg.norm(w0_arr))
    if norm_ideal > 0:
        gap_score = float(max(0.0, 1.0 - np.linalg.norm(w0_arr - wcv_arr) / norm_ideal))
    else:
        gap_score = 1.0
 
  
    jaccard_nodes = _jaccard_sets(covered, required_jd)
 
    covered_nodes  = [(nd, _wcv_norm(nd["id"], cv_scores))
                      for nd in ideal_nodes if _wcv_norm(nd["id"], cv_scores) >= _COVERAGE_THRESH]
    required_nodes = [(nd, _w0(nd)) for nd in ideal_nodes if _w0(nd) >= 0.5]
 
    if covered_nodes:
        weighted_precision = float(
            sum(_w0(nd)                for nd, _ in covered_nodes) /
            sum(max(wcvn, 1e-9)        for _, wcvn in covered_nodes)
        )
    else:
        weighted_precision = 0.0
 
    if required_nodes:
        total_req = sum(_w0(nd) for nd, _ in required_nodes)
        cov_w     = sum(_wcv_norm(nd["id"], cv_scores) for nd, _ in required_nodes
                        if _wcv_norm(nd["id"], cv_scores) >= _COVERAGE_THRESH)
        weighted_recall = float(cov_w / total_req) if total_req > 0 else 0.0
    else:
        weighted_recall = 0.0
 
    p, r = weighted_precision, weighted_recall
    weighted_f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
 
  
    if covered_nodes:
        tw0_cov   = sum(_w0(nd)  for nd, _ in covered_nodes)
        depth_sum = sum(_wcv_norm(nd["id"], cv_scores) * _w0(nd) for nd, _ in covered_nodes)
        skill_depth_score = float(depth_sum / tw0_cov) if tw0_cov > 0 else 0.0
    else:
        skill_depth_score = 0.0
 

    if critical:
        covered_critical = sum(1 for nid in critical if nid in covered)
        missing_critical_penalty = float(covered_critical / len(critical))
    else:
        missing_critical_penalty = 1.0
 
  
    redundant = sum(
        1 for cv, w0 in zip(wcv_arr, w0_arr)
        if cv > 0.8 and w0 < 0.3
    )
    redundancy_info    = float(redundant / n) if n > 0 else 0.0
    redundancy_penalty = 1.0   
 

    paired_rank = sorted(zip(wcv_arr, w0_arr), key=lambda x: x[0], reverse=True)
    ranking_rel = [float(w0) for _, w0 in paired_rank]
 
    ndcg_5    = round(_ndcg_at_k(ranking_rel, 5),  4)
    ndcg_10   = round(_ndcg_at_k(ranking_rel, 10), 4)
    map_score = round(_average_precision(ranking_rel, threshold=0.5), 4)
    mrr_score = round(_reciprocal_rank(ranking_rel,   threshold=0.5), 4)
 
  
    covered_vals = [_wcv_norm(nd["id"], cv_scores) for nd in ideal_nodes
                    if _wcv_norm(nd["id"], cv_scores) >= _COVERAGE_THRESH]
    if covered_vals and np.mean(covered_vals) > 0:
        cv_coef = float(np.std(covered_vals) / np.mean(covered_vals))
        skill_balance_score = float(1.0 / (1.0 + cv_coef))
    else:
        skill_balance_score = 0.0

    if n >= 3:
        k = min(3, n)
        top_jd  = set(np.argsort(w0_arr)[-k:])
        top_cv  = set(np.argsort(wcv_arr)[-k:])
        overlap = len(top_jd & top_cv) / k
        aligned = top_jd & top_cv
        if aligned:
            depth_bonus = float(np.mean([wcv_arr[i] for i in aligned]))
            top_skill_alignment = float((overlap + depth_bonus) / 2.0)
        else:
            top_skill_alignment = overlap
    else:
        top_skill_alignment = 0.0
 
    cos_sim = _cosine_vec(wcv_arr, w0_arr)
    if n >= 2 and np.std(w0_arr) > 0 and np.std(wcv_arr) > 0:
        cov = float(np.mean((wcv_arr - np.mean(wcv_arr)) * (w0_arr - np.mean(w0_arr))))
        prs = cov / (float(np.std(wcv_arr)) * float(np.std(w0_arr)))
        prs = max(0.0, prs) if not math.isnan(prs) else 0.0
    else:
        prs = 0.0
    k5       = min(5, n)
    top5_jd  = set(np.argsort(w0_arr)[-k5:])
    top5_cv  = set(np.argsort(wcv_arr)[-k5:])
    rov      = len(top5_jd & top5_cv) / k5 if k5 else 0.0
    semantic_fit_score = float((cos_sim + prs + rov) / 3.0)
 

    experience_density = float(np.sum(wcv_arr * w0_arr) / n) if n > 0 else 0.0
 
    return {
        "node_matching_cosine":     round(node_matching_cosine,     4),
        "edge_matching_score":      round(edge_matching_score,      4),
        "gap_score":                round(gap_score,                4),
        "jaccard_nodes":            round(jaccard_nodes,            4),
        "weighted_precision":       round(weighted_precision,       4),
        "weighted_recall":          round(weighted_recall,          4),
        "weighted_f1":              round(weighted_f1,              4),
        "skill_depth_score":        round(skill_depth_score,        4),
        "missing_critical_penalty": round(missing_critical_penalty, 4),
        "redundancy_penalty":       round(redundancy_penalty,       4),
        "redundancy_info":          round(redundancy_info,          4),
        "ndcg_5":                   ndcg_5,
        "ndcg_10":                  ndcg_10,
        "map_score":                map_score,
        "mrr_score":                mrr_score,
        "skill_balance_score":      round(skill_balance_score,      4),
        "top_skill_alignment":      round(top_skill_alignment,      4),
        "semantic_fit_score":       round(semantic_fit_score,       4),
        "experience_density":       round(experience_density,       4),
    }
 
 
def _empty_v6_metrics() -> dict:
    return {k: 0.0 for k in [
        "node_matching_cosine", "edge_matching_score",
        "gap_score", "jaccard_nodes",
        "weighted_precision", "weighted_recall", "weighted_f1",
        "skill_depth_score", "missing_critical_penalty", "redundancy_penalty",
        "redundancy_info",
        "ndcg_5", "ndcg_10", "map_score", "mrr_score",
        "skill_balance_score", "top_skill_alignment",
        "semantic_fit_score", "experience_density",
    ]}
 

 
def compute_combo_v6_final(v6: dict) -> float:
    """Score synthèse V7 — wF1 et gap_score en tête, ranking en support léger."""
    score = (
        0.30 * v6.get("weighted_f1",              0.0) +
        0.20 * v6.get("gap_score",                0.0) +
        0.15 * v6.get("edge_matching_score",       0.0) +
        0.15 * v6.get("node_matching_cosine",      0.0) +
        0.10 * v6.get("missing_critical_penalty",  0.0) +
        0.05 * v6.get("ndcg_5",                   0.0) +
        0.03 * v6.get("semantic_fit_score",        0.0) +
        0.02 * v6.get("skill_balance_score",       0.0)
    )
    return round(score * 100, 2)
 
 
def compute_combo_matching(v6: dict) -> float:
    """Combo matching pur : node_matching + edge_matching + jaccard."""
    return round((
        0.40 * v6.get("node_matching_cosine", 0.0) +
        0.35 * v6.get("edge_matching_score",  0.0) +
        0.25 * v6.get("jaccard_nodes",        0.0)
    ) * 100, 2)
 
 
def compute_combo_fit(v6: dict) -> float:
    """Combo qualité de fit : gap + semantic + jaccard."""
    return round((
        0.40 * v6.get("gap_score",          0.0) +
        0.35 * v6.get("semantic_fit_score", 0.0) +
        0.25 * v6.get("jaccard_nodes",      0.0)
    ) * 100, 2)
 
 
def compute_combo_ranking(v6: dict) -> float:
    """Combo ranking-aware : NDCG + MAP + MRR."""
    return round((
        0.40 * v6.get("ndcg_5",    0.0) +
        0.35 * v6.get("map_score", 0.0) +
        0.25 * v6.get("mrr_score", 0.0)
    ) * 100, 2)
 
 
def compute_combo_prf1(v6: dict) -> float:
    """Combo F1 pondéré + profondeur."""
    return round((
        0.40 * v6.get("weighted_f1",        0.0) +
        0.30 * v6.get("skill_depth_score",  0.0) +
        0.20 * v6.get("weighted_recall",    0.0) +
        0.10 * v6.get("weighted_precision", 0.0)
    ) * 100, 2)
 
 
def compute_combo_safe(v6: dict) -> float:
    """Combo sans risque : pénalités critiques + gap + équilibre des skills."""
    return round((
        0.50 * v6.get("missing_critical_penalty", 0.0) +
        0.30 * v6.get("gap_score",                0.0) +
        0.20 * v6.get("skill_balance_score",      0.0)
    ) * 100, 2)
 
 
def compute_combo_specialist(v6: dict) -> float:
    """Combo spécialiste : alignement top-skills + profondeur + densité."""
    return round((
        0.50 * v6.get("top_skill_alignment", 0.0) +
        0.30 * v6.get("skill_depth_score",   0.0) +
        0.20 * v6.get("experience_density",  0.0)
    ) * 100, 2)
 
 

MIN_POOL_FOR_ZSCORE = 3

def _safe_val(obj) -> float:
    """Extrait la valeur numérique d'un dict {'value': x} ou d'un scalaire."""
    if isinstance(obj, dict):
        return float(obj.get("value", 0.0))
    return float(obj) if obj is not None else 0.0


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    n   = len(values)
    std = float(np.std(values))
    if n < MIN_POOL_FOR_ZSCORE or std < 1e-6:
        ranks = np.argsort(np.argsort(values)).astype(float)
        return (ranks / (n - 1)) * 2 - 1 if n > 1 else np.zeros(n)
    return (values - float(np.mean(values))) / std


def _fallback_adjusted_score(raw_score: float, fallback_ratio: float,
                              confidence_score: float) -> float:
    reliability_malus = min(0.15, fallback_ratio * 0.15)
    return round(raw_score * (1.0 - reliability_malus) * max(0.85, confidence_score), 4)


def compute_pool_ranking(reports: Dict[str, dict]) -> Dict[str, dict]:
    """
    Enrichit chaque rapport avec un champ 'pool_ranking' (rang, percentile, tier).
    À appeler après avoir généré tous les rapports du pool.
    """
    keys = list(reports.keys())
    n    = len(keys)
    if n == 0:
        return reports

    RANKING_METRICS = [
        ("combo_v6",  lambda r: _safe_val(r.get("combo_scores_v6", {}).get("combo_v6_final", 0))),
        ("wf1",       lambda r: _safe_val(r.get("advanced_metrics_v6", {}).get("weighted_f1",  0))),
        ("gap_score", lambda r: _safe_val(r.get("advanced_metrics_v6", {}).get("gap_score",    0))),
        ("ntcw",      lambda r: _safe_val(r.get("advanced_metrics", {}).get("NTCW", 0))),
        ("match_pct", lambda r: float(r.get("overall_assessment", {}).get("match_pct", 0))),
        ("ndcg_5",    lambda r: _safe_val(r.get("advanced_metrics_v6", {}).get("ndcg_5",      0))),
    ]
    WEIGHTS = {"combo_v6": 0.35, "wf1": 0.25, "gap_score": 0.20,
               "ntcw": 0.10, "match_pct": 0.05, "ndcg_5": 0.05}

    metric_values: Dict[str, List[float]] = {m: [] for m, _ in RANKING_METRICS}
    for key in keys:
        r = reports[key]
        for metric_name, extractor in RANKING_METRICS:
            metric_values[metric_name].append(extractor(r))

    metric_zscores: Dict[str, np.ndarray] = {}
    metric_stats:   Dict[str, dict]       = {}
    for metric_name, vals in metric_values.items():
        arr = np.array(vals, dtype=float)
        metric_stats[metric_name] = {"mean": round(float(np.mean(arr)), 4),
                                     "std":  round(float(np.std(arr)),  4)}
        metric_zscores[metric_name] = _robust_zscore(arr)

    composite_scores: Dict[str, float] = {}
    for i, key in enumerate(keys):
        r = reports[key]
        z_sum = sum(WEIGHTS.get(mn, 0.0) * metric_zscores[mn][i] for mn, _ in RANKING_METRICS)
        fr = float(r.get("overall_assessment", {}).get("fallback_ratio", 0.0))
        cs = float(r.get("overall_assessment", {}).get("confidence", {}).get("confidence_score", 0.7))
        composite_scores[key] = _fallback_adjusted_score(z_sum, fr, cs)

    sorted_keys = sorted(composite_scores, key=composite_scores.__getitem__, reverse=True)
    rank_map    = {key: rank + 1 for rank, key in enumerate(sorted_keys)}

    def _adaptive_tier(rank: int, total: int) -> str:
        if total <= 3:
            return ["Meilleur candidat", "2ème candidat", "3ème candidat"][rank - 1]
        pct = (total - rank) / max(total - 1, 1)
        if pct >= 0.90: return "Top 10%"
        if pct >= 0.75: return "Top 25%"
        if pct >= 0.50: return "Top 50%"
        return "Bottom 50%"

    for i, key in enumerate(keys):
        r    = reports[key]
        rank = rank_map[key]
        percentile = round((n - rank) / max(n - 1, 1), 4) if n > 1 else 1.0
        r["pool_ranking"] = {
            "rank":              rank,
            "pool_size":         n,
            "percentile":        percentile,
            "composite_z_score": round(composite_scores[key], 4),
            "tier":              _adaptive_tier(rank, n),
            "z_scores":          {mn: round(float(metric_zscores[mn][i]), 3) for mn, _ in RANKING_METRICS},
            "pool_stats":        {mn: {"mean": s["mean"], "std": s["std"]} for mn, s in metric_stats.items()},
            "ranking_reliability": (
                "⚠ Score ajusté (fallback élevé)"
                if float(r.get("overall_assessment", {}).get("fallback_ratio", 0)) > 0.3
                else "✅ Score fiable"
            ),
        }
    return reports



 
def phase4_generate_report(
    job_id:     str,
    cv_id:      str,
    graph_data: dict,
    phase3_data: dict,
) -> dict:
    """
    Phase 4 V6 : génère le rapport final complet.
 
    Args:
        job_id      : identifiant du poste
        cv_id       : identifiant du candidat
        graph_data  : dict issu de GRAPHS_STORE[job_id]
                      Attendu : {"nodes": [...], "edges": [...]}
        phase3_data : dict issu de PHASE3_STORE[pair_key]
                      Attendu : {"scored_nodes": {node_id: {"w_cv": float, ...}}}
 
    Returns:
        dict rapport JSON-serialisable
    """
    print(f"  [Phase 4] Rapport — {job_id}×{cv_id}")
 
    ideal_nodes: List[dict] = graph_data.get("nodes", [])
    edges:       List[dict] = graph_data.get("edges", [])
    cv_scores:   Dict[str, dict] = phase3_data.get("scored_nodes", {})

    ntcw         = compute_ntcw(ideal_nodes, cv_scores)
    ctm          = compute_ctm(ideal_nodes, cv_scores)
    gap_data     = compute_weighted_gap(ideal_nodes, cv_scores)
    clusters     = compute_cluster_scores(ideal_nodes, cv_scores)
    edge_metrics = compute_edge_metrics(edges, cv_scores)
    bias_summary = extract_bias_summary(ideal_nodes, edges, cv_scores, phase3_data.get("scored_edges", {}))
    profile      = compute_profile_summary(ideal_nodes, cv_scores, gap_data)
    final_scores = compute_final_candidate_score(ideal_nodes, edges, cv_scores)
    confidence   = compute_confidence(ideal_nodes, cv_scores, gap_data)
 
    match_pct = gap_data["overall_match_pct"]
    verdict   = compute_verdict(ntcw, ctm, match_pct)
 

    fallback_ratio = float(phase3_data.get("metadata", {}).get("fallback_ratio", 0.0))

    v6_metrics = compute_advanced_metrics_v6(ideal_nodes, edges, cv_scores)
 
    combo_v6_final   = compute_combo_v6_final(v6_metrics)
    combo_matching   = compute_combo_matching(v6_metrics)
    combo_fit        = compute_combo_fit(v6_metrics)
    combo_ranking    = compute_combo_ranking(v6_metrics)
    combo_prf1       = compute_combo_prf1(v6_metrics)
    combo_safe       = compute_combo_safe(v6_metrics)
    combo_specialist = compute_combo_specialist(v6_metrics)

    print(f"    Verdict : {verdict} | Match : {match_pct}% | NTCW : {ntcw} | CTM : {ctm}")
    print(f"    combo_v6_final : {combo_v6_final} | wF1 : {v6_metrics['weighted_f1']} | gap : {v6_metrics['gap_score']}")
    print(f"    Confiance : {confidence['confidence_label']} ({confidence['confidence_score']})")
    if fallback_ratio > 0.3:
        print(f"    ⚠ fallback_ratio élevé : {fallback_ratio:.1%} — scores moins fiables")
 
    return {
        "job_id":       job_id,
        "cv_id":        cv_id,
        "generated_at": datetime.now().isoformat(),
 
      
        "overall_assessment": {
            "verdict":        verdict,
            "match_pct":      match_pct,
            "confidence":     confidence,
            "fallback_ratio": fallback_ratio,
        },

        "final_scores": final_scores,
 
        "advanced_metrics": {
            "NTCW": {
                "value": ntcw,
                "description": "Normalized Total Contextual Weight — alignement global",
                "interpretation": (
                    "Sur-qualifié"   if ntcw > 1.05 else
                    "Aligné"         if ntcw >= 0.80 else
                    "Sous-qualifié"
                ),
            },
            "CTM": {
                "value": ctm,
                "description": "Contextual Transformation Magnitude — distance structurelle",
                "interpretation": (
                    "Profil proche"         if ctm <= 0.25 else
                    "Écart modéré"          if ctm <= 0.50 else
                    "Profil très différent"
                ),
            },
            "weighted_gap":   gap_data["avg_weighted_gap"],
            "matched_nodes":  f"{gap_data['matched_nodes']}/{gap_data['total_nodes']}",
            "edge_match_pct": edge_metrics["avg_edge_match_pct"],
        },
 
    
        "advanced_metrics_v6": {
            "node_matching_cosine": {
                "value":         v6_metrics["node_matching_cosine"],
                "description":   "Matching nœuds pondéré par importance JD",
                "interpretation": (
                    "Excellent" if v6_metrics["node_matching_cosine"] >= 0.80 else
                    "Bon"       if v6_metrics["node_matching_cosine"] >= 0.60 else
                    "Faible"
                ),
            },
            "edge_matching_score": {
                "value":         v6_metrics["edge_matching_score"],
                "description":   "Matching des relations/synergies JD→CV",
                "interpretation": (
                    "Excellent" if v6_metrics["edge_matching_score"] >= 0.75 else
                    "Bon"       if v6_metrics["edge_matching_score"] >= 0.50 else
                    "Faible"
                ),
            },
            "gap_score": {
                "value":         v6_metrics["gap_score"],
                "description":   "1 - distance L2 normalisée vs profil idéal (1=parfait)",
                "interpretation": (
                    "Très proche" if v6_metrics["gap_score"] >= 0.80 else
                    "Acceptable"  if v6_metrics["gap_score"] >= 0.60 else
                    "Éloigné"
                ),
            },
            "jaccard_nodes": {
                "value":       v6_metrics["jaccard_nodes"],
                "description": "Jaccard(skills couverts CV, skills requis JD)",
            },
            "weighted_precision": v6_metrics["weighted_precision"],
            "weighted_recall":    v6_metrics["weighted_recall"],
            "weighted_f1": {
                "value":         v6_metrics["weighted_f1"],
                "description":   "F1 harmonique pondéré par importance JD",
                "interpretation": (
                    "Excellent" if v6_metrics["weighted_f1"] >= 0.80 else
                    "Bon"       if v6_metrics["weighted_f1"] >= 0.60 else
                    "Faible"
                ),
            },
            "skill_depth_score":        v6_metrics["skill_depth_score"],
            "missing_critical_penalty": {
                "value":         v6_metrics["missing_critical_penalty"],
                "description":   "Proportion de skills critiques (w_0≥0.7) couverts",
                "interpretation": (
                    "Aucun skill critique manquant"          if v6_metrics["missing_critical_penalty"] >= 0.99 else
                    "Quelques skills critiques manquants"    if v6_metrics["missing_critical_penalty"] >= 0.70 else
                    "Skills critiques manquants — risque élevé"
                ),
            },
            "redundancy_penalty": {
                "value":       v6_metrics["redundancy_penalty"],
                "description": "Pénalité surqualification inutile (1=aucune redondance)",
            },
            "redundancy_info": {
                "value": v6_metrics["redundancy_info"],
                "label": ("Profil sur-qualifié sur quelques skills" if v6_metrics["redundancy_info"] > 0.15
                          else "Profil bien calibré"),
                "note":  "Informatif uniquement — ne pénalise pas le candidat",
            },
            "ndcg_5":    v6_metrics["ndcg_5"],
            "ndcg_10":   v6_metrics["ndcg_10"],
            "map_score": v6_metrics["map_score"],
            "mrr_score": v6_metrics["mrr_score"],
            "skill_balance_score": v6_metrics["skill_balance_score"],
            "top_skill_alignment": v6_metrics["top_skill_alignment"],
            "semantic_fit_score":  v6_metrics["semantic_fit_score"],
            "experience_density":  v6_metrics["experience_density"],
        },
 
        "combo_scores_v6": {
            "combo_v6_final": {
                "value":       combo_v6_final,
                "description": "Score synthèse V6 (toutes familles représentées)",
            },
            "combo_matching": {
                "value":       combo_matching,
                "description": "Matching pur (node + edge + jaccard)",
            },
            "combo_fit": {
                "value":       combo_fit,
                "description": "Qualité de fit (gap + semantic + jaccard)",
            },
            "combo_ranking": {
                "value":       combo_ranking,
                "description": "Ranking-aware (ndcg + map + mrr)",
            },
            "combo_prf1": {
                "value":       combo_prf1,
                "description": "F1 pondéré + profondeur",
            },
            "combo_safe": {
                "value":       combo_safe,
                "description": "Candidat sans risque (pénalités + gap)",
            },
            "combo_specialist": {
                "value":       combo_specialist,
                "description": "Top alignement + profondeur (profil spécialisé)",
            },
        },
 
        "cluster_analysis": clusters,
 

        "edge_analysis": edge_metrics,
 

        "critical_gaps": [
            {
                "skill":      g["label"],
                "type":       g["type"],
                "gap":        g["gap"],
                "severity":   g["severity"],
                "importance": g["w_0"],
            }
            for g in gap_data["critical_gaps"][:10]
        ],
 
        "profile_summary": {
            "strengths": [
                {"label": s["label"], "type": s["type"], "w_cv": s["w_cv"]}
                for s in profile["strengths"]
            ],
            "critical_weaknesses": [
                {"label": w["label"], "type": w["type"], "gap": w["gap"]}
                for w in profile["critical_weaknesses"]
            ],
        },
 
    
        "bias_audit": bias_summary,
 

        "decision_rationale": {
            "why_this_verdict": (
                f"NTCW={ntcw} ({('sur' if ntcw > 1.05 else 'sous') if abs(ntcw - 1.0) > 0.05 else 'bien'}-qualifié), "
                f"CTM={ctm} (écart structurel {'faible' if ctm <= 0.25 else 'modéré' if ctm <= 0.50 else 'élevé'}), "
                f"match_pct={match_pct}% → {verdict}"
            ),
            "v6_summary": (
                f"weighted_f1={v6_metrics['weighted_f1']}, "
                f"gap_score={v6_metrics['gap_score']}, "
                f"ndcg_5={v6_metrics['ndcg_5']}, "
                f"missing_critical_penalty={v6_metrics['missing_critical_penalty']}, "
                f"combo_v6_final={combo_v6_final}"
            ),
            "key_strengths":   f"{len(profile['strengths'])} compétences clés validées",
            "key_gaps":        f"{len(gap_data['critical_gaps'])} gaps critiques identifiés",
            "bias_note":       f"{bias_summary['nodes_with_bias']} scores corrigés pour biais",
            "confidence_note": f"Confiance : {confidence['confidence_label']} ({confidence['confidence_score']})",
            "scoring_reliability": (
                f"⚠ {fallback_ratio:.1%} des scores via fallback mathématique (LLM instable)"
                if fallback_ratio > 0.3 else
                f"✅ {(1-fallback_ratio):.1%} des scores via LLM (fiable)"
            ),
        },
    }
 
 
def generate_executive_summary(reports: Dict[str, dict]) -> str:
    """Résumé Markdown de tous les rapports phase 4, trié par rang pool (si disponible)."""
    lines = [
        "# RÉSUMÉ EXÉCUTIF — ANALYSE DES CANDIDATS (V7 + POOL RANKING)",
        f"*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        f"**Total analysé : {len(reports)} paire(s) job×cv**\n",
        "| Rang | Paire | Tier | Percentile | Verdict | Match% | NTCW | combo_v6 | wF1 | gap | Confiance | Fiabilité |",
        "|------|-------|------|------------|---------|--------|------|----------|-----|-----|-----------|-----------|",
    ]

    def _sort_key(r):
        return r.get("pool_ranking", {}).get("rank", 9999)

    for r in sorted(reports.values(), key=_sort_key):
        pr       = r.get("pool_ranking", {})
        pair     = f"{r['job_id']}×{r['cv_id']}"
        rank     = pr.get("rank", "?")
        tier     = pr.get("tier", "?")
        pct      = f"{round(pr.get('percentile', 0)*100, 1)}%" if pr else "-"
        verdict  = r["overall_assessment"]["verdict"]
        match    = r["overall_assessment"]["match_pct"]
        ntcw_v   = _safe_val(r["advanced_metrics"]["NTCW"])
        combo_v6 = _safe_val(r.get("combo_scores_v6", {}).get("combo_v6_final", 0.0))
        wf1_val  = _safe_val(r.get("advanced_metrics_v6", {}).get("weighted_f1",  0.0))
        gap_val  = _safe_val(r.get("advanced_metrics_v6", {}).get("gap_score",    0.0))
        conf     = r["overall_assessment"]["confidence"]["confidence_label"]
        fiab     = pr.get("ranking_reliability", "—")
        lines.append(
            f"| #{rank} | {pair} | {tier} | {pct} | {verdict} | {match}% "
            f"| {ntcw_v} | {combo_v6} | {wf1_val} | {gap_val} | {conf} | {fiab} |"
        )
    return "\n".join(lines)
 
 

 
def run_phase4(
    test_mode:  bool = TEST_MODE,
    output_dir: str  = REPORTS_DIR,
) -> Dict[str, dict]:
    """
    Phase 4 V6 pour toutes les paires dans PHASE3_STORE.
 
    Utilise les stores globaux :
      PHASE3_STORE  : {pair_key: {"job_id", "cv_id", "scored_nodes": {...}}}
      GRAPHS_STORE  : {job_id:   {"nodes": [...], "edges": [...]}}
 
    Returns:
        {pair_key: report_dict}
    """
    print("\n" + "█" * 70)
    print("█  PHASE 4 V6 — SCORING FINAL & RAPPORT")
    print(f"█  Mode : {'🧪 TEST' if test_mode else '🚀 COMPLET'}")
    print("█" * 70)
 
    if not PHASE3_STORE:
        print("  ❌ PHASE3_STORE vide")
        return {}
 
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, dict] = {}
 
    for pair_key, phase3_data in PHASE3_STORE.items():
        job_id     = phase3_data.get("job_id", "")
        cv_id      = phase3_data.get("cv_id", "")
        graph_data = GRAPHS_STORE.get(job_id, {})
 
        if not graph_data:
            print(f"  ⏩ {pair_key} — graphe manquant dans GRAPHS_STORE")
            continue
 
        report = phase4_generate_report(job_id, cv_id, graph_data, phase3_data)
        results[pair_key] = report
        REPORTS_STORE[pair_key] = report
 
        out_path = os.path.join(output_dir, f"report_{pair_key}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"    → {out_path}")
 
    print(f"\n  ✅ Phase 4 V6 : {len(results)} rapport(s) générés")
 
    if results:
        
        results = compute_pool_ranking(results)
        for pair_key, report in results.items():
            REPORTS_STORE[pair_key] = report
            out_path_r = os.path.join(output_dir, f"report_{pair_key}.json")
            with open(out_path_r, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        print("\n" + generate_executive_summary(results))

    return results
 



def run_full_pipeline(
    csv_path:    str  = CSV_INPUT,
    test_mode:   bool = TEST_MODE,
    max_jobs:    int  = TEST_MAX_JOBS,
    max_cvs:     int  = TEST_MAX_CVS,
    run_graph:   bool = True,
    run_p2:      bool = True,
    run_p3:      bool = True,
    run_p4:      bool = True,
):
    """Runner principal — exécute les phases en séquence."""
    t_start = time.time()
    print("\n" + "═" * 70)
    print(f"  PIPELINE OLLAMA — {'🧪 TEST' if test_mode else '🚀 PROD'}")
    print(f"  CSV : {csv_path} | Jobs : {max_jobs} | CVs/job : {max_cvs}")
    print("═" * 70)

    if run_graph:
        run_phase1_graph_generation(
            csv_path=csv_path,
            test_mode=test_mode, max_jobs=max_jobs,
        )

    if run_p2:
        run_phase2(
            csv_path=csv_path,
            test_mode=test_mode, max_jobs=max_jobs, max_cvs=max_cvs,
        )

    if run_p3:
        run_phase3(test_mode=test_mode)

  
    if run_p3:
        run_phase3_5(
            csv_path=csv_path,
            test_mode=test_mode, max_jobs=max_jobs, max_cvs=max_cvs,
        )

    if run_p4:
        run_phase4(test_mode=test_mode)

    elapsed = round(time.time() - t_start, 1)
    print("\n" + "═" * 70)
    print(f"  ✅ PIPELINE TERMINÉ en {elapsed}s")
    print(f"  Graphes   : {len(GRAPHS_STORE)}")
    print(f"  Phase 2   : {len(PHASE2_STORE)}")
    print(f"  Phase 3   : {len(PHASE3_STORE)}")
    print(f"  Biais     : {len(BIAS_STORE)}")
    print(f"  Rapports  : {len(REPORTS_STORE)}")
    print("═" * 70)


if __name__ == "__main__":
   
    run_full_pipeline(
        csv_path  = CSV_INPUT,
        test_mode = True,
        max_jobs  = 2,
        max_cvs   = 2,
        run_graph = True,
        run_p2    = True,
        run_p3    = True,
        run_p4    = True,
    )
