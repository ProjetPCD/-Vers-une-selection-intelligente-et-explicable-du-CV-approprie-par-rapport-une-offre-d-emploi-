# -*- coding: utf-8 -*-

import anthropic
import time
import itertools
from google.colab import drive
drive.mount('/content/drive')

import os

CSV_PATH   = '/content/drive/MyDrive/job(1).csv'
OUTPUT_DIR = '/content/drive/MyDrive/naive_baseline_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'✅ Drive monté')
print(f'   Source  : {CSV_PATH}')
print(f'   Sorties : {OUTPUT_DIR}')

"""## 2. 📦 Installation"""

!pip install -q huggingface_hub pandas anthropic

"""## 3. 🔑 Clé Hugging Face"""

from google.colab import userdata
import os

try:
    HF_TOKEN = userdata.get('HF_TOKEN')
    print('✅ Token chargé depuis les secrets Colab')
except Exception:
    import getpass
    HF_TOKEN = getpass.getpass('🔑 Entre ta clé Hugging Face (hf_...) : ')

os.environ['HF_TOKEN'] = HF_TOKEN
print('Token :', HF_TOKEN[:8] + '...')

"""## 4. ⚙️ Client Hugging Face"""

from huggingface_hub import InferenceClient

MODEL_ID = 'meta-llama/Meta-Llama-3-70B-Instruct'
client   = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
print(f'✅ Client prêt → {MODEL_ID}')

"""## 5. 📂 Chargement du CSV"""

import pandas as pd

df_data = pd.read_csv(CSV_PATH)

JOB_COL = 'job_adequate'
CV_COLS = ['cv1', 'cv2', 'cv3', 'cv4', 'cv5']

missing = [c for c in [JOB_COL] + CV_COLS if c not in df_data.columns]
if missing:
    print(f'⚠️  Colonnes manquantes : {missing}')
    print(f'   Colonnes disponibles : {list(df_data.columns)}')
else:
    total_pairs = len(df_data) * len(CV_COLS)
    print(f'✅ CSV chargé : {len(df_data)} job(s) × {len(CV_COLS)} CV = {total_pairs} paires')
    display(df_data[[JOB_COL] + CV_COLS].head(3))

"""## 6. 📝 Prompt simplifié — Baseline honnête
Un seul prompt, une seule sortie : score · label · 2 phrases de justification.  
Pas de métriques calculées — juste ce qu'un LLM peut produire de façon fiable.
"""

import json, re
from datetime import datetime

SYSTEM_PROMPT = """
Tu es un recruteur expert.
Étant donné une offre d'emploi et un CV, évalue si le candidat est un bon match.
Retourne UNIQUEMENT un JSON valide, sans markdown ni backticks.

Structure exacte :
{
  "score": <entier de 0 à 100>,
  "label": "Strong Match" | "Good Fit" | "Moderate Match" | "Weak Match" | "Poor Match",
  "justification": "<exactement 2 phrases expliquant le verdict>"
}

Règles du label basées sur le score :
- 85-100 → Strong Match
- 70-84  → Good Fit
- 50-69  → Moderate Match
- 30-49  → Weak Match
- 0-29   → Poor Match
"""

def build_prompt(cv_text: str, job_text: str) -> str:
    return (
        f"OFFRE D'EMPLOI :\n---\n{str(job_text).strip()}\n\n"
        f"CV DU CANDIDAT :\n---\n{str(cv_text).strip()}\n\n"
        "Retourne le JSON maintenant."
    )

def parse_json(raw: str) -> dict:
    clean = re.sub(r'```json|```', '', raw).strip()
    m = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError(f'JSON introuvable dans : {raw[:300]}')

def score_to_label(score: int) -> str:
    if score >= 85: return 'Strong Match'
    if score >= 70: return 'Good Fit'
    if score >= 50: return 'Moderate Match'
    if score >= 30: return 'Weak Match'
    return 'Poor Match'

def run_matching(cv_text, job_text, job_idx, cv_col, temperature=0.1):
    raw = client.chat_completion(
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': build_prompt(cv_text, job_text)},
        ],
        max_tokens=400,
        temperature=temperature,
        stream=False,
    ).choices[0].message.content

    r = parse_json(raw)

    r['score']         = int(r.get('score', 0))
    r['label']         = r.get('label', score_to_label(r['score']))
    r['label_checked'] = score_to_label(r['score'])
    r['job_idx']       = int(job_idx)
    r['cv_col']        = cv_col
    r['model']         = MODEL_ID
    r['generated_at']  = datetime.now().isoformat()
    return r

print('✅ Prompt baseline et fonctions chargés')

"""## 7. 🚀 Lancement — toutes les paires Job × CV"""

import csv

TEST_ROWS   = None
TEMPERATURE = 0.1

from datetime import datetime
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LIVE_CSV  = os.path.join(OUTPUT_DIR, f'baseline_live_{RUN_TIMESTAMP}.csv')
LIVE_JSON = os.path.join(OUTPUT_DIR, f'baseline_live_{RUN_TIMESTAMP}.json')

CSV_FIELDNAMES = ['job_idx', 'cv_col', 'label', 'score', 'label_checked', 'justification', 'model', 'generated_at']

with open(LIVE_CSV, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
    writer.writeheader()

with open(LIVE_JSON, 'w', encoding='utf-8') as f:
    json.dump([], f)

print(f'📂 Sauvegarde incrémentale dans :')
print(f'   CSV  → {LIVE_CSV}')
print(f'   JSON → {LIVE_JSON}\n')


def save_result(r: dict):
    """Ajoute le résultat r au CSV et au JSON à chaud, sans attendre la fin."""
    with open(LIVE_CSV, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
        writer.writerow(r)

    with open(LIVE_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data.append(r)
    with open(LIVE_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


subset      = df_data if TEST_ROWS is None else df_data.head(TEST_ROWS)
total       = len(subset) * len(CV_COLS)
done        = 0
all_results = []
errors      = []

print(f'🔁 {len(subset)} job(s) × {len(CV_COLS)} CV = {total} paires\n')

for job_idx, row in subset.iterrows():
    job_text = row[JOB_COL]

    if pd.isna(job_text) or str(job_text).strip() == '':
        print(f'  ⏩ Job #{job_idx} — offre vide, ignoré')
        continue

    for cv_col in CV_COLS:
        cv_text = row.get(cv_col, '')

        if pd.isna(cv_text) or str(cv_text).strip() == '':
            print(f'  ⏩ Job #{job_idx} / {cv_col} — CV vide, ignoré')
            continue

        done += 1
        print(f'  [{done}/{total}] Job #{job_idx} × {cv_col} ...', end=' ', flush=True)

        try:
            r = run_matching(
                cv_text=cv_text, job_text=job_text,
                job_idx=job_idx, cv_col=cv_col,
                temperature=TEMPERATURE,
            )
            all_results.append(r)
            save_result(r)
            print(f'{r["label"]} | score={r["score"]} | {r["justification"][:60]}... 💾')

        except Exception as e:
            errors.append({'job_idx': job_idx, 'cv_col': cv_col, 'error': str(e)})
            print(f'❌ {e}')

print(f'\n✅ {len(all_results)} matching(s) réussi(s), {len(errors)} erreur(s)')
print(f'💾 Résultats dans Drive : {OUTPUT_DIR}')

"""## 8. 📊 Tableau de résultats baseline"""

METRIC_COLS = ['job_idx', 'cv_col', 'label', 'score', 'label_checked', 'justification']

df_results = pd.DataFrame(all_results)
cols_ok    = [c for c in METRIC_COLS if c in df_results.columns]
df_display = df_results[cols_ok].copy()

pd.set_option('display.max_colwidth', 120)
pd.set_option('display.max_rows', 200)
display(df_display)

print('\n--- Répartition des labels ---')
print(df_results['label'].value_counts().to_string())

print('\n--- Meilleur CV par job (score baseline) ---')
best = (
    df_results
    .loc[df_results.groupby('job_idx')['score'].idxmax()]
    [['job_idx', 'cv_col', 'label', 'score']]
    .reset_index(drop=True)
)
display(best)

"""## 9. 💾 Sauvegarde dans Google Drive"""

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

csv_path = os.path.join(OUTPUT_DIR, f'baseline_results_{timestamp}.csv')
df_display.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f'✅ CSV    → {csv_path}')

json_path = os.path.join(OUTPUT_DIR, f'baseline_results_{timestamp}.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f'✅ JSON   → {json_path}')

if errors:
    err_path = os.path.join(OUTPUT_DIR, f'baseline_errors_{timestamp}.json')
    with open(err_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    print(f'⚠️  Erreurs → {err_path} ({len(errors)} paire(s) échouée(s))')

print(f'\n📁 Tous les fichiers dans : {OUTPUT_DIR}')

"""## 10. ⚖️ Claude-as-Judge — Comparaison Baseline vs Pipeline

Cette cellule charge les deux JSON (baseline + résultats de ton pipeline),  
puis demande à **Claude** de juger pour chaque paire job×CV :
- Quel système a mieux rankéle candidat ?
- Y a-t-il un accord ou désaccord sur le verdict final ?
- Lequel est plus fiable ?

Le client Claude tourne avec **rotation automatique de clés Anthropic** :  
si une clé atteint son quota, il passe automatiquement à la suivante.
"""

from google.colab import userdata as _ud

def _load_secret(name, fallback=None):
    try:
        return _ud.get(name)
    except Exception:
        return fallback

ANTHROPIC_KEYS = [
    k for k in [
        _load_secret('ANTHROPIC_KEY_1'),
        _load_secret('ANTHROPIC_KEY_2'),
        _load_secret('ANTHROPIC_KEY_3'),
    ]
    if k
]

if not ANTHROPIC_KEYS:
    import getpass
    k1 = getpass.getpass('🔑 Clé Anthropic #1 (sk-ant-...) : ')
    ANTHROPIC_KEYS = [k1]
    more = input('Ajouter une autre clé ? (laisser vide pour continuer) : ').strip()
    if more:
        ANTHROPIC_KEYS.append(more)

print(f'✅ {len(ANTHROPIC_KEYS)} clé(s) Anthropic chargée(s)')


BASELINE_JSON_PATH = json_path
PIPELINE_JSON_PATH = '/content/drive/MyDrive/pipeline_results/results.json'


JUDGE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'judge_results')
os.makedirs(JUDGE_OUTPUT_DIR, exist_ok=True)
print(f'✅ Config chargée. Sortie jugement → {JUDGE_OUTPUT_DIR}')





class RotatingAnthropicClient:
    """
    Wrapper autour du client Anthropic SDK.
    Tourne sur la liste de clés : si une clé retourne RateLimitError
    ou OverloadedError, elle passe automatiquement à la suivante.
    Si toutes les clés sont épuisées, attend RETRY_WAIT secondes
    puis recommence depuis la première.
    """
    RETRY_WAIT   = 60
    MAX_CYCLES   = 3
    QUOTA_ERRORS = (
        'rate_limit_error',
        'overloaded_error',
        'quota',
        'too many requests',
    )

    def __init__(self, keys: list[str]):
        assert keys, 'Aucune clé Anthropic fournie !'
        self.keys    = keys
        self.clients = [anthropic.Anthropic(api_key=k) for k in keys]
        self._idx    = 0
        print(f'  🔄 RotatingClient initialisé avec {len(keys)} clé(s)')

    def _is_quota_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(e in msg for e in self.QUOTA_ERRORS)

    def messages_create(self, **kwargs) -> anthropic.types.Message:
        """Drop-in replacement de client.messages.create() avec rotation."""
        for cycle in range(self.MAX_CYCLES):
            for attempt in range(len(self.keys)):
                idx    = (self._idx + attempt) % len(self.keys)
                client_i = self.clients[idx]
                try:
                    result = client_i.messages.create(**kwargs)
                    self._idx = idx
                    return result
                except Exception as e:
                    if self._is_quota_error(e):
                        key_short = self.keys[idx][:12] + '...'
                        print(f'    ⚠️  Quota/rate-limit sur clé #{idx+1} ({key_short}), '
                              f'passage à la suivante ...')
                        continue
                    raise

            if cycle < self.MAX_CYCLES - 1:
                print(f'  ⏳ Toutes les clés à quota. Attente {self.RETRY_WAIT}s '
                      f'(cycle {cycle+1}/{self.MAX_CYCLES}) ...')
                time.sleep(self.RETRY_WAIT)
        raise RuntimeError('Toutes les clés Anthropic sont à quota après '
                           f'{self.MAX_CYCLES} cycles.')


rotating_client = RotatingAnthropicClient(ANTHROPIC_KEYS)
print('✅ Client Anthropic avec rotation prêt')

JUDGE_SYSTEM = """
Tu es un évaluateur neutre de systèmes de matching CV/offre d'emploi.
On te donne, pour une même paire (offre, CV) :
  • La sortie du système BASELINE (score simple 0-100 + label + justification)
  • La sortie du système PIPELINE (métriques avancées : ntcw, ctm, combo_v6, etc.)

Retourne UNIQUEMENT un JSON valide, sans markdown ni backticks :
{
  "agreement": true | false,
  "better_system": "baseline" | "pipeline" | "tie",
  "ranking_consistent": true | false,
  "baseline_score_normalized": <float 0-100, le score baseline>,
  "pipeline_score_normalized": <float 0-100, le combo_v6_final ou match_pct du pipeline>,
  "judge_score": <float 0-100, ton estimation indépendante du match>,
  "verdict": "Strong Match" | "Good Fit" | "Moderate Match" | "Weak Match" | "Poor Match",
  "rationale": "<2-3 phrases : accord/désaccord, lequel est plus fiable et pourquoi>"
}

Règles :
- agreement=true si les deux systèmes donnent le même label ou un label adjacent
- better_system est le système dont le score/label te semble le mieux refléter la réalité
- ranking_consistent=true si le CV le mieux rankédans chaque système est le même
"""


def build_judge_prompt(job_text, cv_text, baseline_result, pipeline_result):
    return (
        f"OFFRE D'EMPLOI :\n---\n{str(job_text).strip()}\n\n"
        f"CV DU CANDIDAT :\n---\n{str(cv_text).strip()}\n\n"
        f"RÉSULTAT BASELINE :\n{json.dumps(baseline_result, ensure_ascii=False, indent=2)}\n\n"
        f"RÉSULTAT PIPELINE :\n{json.dumps(pipeline_result, ensure_ascii=False, indent=2)}\n\n"
        "Donne ton jugement JSON maintenant."
    )


def run_judge(job_text, cv_text, baseline_result, pipeline_result,
              job_idx, cv_col, temperature=0.0):
    response = rotating_client.messages_create(
        model='claude-opus-4-5',
        max_tokens=600,
        temperature=temperature,
        system=JUDGE_SYSTEM,
        messages=[{
            'role': 'user',
            'content': build_judge_prompt(job_text, cv_text,
                                          baseline_result, pipeline_result)
        }]
    )
    raw = response.content[0].text
    r   = parse_json(raw)
    r['job_idx']      = int(job_idx)
    r['cv_col']       = cv_col
    r['judge_model']  = 'claude-opus-4-5'
    r['generated_at'] = datetime.now().isoformat()
    return r


print('✅ Prompt juge et fonctions prêts')

import glob, re

with open(BASELINE_JSON_PATH, 'r', encoding='utf-8') as f:
    baseline_data = json.load(f)

baseline_index = {(r['job_idx'], r['cv_col']): r for r in baseline_data}
print(f'✅ Baseline chargé : {len(baseline_data)} résultats')


PIPELINE_DIR = '/content/drive/MyDrive/pipeline_results'

pipeline_index = {}
_pattern = re.compile(r'report_job_(\d+)_cv_(\d+)\.json$')

for fpath in glob.glob(os.path.join(PIPELINE_DIR, 'report_job_*_cv_*.json')):
    m = _pattern.search(os.path.basename(fpath))
    if not m:
        continue
    job_idx = int(m.group(1))
    cv_num  = int(m.group(2))
    cv_col  = f'cv{cv_num}'
    with open(fpath, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    pipeline_index[(job_idx, cv_col)] = raw

print(f'✅ Pipeline chargé : {len(pipeline_index)} fichiers individuels')


def extract_pipeline_summary(p: dict) -> dict:
    """Extrait les champs clés d'un rapport pipeline pour le juge."""
    oa   = p.get('overall_assessment', {})
    am   = p.get('advanced_metrics', {})
    amv6 = p.get('advanced_metrics_v6', {})
    cv6  = p.get('combo_scores_v6', {})
    pr   = p.get('pool_ranking', {})
    dr   = p.get('decision_rationale', {})
    gaps = [g['skill'] for g in p.get('critical_gaps', [])[:5]]
    strengths = p.get('profile_summary', {}).get('strengths', [])

    return {
        'verdict'            : oa.get('verdict'),
        'match_pct'          : oa.get('match_pct'),
        'confidence'         : oa.get('confidence', {}).get('confidence_label'),
        'ntcw'               : am.get('NTCW', {}).get('value'),
        'ctm'                : am.get('CTM', {}).get('value'),
        'weighted_f1'        : amv6.get('weighted_f1', {}).get('value'),
        'gap_score'          : amv6.get('gap_score', {}).get('value'),
        'semantic_fit'       : amv6.get('semantic_fit_score'),
        'missing_critical'   : amv6.get('missing_critical_penalty', {}).get('value'),
        'combo_v6_final'     : cv6.get('combo_v6_final', {}).get('value'),
        'combo_ranking'      : cv6.get('combo_ranking', {}).get('value'),
        'pool_rank'          : pr.get('rank'),
        'pool_size'          : pr.get('pool_size'),
        'pool_tier'          : pr.get('tier'),
        'top5_critical_gaps' : gaps,
        'strengths'          : strengths,
        'rationale'          : dr.get('why_this_verdict', ''),
    }


common_keys = sorted(set(baseline_index) & set(pipeline_index))
only_baseline = set(baseline_index) - set(pipeline_index)
only_pipeline = set(pipeline_index) - set(baseline_index)

print(f'\n   Paires communes    : {len(common_keys)}')
if only_baseline:
    print(f'   ⚠️  Baseline seul   : {sorted(only_baseline)}')
if only_pipeline:
    print(f'   ⚠️  Pipeline seul   : {sorted(only_pipeline)}')

judge_results = []
judge_errors  = []

for i, (job_idx, cv_col) in enumerate(common_keys, 1):
    b_res = baseline_index[(job_idx, cv_col)]
    p_raw = pipeline_index[(job_idx, cv_col)]
    p_res = extract_pipeline_summary(p_raw)

    job_text = df_data.loc[job_idx, JOB_COL] if job_idx in df_data.index else ''
    cv_text  = df_data.loc[job_idx, cv_col]  if job_idx in df_data.index else ''

    print(f'  [{i}/{len(common_keys)}] Job #{job_idx} × {cv_col}'
          f' — baseline={b_res.get("score", "?")}'
          f' | pipeline_combo={p_res.get("combo_v6_final", "?")}'
          f' | rank={p_res.get("pool_rank", "?")}/{p_res.get("pool_size", "?")} ... ',
          end='', flush=True)

    try:
        r = run_judge(
            job_text=job_text, cv_text=cv_text,
            baseline_result=b_res, pipeline_result=p_res,
            job_idx=job_idx, cv_col=cv_col
        )
        judge_results.append(r)
        icon = '✅' if r['agreement'] else '⚡'
        print(f"{icon} accord={r['agreement']} | meilleur={r['better_system']}"
              f" | judge_score={r.get('judge_score', '?')}")

    except Exception as e:
        judge_errors.append({'job_idx': job_idx, 'cv_col': cv_col, 'error': str(e)})
        print(f'❌ {e}')

print(f'\n✅ {len(judge_results)} jugements réussis, {len(judge_errors)} erreur(s)')

df_judge = pd.DataFrame(judge_results)

print('═' * 60)
print('  RÉSUMÉ — Claude-as-Judge')
print('═' * 60)

total_j = len(df_judge)

pct_agreement = df_judge['agreement'].mean() * 100
print(f'  Accord baseline ↔ pipeline : {pct_agreement:.1f}% ({df_judge["agreement"].sum()}/{total_j})')

print('\n  Système jugé meilleur :')
print(df_judge['better_system'].value_counts().to_string())

pct_rank = df_judge['ranking_consistent'].mean() * 100
print(f'\n  Ranking cohérent entre systèmes : {pct_rank:.1f}%')

print(f'\n  Score moyen baseline  : {df_judge["baseline_score_normalized"].mean():.1f}')
print(f'  Score moyen pipeline  : {df_judge["pipeline_score_normalized"].mean():.1f}')
print(f'  Score moyen juge      : {df_judge["judge_score"].mean():.1f}')

display_cols = [
    'job_idx', 'cv_col', 'agreement', 'better_system',
    'baseline_score_normalized', 'pipeline_score_normalized',
    'judge_score', 'verdict', 'rationale'
]
cols_ok = [c for c in display_cols if c in df_judge.columns]
pd.set_option('display.max_colwidth', 150)
display(df_judge[cols_ok])

disagreements = df_judge[~df_judge['agreement']]
if not disagreements.empty:
    print(f'\n⚡ {len(disagreements)} cas de désaccord :')
    display(disagreements[cols_ok])

judge_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

judge_csv = os.path.join(JUDGE_OUTPUT_DIR, f'judge_{judge_timestamp}.csv')
df_judge[cols_ok].to_csv(judge_csv, index=False, encoding='utf-8-sig')
print(f'✅ CSV juge    → {judge_csv}')

judge_json = os.path.join(JUDGE_OUTPUT_DIR, f'judge_{judge_timestamp}.json')
with open(judge_json, 'w', encoding='utf-8') as f:
    json.dump(judge_results, f, indent=2, ensure_ascii=False)
print(f'✅ JSON juge   → {judge_json}')


if judge_errors:
    err_path = os.path.join(JUDGE_OUTPUT_DIR, f'judge_errors_{judge_timestamp}.json')
    with open(err_path, 'w', encoding='utf-8') as f:
        json.dump(judge_errors, f, indent=2, ensure_ascii=False)
    print(f'⚠️  Erreurs juge → {err_path}')

print(f'\n📁 Résultats juge dans : {JUDGE_OUTPUT_DIR}')