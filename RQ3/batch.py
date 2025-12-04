import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
except Exception:
    # google.genai is optional at static analysis time; use None fallback and emit a warning.
    genai = None
    types = None


def find_datasets_dir(start: Optional[Path] = None) -> Path:
    start = start or Path.cwd()
    for path in (start, *start.parents):
        candidate = path / "datasets"
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "datasets directory not found; run from repo root or pass start."
    )


def build_genai_client():
    if genai is None:
        raise RuntimeError(
            "google-genai not installed. pip install google-genai and re-run."
        )
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Export GEMINI_API_KEY or configure your Vertex AI credentials."
        )
    return genai.Client(api_key=api_key)


DATASETS_DIR = find_datasets_dir()
PROJECT_ROOT = DATASETS_DIR.parent
client = build_genai_client()

# ============================
# Configuración básica
# ============================
RQ3_DIR = PROJECT_ROOT / "RQ3"
RQ3_DIR.mkdir(exist_ok=True, parents=True)
out_path = RQ3_DIR / "rq3_validation_evidence.parquet"

BATCH_SIZE = 20  # puedes subir a 30–40 si quieres, mientras no pases ~20MB total

# ============================
# Prompt (mismo esquema de JSON)
# ============================
SYSTEM_PROMPT = """You classify evidence of performance validation for a PR.
Return compact JSON only with keys:
validation_present (bool), evidence_sources (list of "pipeline","description","comments"),
validation_type (benchmark,profiling,load/canary,unit-only,unspecified,none),
validation_description (short text),
pipeline_signal (short), description_signal (short), comment_signal (short).

Rules:
- Pipelines count only if workflow names clearly imply performance/benchmark/load/canary;
  if they are only unit/lint/generic CI, note that but do NOT treat them as perf validation.
- Description/comments count if they mention performance benchmarks, profiling, latency/throughput
  numbers, CPU/memory usage, load/canary rollout, A/B tests, perf tools, or explicit "no perf validation".
- If nothing indicates performance validation, set:
  validation_present = false,
  validation_type = "none",
  evidence_sources = [].
- If there are only unit tests or generic CI without perf focus, use:
  validation_present = true,
  validation_type = "unit-only",
  evidence_sources = ["pipeline"] (and/or others if mentioned).
- Prefer a single best-fitting validation_type even if multiple signals are present.
Return STRICT JSON only, no extra commentary, no markdown.
"""

# ============================
# Cargar progreso existente
# ============================
try:
    df_existing = pd.read_parquet(out_path)
    if not df_existing.empty and "validation_type" in df_existing.columns:
        df_existing = df_existing[df_existing["validation_type"] != "error"]
    records = df_existing.to_dict(orient="records")
    print(f"Cargadas {len(records)} filas existentes desde {out_path}")
except FileNotFoundError:
    df_existing = pd.DataFrame()
    records = []
    print("No existe parquet previo, empezando desde cero.")

# Set de PRs ya procesados (por author_type + pr_id) que tuvieron salida válida
processed_pairs = set()
if not df_existing.empty:
    for _, row in df_existing.iterrows():
        try:
            if row.get("validation_type") != "error":
                processed_pairs.add((row["author_type"], int(row["pr_id"])))
        except Exception:
            continue


def load_pr_core(prefix: str) -> pd.DataFrame:
    commits = pd.read_parquet(
        DATASETS_DIR / f"{prefix}_pr" / f"{prefix}_pr_commits.parquet"
    )
    return commits.drop_duplicates("pr_id").set_index("pr_id")


def collect_comments(prefix: str, pr_id: int) -> List[str]:
    issue = pd.read_parquet(
        DATASETS_DIR / f"{prefix}_pr" / f"{prefix}_pr_issue_comments.parquet"
    )
    review = pd.read_parquet(
        DATASETS_DIR / f"{prefix}_pr" / f"{prefix}_pr_review_comments.parquet"
    )
    texts = []
    for df in (issue, review):
        subset = df[df["pr_id"] == pr_id]
        texts.extend(subset["body"].dropna().tolist())
    return texts


def collect_pipeline_names(prefix: str, pr_id: int) -> List[str]:
    workflows = pd.read_parquet(
        DATASETS_DIR / f"{prefix}_pr" / f"{prefix}_pr_workflow_runs.parquet"
    )
    subset = workflows[workflows["pr_id"] == pr_id]
    return sorted(subset["workflow_name"].dropna().unique().tolist())


def pr_ids_from_commits(prefix: str, limit: Optional[int] = None) -> Iterable[int]:
    commits = pd.read_parquet(
        DATASETS_DIR / f"{prefix}_pr" / f"{prefix}_pr_commits.parquet"
    )
    pr_ids = sorted(commits["pr_id"].dropna().astype(int).unique().tolist())
    return pr_ids if limit is None else pr_ids[:limit]


# ============================
# Cargar cores y listas de PRs
# ============================
ai_core = load_pr_core("ai")
human_core = load_pr_core("human")

ai_ids_all = list(pr_ids_from_commits("ai", limit=None))
human_ids_all = list(pr_ids_from_commits("human", limit=None))

pending_ai_ids = [
    int(pid) for pid in ai_ids_all if ("ai_agent", int(pid)) not in processed_pairs
]
pending_human_ids = [
    int(pid) for pid in human_ids_all if ("human", int(pid)) not in processed_pairs
]

print(f"AI PRs totales: {len(ai_ids_all)}, pendientes: {len(pending_ai_ids)}")
print(f"Human PRs totales: {len(human_ids_all)}, pendientes: {len(pending_human_ids)}")
print(f"PRs pendientes en total: {len(pending_ai_ids) + len(pending_human_ids)}")


# ============================
# Helper: construir GenerateContentRequest para cada PR
# ============================
def build_request_for_pr(
    prefix: str, pr_id: int, author_type: str, pr_core: pd.DataFrame
) -> dict:
    row = pr_core.loc[pr_id]
    pipeline_names = collect_pipeline_names(prefix, pr_id)
    comments = collect_comments(prefix, pr_id)
    description = (row.get("pr_description") or "").strip()

    repo_owner = row.get("repo_owner")
    repo_name = row.get("repo_name")
    pr_number = row.get("pr_number")
    pr_title = (row.get("pr_title") or "").strip()

    user_prompt = f"""
You are given information about a pull request that claims performance-related changes.
Use the instructions above to classify evidence of performance validation.

PR metadata:
- pr_id: {pr_id}
- author_type: {author_type}
- repo: {repo_owner}/{repo_name}
- pr_number: {pr_number}
- pr_title: {pr_title}

Pipelines (names only):
{json.dumps(pipeline_names, indent=2)}

PR description:
{description or "<empty>"}

Combined issue + review comments:
{comments or "<no comments>"}
"""
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": SYSTEM_PROMPT + "\n\n" + user_prompt}],
            }
        ],
        # GenerateContentConfig: temperature sits at top-level, not nested under generation_config.
        "config": {"temperature": 0.0, "response_mime_type": "application/json"},
    }


# ============================
# Helper: llamar a Gemini Batch API
# ============================
def run_batch_requests(inline_requests: list, display_name: str):
    inline_batch_job = client.batches.create(
        model="gemini-3-pro-preview",
        src=inline_requests,
        config={
            "display_name": display_name,
        },
    )

    job_name = inline_batch_job.name
    print(f"Created batch job: {job_name}")
    print(f"Polling status for job: {job_name}")

    completed_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    while True:
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state.name
        if state in completed_states:
            print(f"Job finished with state: {state}")
            break
        print(f"Job not finished. Current state: {state}. Waiting 30 seconds...")
        time.sleep(30)

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(
            f"Batch job did not succeed. Final state: {batch_job.state.name}"
        )

    # Inline requests → resultados en batch_job.dest.inlined_responses
    return batch_job.dest.inlined_responses


# ============================
# Helper: construir record de salida a partir de la respuesta
# ============================
def build_record_from_response(
    pr_id: int,
    author_type: str,
    prefix: str,
    pr_core: pd.DataFrame,
    inline_response,
):
    row = pr_core.loc[pr_id]
    repo = f"{row.get('repo_owner')}/{row.get('repo_name')}"
    pr_number = row.get("pr_number")
    pr_title = row.get("pr_title")

    if inline_response.error:
        return {
            "pr_id": pr_id,
            "author_type": author_type,
            "repo": repo,
            "pr_number": pr_number,
            "pr_title": pr_title,
            "pipeline_names": collect_pipeline_names(prefix, pr_id),
            "validation_present": None,
            "evidence_sources": [],
            "validation_type": "error",
            "validation_description": f"batch error: {inline_response.error}",
            "pipeline_signal": "",
            "description_signal": "",
            "comment_signal": "",
        }

    text = ""
    try:
        # La doc dice que normalmente puedes usar .text
        text = inline_response.response.text
    except Exception:
        # fallback genérico
        text = str(inline_response.response)

    try:
        parsed = json.loads(text)
    except Exception as exc:
        parsed = {
            "validation_present": None,
            "evidence_sources": [],
            "validation_type": "error",
            "validation_description": f"json parse error: {exc}",
            "pipeline_signal": "",
            "description_signal": "",
            "comment_signal": "",
        }

    evidence_sources = parsed.get("evidence_sources") or []
    if not isinstance(evidence_sources, list):
        evidence_sources = []

    return {
        "pr_id": pr_id,
        "author_type": author_type,
        "repo": repo,
        "pr_number": pr_number,
        "pr_title": pr_title,
        "pipeline_names": collect_pipeline_names(prefix, pr_id),
        "validation_present": parsed.get("validation_present"),
        "evidence_sources": evidence_sources,
        "validation_type": parsed.get("validation_type"),
        "validation_description": parsed.get("validation_description"),
        "pipeline_signal": parsed.get("pipeline_signal"),
        "description_signal": parsed.get("description_signal"),
        "comment_signal": parsed.get("comment_signal"),
    }


# ============================
# Proceso principal en batch
# (primero AI pendientes, luego humanos pendientes)
# ============================
def save_partial(records, out_path: Path):
    df_tmp = pd.DataFrame(records)
    df_tmp.to_parquet(out_path, index=False)
    print(f"[partial save] Saved {len(df_tmp)} rows to {out_path}")


# ---- 1) AI pendientes ----
if pending_ai_ids:
    print("\n=== Procesando AI pendientes en batch ===")
    for start in range(0, len(pending_ai_ids), BATCH_SIZE):
        batch_ids = pending_ai_ids[start : start + BATCH_SIZE]
        print(f"\nAI batch {start // BATCH_SIZE + 1}: {len(batch_ids)} PRs")

        inline_requests = [
            build_request_for_pr("ai", pr_id, "ai_agent", ai_core)
            for pr_id in batch_ids
        ]

        try:
            inline_responses = run_batch_requests(
                inline_requests,
                display_name=f"rq3-ai-batch-{start // BATCH_SIZE + 1}",
            )
        except Exception as exc:
            print(f"ERROR en batch AI: {exc}")
            # si falla el batch, marcar cada PR con error
            for pr_id in batch_ids:
                records.append(
                    {
                        "pr_id": pr_id,
                        "author_type": "ai_agent",
                        "repo": "",
                        "pr_number": None,
                        "pr_title": "",
                        "pipeline_names": [],
                        "validation_present": None,
                        "evidence_sources": [],
                        "validation_type": "error",
                        "validation_description": f"batch-level error: {exc}",
                        "pipeline_signal": "",
                        "description_signal": "",
                        "comment_signal": "",
                    }
                )
            save_partial(records, out_path)
            continue

        for pr_id, inline_response in zip(batch_ids, inline_responses):
            rec = build_record_from_response(
                pr_id=pr_id,
                author_type="ai_agent",
                prefix="ai",
                pr_core=ai_core,
                inline_response=inline_response,
            )
            records.append(rec)

        save_partial(records, out_path)

# ---- 2) Human pendientes ----
if pending_human_ids:
    print("\n=== Procesando HUMAN pendientes en batch ===")
    for start in range(0, len(pending_human_ids), BATCH_SIZE):
        batch_ids = pending_human_ids[start : start + BATCH_SIZE]
        print(f"\nHuman batch {start // BATCH_SIZE + 1}: {len(batch_ids)} PRs")

        inline_requests = [
            build_request_for_pr("human", pr_id, "human", human_core)
            for pr_id in batch_ids
        ]

        try:
            inline_responses = run_batch_requests(
                inline_requests,
                display_name=f"rq3-human-batch-{start // BATCH_SIZE + 1}",
            )
        except Exception as exc:
            print(f"ERROR en batch HUMAN: {exc}")
            for pr_id in batch_ids:
                records.append(
                    {
                        "pr_id": pr_id,
                        "author_type": "human",
                        "repo": "",
                        "pr_number": None,
                        "pr_title": "",
                        "pipeline_names": [],
                        "validation_present": None,
                        "evidence_sources": [],
                        "validation_type": "error",
                        "validation_description": f"batch-level error: {exc}",
                        "pipeline_signal": "",
                        "description_signal": "",
                        "comment_signal": "",
                    }
                )
            save_partial(records, out_path)
            continue

        for pr_id, inline_response in zip(batch_ids, inline_responses):
            rec = build_record_from_response(
                pr_id=pr_id,
                author_type="human",
                prefix="human",
                pr_core=human_core,
                inline_response=inline_response,
            )
            records.append(rec)

        save_partial(records, out_path)

# ============================
# Guardado final
# ============================
df_final = pd.DataFrame(records)
df_final.to_parquet(out_path, index=False)
print(f"\n✅ Saved FINAL {len(df_final)} rows to {out_path}")
