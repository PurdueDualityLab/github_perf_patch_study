# GitHub Performance Patch Study

Comparison of performance-focused pull requests opened by AI coding agents and human developers. The repository now hosts the entire analysis pipeline: raw data filtering, LLM-based optimization-pattern labeling, language/time-to-merge statistics, cyclomatic-complexity deltas, and validation of test evidence. Use this README to navigate the code and reproduce the study end to end.

## Repository Layout

```
.
├── datasets/
│   ├── ai_pr/                        # AI PR extracts + cached parquet files
│   ├── human_pr/                     # Human PR extracts + cached parquet files
│   ├── pr_filtering/                 # Notebook that filters valid perf PR ids
│   └── performance_prs_*.csv         # Unified AI vs human datasets
├── RQ1_merge_time_and_language/      # Merge/review latency and language mix
├── RQ2_pattern_analysis/             # LLM prompts, catalogs, and pattern stats
├── RQ3_complexity_analysis/          # Lizard-based maintainability deltas
├── RQ4_test_and_validation/          # Evidence of testing/validation signals
├── ai_pr_* / human_pr_* .parquet     # Convenience caches outside datasets/
├── human_pr.ipynb                    # Legacy EDA / sanity-check notebook
└── requirements.txt
```

Key sub-folders:
- **datasets/** contains notebook-driven pipelines (`ai_pr/ai_pr.ipynb`, `human_pr/human_pr.ipynb`, `pr_filtering/get_valid_pr.ipynb`) that read Hugging Face parquet dumps, filter to performance work, and materialize curated CSVs such as `performance_prs_ai_vs_human.csv`.
- **RQ1_merge_time_and_language** holds `rq1_analysis.ipynb`, which generates descriptive figures (e.g., `review_time_distribution.png`) about merge time, review latency, description quality, and programming-language mix.
- **RQ2_pattern_analysis** keeps the optimization-pattern taxonomy (`catalog/`), multi-model labeling notebooks (`optimization_pattern_detection_gpt.ipynb`, `optimization_pattern_detection_gemini.ipynb`), the Qwen/Ollama script (`optimization_pattern_detection_qwen.py`), and comparison utilities like `compare_pattern.py`. Intermediate model transcripts live in `llm_data/`, and chart-ready tables in `results/`.
- **RQ3_complexity_analysis** provides CLI scripts (`ai.py`, `human.py`) that download patches via the GitHub API and call `lizard` to compute NLOC/CCN deltas per PR. Outputs land in `RQ3_complexity_analysis/data/` with plots in `results/`.
- **RQ4_test_and_validation** hosts notebooks (`validation-gpt.ipynb`, `validation-gemini2.ipynb`, `analysis.ipynb`) plus parquet checkpoints that compare the narrative evidence AI vs human authors supply for testing/benchmarking. Mismatched cases are logged under `unmatched_data/`.

## Data Sources

1. **Hugging Face `hao-li/AIDev` dataset** – All notebooks reference the following parquet tables via the `hf://` URI scheme (downloaded on demand with `datasets`/`fsspec`):
   - `pull_request.parquet`, `human_pull_request.parquet`
   - `pr_task_type.parquet`, `human_pr_task_type.parquet`
   - `pr_commit_details.parquet`
   - `all_repository.parquet`
2. **Local caches** – Frequently accessed aggregates are stored at the repo root (`ai_pr_commits.parquet`, `ai_pr_commit_details.parquet`, `human_pr_commits.parquet`, etc.) to avoid re-downloading. The curated join of AI and human PRs sits in `datasets/performance_prs_ai_vs_human.csv` and `datasets/performance_prs_ai_vs_human_raw.csv`.
3. **LLM outputs** – Intermediate GPT/Gemini/Qwen responses are saved under `RQ2_pattern_analysis/llm_data/`. Validation evidence for RQ4 is persisted as parquet files in `RQ4_test_and_validation/` so notebooks can focus on analysis rather than regeneration.

Authenticate with Hugging Face (`huggingface-cli login`) before running notebooks so `hf://` reads succeed. Large parquet files are intentionally kept out of Git history—expect long downloads on first run.

## Environment & Credentials

1. Use Python 3.10+ and install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   pip install lizard PyGithub huggingface_hub  # used by RQ3 scripts
   ```
2. Create a `.env` in the repo root for GitHub API access (used by `RQ3_complexity_analysis/*.py` and any notebook touching the GitHub REST API):
   ```
   GITHUB_TOKEN=ghp_your_personal_access_token
   ```
3. LLM-specific tooling:
   - `optimization_pattern_detection_qwen.py` calls a local Ollama endpoint; configure `OLLAMA_HOST` or update the client block if you point at a remote model server.
   - The GPT/Gemini notebooks expect API keys to be loaded via environment variables (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`). Store them in `.env` and load with `python-dotenv` inside the notebooks.

## Running the Research Questions

1. **Prepare datasets** (once per machine)
   - Run `datasets/pr_filtering/get_valid_pr.ipynb` to regenerate `valid_perf_pr_ids.csv` if you adjust filtering rules.
   - Execute `datasets/ai_pr/ai_pr.ipynb` and `datasets/human_pr/human_pr.ipynb` to download/cross-check PR metadata, commit histories, comments, reviews, and workflow runs. These notebooks output the parquet bundles under each folder and refresh `performance_prs_ai_vs_human_raw.csv`.
   - Optionally, use `human_pr.ipynb` in the repo root for rapid EDA on the human subset before joining with AI data.
2. **RQ1 – Merge time & language diversity**
   - Open `RQ1_merge_time_and_language/rq1_analysis.ipynb`.
   - Point the intake cell to `datasets/performance_prs_ai_vs_human.csv` (already includes additions/deletions, language, merge timestamps, and metadata).
   - Running the full notebook recreates the summary tables plus `review_time_distribution.png` under the same folder.
3. **RQ2 – Optimization pattern analysis**
   - Choose a labeling notebook (`optimization_pattern_detection_gpt.ipynb` or `optimization_pattern_detection_gemini.ipynb`) to send batched prompts that classify each performance PR into the updated catalog defined in `RQ2_pattern_analysis/catalog/updated_optimization_catalog.csv`.
   - For local models, execute `python RQ2_pattern_analysis/optimization_pattern_detection_qwen.py` (uses Ollama) to replicate the same pipeline without external APIs.
   - Merge the resulting CSVs and evaluate agreement via `python RQ2_pattern_analysis/compare_pattern.py`, which produces Cohen’s kappa scores and mismatch exports under `RQ2_pattern_analysis/`.
   - Figures and aggregate stats are written to `RQ2_pattern_analysis/results/` for downstream storytelling.
4. **RQ3 – Maintainability & complexity impact**
   - Ensure `GITHUB_TOKEN` is set, then run:
     ```bash
     python RQ3_complexity_analysis/ai.py
     python RQ3_complexity_analysis/human.py
     ```
   - The scripts iterate over perf PR URLs, fetch files pre/post change, and call `lizard` to compute `Total nloc`, `Avg.NLOC`, `AvgCCN`, `Fun Cnt`, and `Avg.token`. Results and skip logs save under `RQ3_complexity_analysis/data/`.
   - Use `RQ3_complexity_analysis/analyze_result.ipynb` to load those CSVs and regenerate plots in `RQ3_complexity_analysis/results/` (box plots, CCN distributions, etc.).
5. **RQ4 – Testing & validation evidence**
   - Start with `RQ4_test_and_validation/validation-gpt.ipynb` or `validation-gemini2.ipynb` to summarize the qualitative validation statements extracted from PR descriptions/reviews.
   - `analysis.ipynb` aligns GPT vs Gemini interpretations and exports disagreement lists to `unmatched_data/` for manual inspection.
   - Use the provided parquet files (`rq3_validation_evidence_*.parquet`) if you simply want to replicate figures without rerunning LLMs.

## Outputs & Reuse

- Final, analysis-ready tables: `datasets/performance_prs_ai_vs_human.csv`, `RQ2_pattern_analysis/ai_perf_prs_with_*_analysis_*.csv`, `RQ3_complexity_analysis/data/*deltas.csv`, and `RQ4_test_and_validation/rq3_validation_evidence_*.parquet`.
- Visualizations: PNGs in each RQ folder plus `RQ2_pattern_analysis/results/` summaries (pattern frequency, additions/deletions distributions, etc.).
- Intermediate audit artifacts: mismatch CSVs in `RQ2_pattern_analysis/`, skip logs from `RQ3_complexity_analysis`, and `unmatched_data/` exports from RQ4.

## Troubleshooting

- **Huge parquet pulls** – First-run downloads from Hugging Face can take minutes. Cache them locally or mount a shared volume if running on a remote machine.
- **LLM rate limits / token overflows** – Adjust batch sizes and truncation thresholds inside the RQ2 notebooks/scripts. The helpers already log skipped PRs and allow resumes.
- **GitHub API quotas** – RQ3 scripts use GraphQL/REST calls per file. Create a PAT with the `repo` scope and consider raising the `Github` per-page limits only if you stay under the hourly quota.
- **Encoding issues** – `RQ3_complexity_analysis/ai.py` and `human.py` automatically fall back to UTF-8 with replacement characters, but you can narrow `SUPPORTED_EXTS` if exotic extensions cause failures.

With the structure above you can run any subset of the questions independently or stitch them together for a full refresh of the GitHub Performance Patch Study.
