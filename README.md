# GitHub Performance Patch Study

Analysis of performance-focused pull requests (PRs) raised by AI coding agents versus human developers. The project currently lives in two notebooks:

- `optimization_pattern_detection.ipynb` maps each optimization-focused PR to a standardized optimization pattern catalog with GPT.
- `Performance_PR_Analysis_AI_vs_Human.ipynb` is the main exploratory data analysis (EDA) workbook covering the research questions that compare AI and human behavior.

## Data Sources

Both notebooks expect direct access to the Hugging Face dataset `hao-li/AIDev` via pandas’ `hf://` protocol:

- `pull_request.parquet` / `human_pull_request.parquet`
- `pr_task_type.parquet` / `human_pr_task_type.parquet`
- `pr_commit_details.parquet`
- `all_repository.parquet`

Make sure you are authenticated with Hugging Face (`huggingface-cli login`) or have the files cached locally, otherwise pandas will not be able to stream them.

## Notebook 1: Optimization Pattern Detection

This notebook builds the labeled dataset that downstream analysis relies on.

1. **Load & stitch datasets.** AI-agent and human PRs are read from the Hugging Face parquet dumps, filtered to `type == 'perf'`, and enriched with repository metadata and aggregated commit stats (additions, deletions, concatenated patches, commit counts).
2. **Call GPT with taxonomy prompts.** The function `analyze_optimization_with_gpt` composes a context from PR title, description, and git patches (with automatic truncation for long patches). GPT is asked to:
   - explain the optimization,
   - contrast original vs. optimized code,
   - classify the change into a high-level optimization pattern **and** a representative sub-pattern chosen from the catalog below.
3. **Batching + resiliency.**
   - `batch_analyze_performance_prs` processes PRs in configurable batches, throttles requests, tracks token usage, and writes rolling checkpoints (`perf_prs_checkpoint_*.parquet`) plus a master CSV (`perf_prs_with_gpt_analysis.csv`).
   - Resume mode skips PRs already seen in either the checkpoint folder or the consolidated CSV, so interrupted runs can restart safely.
4. **Outputs.** Besides the CSV, the notebook prints counts by author type, confidence scores, and pattern distribution samples for quick validation.

### Optimization Pattern Catalog (High-Level Buckets)

- Algorithm-Level Optimizations (e.g., pick more efficient algorithms, ILP-friendly structures, space-efficient implementations).
- Control-Flow & Branching (branch prediction, removal/rearrangement, masking, combining branches).
- Memory & Data Locality (cache locality, prefetching, smaller data types, object/structure tuning).
- Loop Transformations (unrolling, fusion/fission, peeling, stripping, invariant extraction).
- I/O & Synchronization (I/O batch sizing, polling vs. blocking, non-blocking primitives).
- Data Structure Selection & Adaptation (energy-aware structures, cross-library comparisons, method-call based choices).
- Code Smells & Structural Simplification (pruning optional features, redundant calls, long methods, duplicates, feature envy, god classes, type checking).
- No Meaningful Change (fallback when patches cannot be categorized).

Use this notebook first—`perf_prs_with_gpt_analysis.csv` is an input to the main analysis.

## Notebook 2: Performance PR Analysis (AI vs Human)

The second notebook answers the research questions with descriptive stats and visualizations.

1. **Data preparation.**
   - Reuses the same PR ingestion logic, joins repository language data, converts timestamps, and derives helper columns (`has_body`, `time_to_merge_hours`, `body_length`, etc.).
   - Imports the GPT-enriched CSV to add `high_level_pattern`, `sub_pattern`, and GPT confidence to each PR.
2. **Feature engineering & scoring.**
   - Functions such as `assess_description_quality` score PR descriptions on a 0–5 scale.
   - Categorical bins for PR size, language, and success outcomes are created for group-by comparison.
3. **Research questions covered by dedicated sections:**
   - **RQ1 Adoption & Practices:** PR size vs. merge rate, description quality, programming language mix.
   - **RQ2 Optimization Patch Characteristics:** Distribution of pattern families, additions/deletions, commit volume.
   - **RQ3 Testing & Evaluation (WIP):** Stubbed; hooks exist for future metrics.
   - **RQ4 Review Dynamics:** Merge/review latency distributions, state transitions.
   - **RQ5 Failure Patterns:** Characteristics of closed-but-unmerged PRs and qualitative failure reasons.
4. **Visualization.** Uses Matplotlib/Seaborn to render bar charts, KDEs, stacked distributions, and pattern comparison plots by author type.
5. **Exports.** Final aggregated tables and the enriched PR dataset can be written back out for slide decks or further modeling.

> ⚠️ GPT generation cells inside this notebook are marked “DO NOT rerun” to avoid duplicate spend. Use the outputs from the pattern-detection notebook unless you intentionally want to regenerate labels.

## How to Reproduce the Analysis

1. Run `optimization_pattern_detection.ipynb` end-to-end (or rerun only the batching section with `resume=True`) until `perf_prs_with_gpt_analysis.csv` is produced.
2. Open `Performance_PR_Analysis_AI_vs_Human.ipynb`, run the setup + data prep cells, and then execute individual research-question sections as needed.
3. Inspect or export the figures/statistics that align with your study goals.

## Troubleshooting & Tips

- **Dataset scale:** The Hugging Face parquet files are sizable; prefer running locally with adequate memory or mirror the data to object storage close to your compute.
- **API rate limits:** Adjust `batch_size`, `delay`, or provide Azure/OpenAI-style rate limit configs if you hit throttling.
- **Long patches:** The helper automatically trims patch payloads, but you can tighten the `patch_str` cap if you see token limit errors.
- **Version control:** Because notebooks produce large intermediate artifacts, add the CSV/plots you care about and ignore checkpoints in Git as needed.

Feel free to adapt the notebooks into Python scripts if you plan to schedule recurring refreshes or integrate with other analytics tooling.
