# Install dependencies
# !pip install pandas numpy matplotlib seaborn scipy wordcloud pyarrow datasets --quiet

import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from ollama import Client
from pydantic import BaseModel, Field
from scipy import stats
from tqdm import tqdm
import subprocess
import sys


# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# Plot settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Environment ready!")

# Compatibility shim: some versions of fsspec don't expose url_to_fs at top-level.
# This ensures code that expects fsspec.url_to_fs (used by some IO backends) continues to work.
try:
    import fsspec
    if not hasattr(fsspec, "url_to_fs"):
        try:
            from fsspec.core import url_to_fs as _url_to_fs
        except Exception:
            try:
                import fsspec.core as _core
                _url_to_fs = _core.url_to_fs
            except Exception:
                # Fallback shim: create a minimal url_to_fs that returns a filesystem and the path.
                def _url_to_fs(url, **kwargs):
                    protocol = url.split("://")[0] if "://" in url else "file"
                    fs = fsspec.filesystem(protocol)
                    return fs, url
        fsspec.url_to_fs = _url_to_fs
except Exception:
    # If anything goes wrong, continue without failing here; subsequent IO calls will raise their own errors.
    pass

# Load datasets
print("Loading AIDev datasets...")

# AI Agent PRs
pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pull_request.parquet")
pr_task_type_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_task_type.parquet")

ai_perf_prs = (
    pr_df
    .merge(
        pr_task_type_df[["id", "type", "reason"]],
        on="id",
        how="inner"
    )
    .query("type == 'perf'")
    .copy()
)

ai_perf_prs['classification_reason'] = ai_perf_prs['reason']
ai_perf_prs['author_type'] = 'AI Agent'


# Human PRs
human_pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/human_pull_request.parquet")
human_pr_task_type_df = pd.read_parquet("hf://datasets/hao-li/AIDev/human_pr_task_type.parquet")
human_perf_prs = (
    human_pr_df
    .merge(
        human_pr_task_type_df[["id", "type", "reason"]],
        on="id",
        how="inner"
    )
    .query("type == 'perf'")
    .copy()
)
human_perf_prs['classification_reason'] = human_perf_prs['reason']
human_perf_prs['author_type'] = 'Human'
human_perf_prs['agent'] = 'Human'

# Repository data for language info
all_repo_df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_repository.parquet")

# Get list of performance PR IDs we care about
perf_pr_ids = set(ai_perf_prs['id'].tolist() + human_perf_prs['id'].tolist())
print(f"\n✓ Performance PR IDs to process: {len(perf_pr_ids):,}")

# PR commits details - FILTER FIRST, then aggregate
print("\nProcessing commit details (filtering to performance PRs only)...")
pr_commits_details = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commit_details.parquet")

if 'pr_id' in pr_commits_details.columns:
    print(f"  Total commit records in dataset: {len(pr_commits_details):,}")
    
    # FILTER: Keep only commits for performance PRs
    pr_commits_filtered = pr_commits_details[pr_commits_details['pr_id'].isin(perf_pr_ids)].copy()
    print(f"  Filtered to performance PRs: {len(pr_commits_filtered):,} commit records")
    print(f"  Unique performance PRs with commits: {pr_commits_filtered['pr_id'].nunique():,}")
    
    if len(pr_commits_filtered) > 0:
        # AGGREGATE: Now aggregate only the filtered commits
        commit_aggregated = pr_commits_filtered.groupby('pr_id').agg({
            'additions': 'sum',      # Total lines added across all commits
            'deletions': 'sum',      # Total lines deleted across all commits
            'patch': lambda x: '\n\n'.join([str(p) for p in x if pd.notna(p)])  # Concatenate all patches
        }).reset_index()
        
        # Add derived metrics
        commit_aggregated['num_commits'] = pr_commits_filtered.groupby('pr_id').size().values
        
        # Calculate patch length (for analysis)
        commit_aggregated['patch_length'] = commit_aggregated['patch'].str.len()
        
        print(f"  ✓ Aggregated to {len(commit_aggregated):,} unique performance PRs")
        print(f"  Avg commits per PR: {commit_aggregated['num_commits'].mean():.1f}")
        
        # Merge commit stats into AI Agent PR table
        ai_perf_prs = ai_perf_prs.merge(
            commit_aggregated,
            left_on='id',
            right_on='pr_id',
            how='left'
        )
        if 'pr_id' in ai_perf_prs.columns:
            ai_perf_prs = ai_perf_prs.drop(columns=['pr_id'])
        
        ai_with_commits = ai_perf_prs['additions'].notna().sum()
        print(f"  AI Agent PRs with commit data: {ai_with_commits:,} / {len(ai_perf_prs):,} ({ai_with_commits/len(ai_perf_prs)*100:.1f}%)")
        
        # Merge commit stats into Human PR table
        human_perf_prs = human_perf_prs.merge(
            commit_aggregated,
            left_on='id',
            right_on='pr_id',
            how='left'
        )
        if 'pr_id' in human_perf_prs.columns:
            human_perf_prs = human_perf_prs.drop(columns=['pr_id'])
        
        human_with_commits = human_perf_prs['additions'].notna().sum()
        print(f"  Human PRs with commit data: {human_with_commits:,} / {len(human_perf_prs):,} ({human_with_commits/len(human_perf_prs)*100:.1f}%)")
    else:
        print("  ⚠ No commits found for performance PRs")
        # Add placeholder columns
        for df in [ai_perf_prs, human_perf_prs]:
            df['additions'] = None
            df['deletions'] = None
            df['num_commits'] = None
            df['patch'] = None
            df['patch_length'] = None
    
else:
    print('⚠ pr_commit_details missing pr_id column; skipping commit merges.')
    # Add placeholder columns
    for df in [ai_perf_prs, human_perf_prs]:
        df['additions'] = None
        df['deletions'] = None
        df['num_commits'] = None
        df['patch'] = None
        df['patch_length'] = None

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"✓ AI Agent Performance PRs: {len(ai_perf_prs):,}")
print(f"✓ Human Performance PRs: {len(human_perf_prs):,}")
print(f"✓ Total Performance PRs: {len(ai_perf_prs) + len(human_perf_prs):,}")

# Distribution by AI agent
print(f"\nAI Agent Distribution:")
for agent, count in ai_perf_prs['agent'].value_counts().items():
    pct = count / len(ai_perf_prs) * 100
    print(f"  {agent:20s} {count:5,d} ({pct:5.1f}%)")

# Commit statistics summary
if 'num_commits' in ai_perf_prs.columns and ai_perf_prs['num_commits'].notna().any():
    print(f"\n{'='*80}")
    print(f"COMMIT STATISTICS")
    print(f"{'='*80}")
    
    for author_type, df in [('AI Agent', ai_perf_prs), ('Human', human_perf_prs)]:
        with_commits = df['num_commits'].notna()
        if with_commits.sum() > 0:
            print(f"\n{author_type}:")
            print(f"  PRs with commit data: {with_commits.sum():,} ({with_commits.mean()*100:.1f}%)")
            print(f"  Avg commits per PR: {df.loc[with_commits, 'num_commits'].mean():.1f}")
            print(f"  Median commits per PR: {df.loc[with_commits, 'num_commits'].median():.1f}")
            print(f"  Avg additions: {df.loc[with_commits, 'additions'].mean():.0f} lines")
            print(f"  Median additions: {df.loc[with_commits, 'additions'].median():.0f} lines")
            print(f"  Avg deletions: {df.loc[with_commits, 'deletions'].mean():.0f} lines")
            print(f"  Median deletions: {df.loc[with_commits, 'deletions'].median():.0f} lines")

print(f"\n{'='*80}")

# Combine AI and Human PRs
perf_prs = pd.concat([ai_perf_prs, human_perf_prs], ignore_index=True)

print(f"Combined dataset: {len(perf_prs):,} performance PRs")
print(f"  AI Agents: {(perf_prs['author_type'] == 'AI Agent').sum():,}")
print(f"  Humans: {(perf_prs['author_type'] == 'Human').sum():,}")


# ============================================================================
# Optimization taxonomy (used for prompt + validation)
# ============================================================================
TAXONOMY = {
    "Algorithm-Level Optimizations": [
        "Select Computationally Efficient Algorithms",
        "Select Algorithm Based on Instruction Speed",
        "Structure Algorithm to Support instruction level parallelism (ILP)",
        "Select Space Efficient Algorithm",
        "Inheritance over Delegation for Energy Efficiency",
    ],
    "Control-Flow and Branching Optimizations": [
        "Make Conditional Branches More Predictable",
        "Remove Branches with min/max Instructions",
        "Remove Branches by Doing Extra Work",
        "Remove Branching with Masking",
        "Rearranging Branches",
        "Combining Branches",
    ],
    "Memory and Data Locality Optimizations": [
        "Access Data with Appropriate Type (Prevent Store Forwarding Issues)",
        "Increase Cache Efficiency via Locality",
        "Arrange Data for Optimal Hardware Prefetching",
        "Avoid cache capacity issues by segmenting work",
        "Increase Workload to Mitigate Memory Access Latency",
        "Use Smaller Data Types",
        "Caching",
        "Buffering",
        "Improve cache locality via data structure",
        "Optimize Object Use",
        "Reduce memory bloat from RTSJ Immortal Memory",
    ],
    "Loop Transformations": [
        "Remove Conditional by Loop Unrolling",
        "Loop Distribution (Fission)",
        "Loop Fusion",
        "Loop Peeling",
        "Loop Interchanging",
        "Loop Invariant Branches",
        "Loop Strip-mining",
    ],
    "I/O and Synchronization": [
        "Selection of I/O Size",
        "Polling",
        "Non-Blocking I/O",
    ],
    "Data Structure Selection and Adaptation": [
        "Choose Structure for Energy Efficiency",
        "Darwinian Data Structure Selection",
        "Choose more energy-efficient data structure across Java Collections Framework, Apache Common Collections, and Eclipse Collections",
        "Choose energy-efficient data structure by method calls",
    ],
    "Code Smells and Structural Simplification": [
        "Remove code bloat by removing optional features",
        "Remove Unnecessary Method Calls",
        "Remove long method by extracting new method",
        "Remove Duplicate code",
        "Minimize feature envy by moving methods",
        "Minimize occurrences of God Class",
        "Type Checking",
    ],
}

HIGH_LEVEL_PATTERNS = list(TAXONOMY.keys()) + ["No Meaningful Change"]
ALL_SUB_PATTERNS = sorted({p for patterns in TAXONOMY.values() for p in patterns})
ATTEMPT_LOG_FILE = Path("pattern_attempts.log")


def _validate_taxonomy_selection(high_level_pattern: str, sub_pattern: Optional[str]) -> None:
    """
    Ensure the model output sticks to the taxonomy. Raises ValueError when invalid.
    """
    if high_level_pattern not in HIGH_LEVEL_PATTERNS:
        raise ValueError(f"Invalid high_level_pattern: {high_level_pattern!r}")

    if high_level_pattern == "No Meaningful Change":
        if sub_pattern not in (None, "", "No Meaningful Change"):
            raise ValueError("sub_pattern must be null/empty when high_level_pattern is 'No Meaningful Change'")
        return

    allowed_subpatterns = TAXONOMY[high_level_pattern]
    if sub_pattern not in allowed_subpatterns:
        raise ValueError(
            f"Invalid sub_pattern {sub_pattern!r} for high_level_pattern {high_level_pattern!r}. "
            f"Allowed: {allowed_subpatterns}"
        )


def _append_attempt_log(pr_id, attempts, high_level_pattern, sub_pattern, success, error=None):
    """
    Append a single log entry describing how many attempts were needed for taxonomy compliance.
    """
    timestamp = datetime.utcnow().isoformat()
    entry = {
        "timestamp": timestamp,
        "pr_id": pr_id,
        "attempts": attempts,
        "success": success,
        "high_level_pattern": high_level_pattern,
        "sub_pattern": sub_pattern,
        "error": error,
    }
    try:
        with ATTEMPT_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # Logging failure should not break processing
        pass


# ============================================================================
# Performance Optimization Pattern Detection with Ollama
# ============================================================================
def analyze_optimization_with_ollama(title, body, patch):
    """
    Call the local Ollama model (qwen3-coder:480b) to analyze performance optimization patterns in a commit.
    
    Parameters:
    - title: PR/commit title
    - body: PR/commit description
    - patch: Git diff/patch content
    
    Returns:
    - dict with analysis results or error info
    """
    
    # Prepare the context
    context_parts = []
    
    if pd.notna(title) and str(title).strip():
        context_parts.append(f"**Title**: {title}")
    
    if pd.notna(body) and str(body).strip():
        context_parts.append(f"**Description**: {body}")
    
    if pd.notna(patch) and str(patch).strip():
        # Truncate very long patches to avoid token limits
        patch_str = str(patch)
        if len(patch_str) > 15000:  # Rough character limit
            patch_str = patch_str[:15000] + "\n\n... [patch truncated for length] ..."
        context_parts.append(f"**Code Changes (Patch)**:\n```diff\n{patch_str}\n```")
    
    if not context_parts:
        return {
            "success": False,
            "error": "No content available",
            "explanation": None,
            "optimization_comparison": None,
            "high_level_pattern": None,
            "sub_pattern": None,
            "tokens_used": 0
        }
    
    context = "\n\n".join(context_parts)
    
    try:
        load_dotenv()
    except Exception:
        pass

    # Construct the prompt
    prompt = f"""I have a performance optimization commit with the following information. Please analyze with the following goals:

1. **Code Function Explanation**: Briefly explain what the code is doing—what problem it solves and how it works.

2. **Optimization Comparison**: Compare the original and optimized versions to identify:
   - **Algorithmic changes**: Any differences in logic, algorithm design, or problem-solving approach.
   - **Performance improvements**: Enhancements related to time complexity, space efficiency, or runtime behavior.
   - **Redundant code removal**: Elimination of unnecessary logic, method calls, or control structures.
   - **Other noteworthy changes**: Any structural or stylistic differences that could impact performance or readability.
   
3. **Optimization Pattern Classification**:
   Based on the overall nature of the optimized code, assign the following. Return "No Meaningful Change" if no meaningful change is made.
   - **Exactly one high-level optimization pattern** from the list below  
   - **One most representative sub-pattern** within that high-level category
   - **Pattern and Sub-pattern must exactly match the taxonomy below.**
   
   ### High-Level Optimization Patterns Taxonomy:
   - **Algorithm-Level Optimizations**
        - Select Computationally Efficient Algorithms
        - Select Algorithm Based on Instruction Speed
        - Structure Algorithm to Support instruction level parallelism (ILP)
        - Select Space Efficient Algorithm
        - Inheritance over Delegation for Energy Efficiency
   - **Control-Flow and Branching Optimizations**
        - Make Conditional Branches More Predictable
        - Remove Branches with min/max Instructions
        - Remove Branches by Doing Extra Work
        - Remove Branching with Masking
        - Rearranging Branches
        - Combining Branches
   - **Memory and Data Locality Optimizations**
        - Access Data with Appropriate Type (Prevent Store Forwarding Issues)
        - Increase Cache Efficiency via Locality
        - Arrange Data for Optimal Hardware Prefetching
        - Avoid cache capacity issues by segmenting work
        - Increase Workload to Mitigate Memory Access Latency
        - Use Smaller Data Types
        - Caching
        - Buffering
        - Improve cache locality via data structure
        - Optimize Object Use
        - Reduce memory bloat from RTSJ Immortal Memory
   - **Loop Transformations**
        - Remove Conditional by Loop Unrolling
        - Loop Distribution (Fission)
        - Loop Fusion
        - Loop Peeling
        - Loop Interchanging
        - Loop Invariant Branches
        - Loop Strip-mining
   - **I/O and Synchronization**
        - Selection of I/O Size
        - Polling
        - Non-Blocking I/O
   - **Data Structure Selection and Adaptation**
        - Choose Structure for Energy Efficiency
        - Darwinian Data Structure Selection
        - Choose more energy-efficient data structure across Java Collections Framework, Apache Common Collections, and Eclipse Collections
        - Choose energy-efficient data structure by method calls
   - **Code Smells and Structural Simplification**
        - Remove code bloat by removing optional features
        - Remove Unnecessary Method Calls
        - Remove long method by extracting new method
        - Remove Duplicate code
        - Minimize feature envy by moving methods
        - Minimize occurrences of God Class
        - Type Checking
         
Here are the info:
            
{context}

**Output Structure**:  
Please respond in JSON format with the following structure:
{{
  "explanation": "Brief description of what the code is doing",
  "optimization_comparison": "Detailed comparison highlighting specific optimizations",
  "high_level_pattern": "Single most representative high-level optimization pattern **from the provided 7 high-level Taxonomy** (or 'No Meaningful Change')",
  "sub_pattern": "Most representative sub-pattern within the high_level_pattern **from the provided Taxonomy** (or null if No Meaningful Change)",
}}

Ensure your response is valid JSON that can be parsed. Strictly only return patterns from the provided taxonomy above.
"""

    tokens_total = 0
    attempt_count = 0

    try:
        try:
            load_dotenv()
        except Exception:
            pass

        ollama_model = os.getenv("OLLAMA_MODEL", "qwen3-coder:480b")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11402")

        client = Client(host=ollama_host)
        max_attempts = 3
                
        class AnalysisResult(BaseModel):
            explanation: str
            optimization_comparison: str
            high_level_pattern: str = Field(
                description="High-level optimization pattern from taxonomy or 'No Meaningful Change'",
                json_schema_extra={"enum": HIGH_LEVEL_PATTERNS},
            )
            sub_pattern: Optional[str] = Field(
                default=None,
                description="Sub-pattern from taxonomy (null when 'No Meaningful Change')",
                json_schema_extra={"enum": ALL_SUB_PATTERNS + ["No Meaningful Change", None]},
            )

        schema = AnalysisResult.model_json_schema()
        messages = [
            {
                "role": "system",
                "content": "You are an expert software engineer specializing in performance optimization analysis. Analyze code changes and classify optimization patterns accurately."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        last_error = None
        for attempt in range(max_attempts):
            attempt_count = attempt + 1
            response = client.chat(
                model=ollama_model,
                messages=messages,
                format=schema,
                options={"temperature": 0}
            )

            prompt_tokens = response.get("prompt_eval_count") or 0
            completion_tokens = response.get("eval_count") or 0
            tokens_total += prompt_tokens + completion_tokens

            message = response.get("message") or {}
            content = message.get("content")
            if not content:
                raise RuntimeError("Ollama returned an empty response.")

            result = json.loads(content)
            try:
                validated = AnalysisResult(**result)
                _validate_taxonomy_selection(validated.high_level_pattern, validated.sub_pattern)
                return {
                    "success": True,
                    "explanation": validated.explanation,
                    "optimization_comparison": validated.optimization_comparison,
                    "high_level_pattern": validated.high_level_pattern,
                    "sub_pattern": validated.sub_pattern,
                    "tokens_used": tokens_total,
                    "error": None,
                    "attempts": attempt_count,
                }
            except ValueError as exc:
                last_error = str(exc)
                if attempt < max_attempts - 1:
                    messages.append({
                        "role": "user",
                        "content": f"Your previous output was invalid ({last_error}). Respond again using only the taxonomy values above. Return JSON only."
                    })
                    continue
                fallback_error = f"Validation failed after {max_attempts} attempts: {last_error}"
                return {
                    "success": True,
                    "explanation": None,
                    "optimization_comparison": None,
                    "high_level_pattern": "Not valid Pattern",
                    "sub_pattern": "Not valid Pattern",
                    "tokens_used": tokens_total,
                    "error": fallback_error,
                    "attempts": attempt_count,
                }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "explanation": None,
            "optimization_comparison": None,
            "high_level_pattern": None,
            "sub_pattern": None,
            "tokens_used": tokens_total,
            "attempts": attempt_count,
        }


def batch_analyze_performance_prs(perf_prs, batch_size=10, delay=1.0,resume=False, checkpoint_prefix='perf_prs_checkpoint', output_file='perf_prs_with_gpt_analysis.csv'):
    """
    Analyze all performance PRs in batches.

    Parameters:
    - perf_prs: DataFrame with performance PRs
    - batch_size: Number of PRs to process before saving checkpoint
    - delay: Delay between API calls in seconds
    - resume: Continue from the last available checkpoint if True
    - checkpoint_prefix: Filename prefix used for checkpoint files
    - output_file: Final CSV filename for the aggregated results

    Returns:
    - DataFrame with analysis results added
    """

    print(f"Starting Ollama analysis of {len(perf_prs):,} performance PRs...")

    checkpoint_files = []
    processed_count = 0

    if resume:
        checkpoint_files = sorted(Path('.').glob(f"{checkpoint_prefix}_*.csv"))
        if checkpoint_files:
            def _processed_from_path(path_obj):
                suffix = path_obj.stem.rsplit('_', 1)[-1]
                return int(suffix) if suffix.isdigit() else 0

            latest_checkpoint = max(checkpoint_files, key=_processed_from_path)
            checkpoint_progress = _processed_from_path(latest_checkpoint)
            perf_prs = pd.read_csv(latest_checkpoint)
            processed_count = min(checkpoint_progress, len(perf_prs))
            print(f"↻ Resuming from checkpoint {latest_checkpoint} ({processed_count} PRs processed)...")
        else:
            print("↻ Resume requested but no checkpoint found. Starting from scratch.")

    result_defaults = {
        'model_explanation': None,
        'model_comparison': None,
        'optimization_pattern': None,
        'optimization_subpattern': None,
        'model_success': False,
        'model_error': None,
        'model_tokens': 0,
        'model_attempts': 0,
    }

    for column, default in result_defaults.items():
        if resume and column in perf_prs.columns:
            continue
        perf_prs[column] = default

    start_idx = processed_count if resume else 0
    iterator = range(start_idx, len(perf_prs))
    progress_bar = tqdm(iterator, total=len(perf_prs), desc="Analyzing PRs", initial=start_idx)

    for idx in progress_bar:
        row = perf_prs.iloc[idx]
        result = analyze_optimization_with_ollama(
            title=row.get('title'),
            body=row.get('body'),
            patch=row.get('patch')
        )

        perf_prs.at[idx, 'model_success'] = result['success']
        perf_prs.at[idx, 'model_tokens'] = result['tokens_used']
        perf_prs.at[idx, 'model_attempts'] = result.get('attempts', 0)

        if result['success']:
            perf_prs.at[idx, 'model_explanation'] = result['explanation']
            perf_prs.at[idx, 'model_comparison'] = result['optimization_comparison']
            perf_prs.at[idx, 'optimization_pattern'] = result['high_level_pattern']
            perf_prs.at[idx, 'optimization_subpattern'] = result['sub_pattern']
            perf_prs.at[idx, 'model_error'] = result.get('error')
        else:
            perf_prs.at[idx, 'model_error'] = result['error']

        pr_id = row.get('id', idx)
        _append_attempt_log(
            pr_id=pr_id,
            attempts=result.get('attempts', 0),
            high_level_pattern=result.get('high_level_pattern'),
            sub_pattern=result.get('sub_pattern'),
            success=result['success'],
            error=result.get('error'),
        )

        time.sleep(delay)

        if (idx + 1) % batch_size == 0:
            checkpoint_file = f"{checkpoint_prefix}_{idx+1}.csv"
            perf_prs.to_csv(checkpoint_file, index=False)
            print(f"✓ Checkpoint saved: {checkpoint_file}")

    perf_prs.to_csv(output_file, index=False)
    print(f"✓ Analysis complete! Saved to: {output_file}")

    success_series = perf_prs['model_success'].fillna(False)
    success_count = success_series.sum()
    success_rate = (success_count / len(perf_prs) * 100) if len(perf_prs) else 0
    failure_count = success_series.eq(False).sum()
    total_tokens = perf_prs['model_tokens'].sum()

    print(f"{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total PRs analyzed: {len(perf_prs):,}")
    print(f"Successful: {success_count:,} ({success_rate:.1f}%)")
    print(f"Failed: {failure_count:,}")
    print(f"Total tokens used: {total_tokens:,}")

    if success_count > 0:
        print(f"{'='*80}")
        print("OPTIMIZATION PATTERN DISTRIBUTION")
        print(f"{'='*80}")
        pattern_counts = perf_prs[perf_prs['model_success'] == True]['optimization_pattern'].value_counts()
        for pattern, count in pattern_counts.items():
            pct = count / success_count * 100
            print(f"  {pattern:50s} {count:4d} ({pct:5.1f}%)")

    return perf_prs


# ============================================================================
# Usage
# ============================================================================

# run ai and human pr analysis separately

# ai pr analysis
ai_sample = perf_prs[perf_prs['author_type'] == 'AI Agent']
print(f"Testing Ollama analysis on {len(ai_sample)} AI PRs")

# Run the analysis
perf_prs_analyzed = batch_analyze_performance_prs(
    ai_sample,
    batch_size=20,    # Save checkpoint every 10 PRs
    delay=0.5,        # 0.5 second delay between API calls
    resume=True,      # Continue from the last saved checkpoint if available
    checkpoint_prefix='ai_perf_prs_checkpoint',
    output_file='ai_perf_prs_with_qwen_analysis.csv'
)


# # human pr analysis
# human_sample = perf_prs[perf_prs['author_type'] == 'Human']
# print(f"Testing Ollama analysis on {len(human_sample)} Human PRs")

# # Run the analysis
# perf_prs_analyzed = batch_analyze_performance_prs(
#     human_sample,
#     batch_size=10,    # Save checkpoint every 10 PRs
#     delay=0.5,        # 0.5 second delay between API calls
#     resume=True,      # Continue from the last saved checkpoint if available
#     checkpoint_prefix='human_perf_prs_checkpoint',
#     output_file='human_perf_prs_with_gpt_analysis.csv'
# )
