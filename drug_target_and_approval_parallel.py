import pandas as pd
import anthropic
import time
import json
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
from dataclasses import dataclass
from typing import Optional, Tuple
import queue

# ============ CONFIGURATION ============
INPUT_CSV = "drug_studies.csv"
OUTPUT_CSV = "drug_studies_with_targets_and_approvals.csv"
CHECKPOINT_FILE = "checkpoint.json"

# Parallelization settings
NUM_WORKERS = 10  # Number of concurrent API calls (adjust based on your rate limits)
MAX_REQUESTS_PER_MINUTE = 50  # Anthropic rate limit (adjust to your tier)

BATCH_SIZE = 50  # Save checkpoint every N completed rows
MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds to wait on rate limit

# Model configuration - using simple mode for speed/cost
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 1024

# ============ THREAD-SAFE STATE ============
checkpoint_lock = Lock()
results_lock = Lock()
rate_limiter = Semaphore(NUM_WORKERS)


@dataclass
class ProcessingResult:
    """Result from processing a single row."""
    index: int
    target: str
    approval: str
    nct_number: str
    success: bool
    error: Optional[str] = None


def create_client():
    """Create Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set.\n"
            "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
        )
    return anthropic.Anthropic(api_key=api_key)


def load_checkpoint():
    """Load checkpoint if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"completed_indices": [], "results": {}}


def save_checkpoint(checkpoint):
    """Save checkpoint to file (thread-safe)."""
    with checkpoint_lock:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f)


def get_target_and_approval_combined(client, drug: str, indication: str) -> dict:
    """
    Query Claude API for BOTH target and approval in a single call.
    This halves your API requests.
    """
    prompt = f"""Analyze this drug and indication, then provide two pieces of information:

Drug: {drug}
Indication: {indication}

1. TARGET: What is the molecular/genetic target of this drug (its mechanism of action)? 
   - If identifiable, give the target name (e.g., "COX-2", "IL-6", "EGFR")
   - If unknown or not applicable, say "NULL"

2. FDA_APPROVED: Is this drug, or are all these drugs, FDA-approved for this indication (or any indication if indication is "healthy")?
   - Answer YES, NO, or NULL (if unknown)
   - Ignore any mention of "placebo"

Respond in EXACTLY this format with no other text:
TARGET: [target name or NULL]
FDA_APPROVED: [YES/NO/NULL]"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}]
    )
    
    text = response.content[0].text.strip()
    
    # Parse the response
    target = "NULL"
    approval = "NULL"
    
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('TARGET:'):
            target = line.replace('TARGET:', '').strip()
        elif line.startswith('FDA_APPROVED:'):
            approval = line.replace('FDA_APPROVED:', '').strip()
    
    return {"target": target, "approval": approval}


def process_single_row(client, idx: int, drug: str, indication: str, nct_number: str) -> ProcessingResult:
    """
    Process a single row with retries. Called by worker threads.
    """
    if pd.isna(drug) or str(drug).strip() == "":
        return ProcessingResult(
            index=idx,
            target="NULL",
            approval="NULL", 
            nct_number=nct_number,
            success=True
        )
    
    for attempt in range(MAX_RETRIES):
        try:
            with rate_limiter:  # Limit concurrent requests
                result = get_target_and_approval_combined(
                    client,
                    str(drug),
                    str(indication)
                )
                return ProcessingResult(
                    index=idx,
                    target=result["target"],
                    approval=result["approval"],
                    nct_number=nct_number,
                    success=True
                )
                
        except anthropic.RateLimitError as e:
            wait_time = RETRY_DELAY * (attempt + 1)  # Exponential backoff
            print(f"  [{idx}] Rate limit hit, waiting {wait_time}s... (attempt {attempt + 1})")
            time.sleep(wait_time)
            
        except anthropic.APIError as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [{idx}] API error: {e}, retrying...")
                time.sleep(5)
            else:
                return ProcessingResult(
                    index=idx,
                    target="ERROR",
                    approval="ERROR",
                    nct_number=nct_number,
                    success=False,
                    error=str(e)
                )
                
        except Exception as e:
            return ProcessingResult(
                index=idx,
                target="ERROR",
                approval="ERROR",
                nct_number=nct_number,
                success=False,
                error=str(e)
            )
    
    return ProcessingResult(
        index=idx,
        target="ERROR",
        approval="ERROR",
        nct_number=nct_number,
        success=False,
        error="Max retries exceeded"
    )


def process_studies_parallel(
    input_csv: str = INPUT_CSV,
    output_csv: str = OUTPUT_CSV,
    num_workers: int = NUM_WORKERS,
    start_from: int = None,
    limit: int = None
):
    """
    Process drug studies CSV with parallel API calls.
    """
    print(f"[{datetime.now()}] Starting PARALLEL drug target and approval extraction")
    print(f"  Input: {input_csv}")
    print(f"  Output: {output_csv}")
    print(f"  Workers: {num_workers}")
    
    # Load data
    print(f"\n[{datetime.now()}] Loading data...")
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    print(f"  Loaded {total_rows} rows")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed_set = set(checkpoint.get("completed_indices", []))
    print(f"  Already completed: {len(completed_set)} rows")
    
    # Initialize columns if not exist
    if "Target" not in df.columns:
        df["Target"] = None
    if "Approval" not in df.columns:
        df["Approval"] = None
    
    # Apply cached results from checkpoint
    for idx_str, result in checkpoint.get("results", {}).items():
        idx = int(idx_str)
        if idx < len(df):
            if isinstance(result, (list, tuple)):
                df.at[idx, "Target"] = result[0]
                df.at[idx, "Approval"] = result[1]
            else:
                df.at[idx, "Target"] = result
    
    # Determine rows to process
    start_idx = start_from if start_from is not None else 0
    end_idx = min(start_idx + limit, total_rows) if limit else total_rows
    
    # Filter out already completed rows
    indices_to_process = [
        i for i in range(start_idx, end_idx) 
        if i not in completed_set
    ]
    
    print(f"  Rows to process: {len(indices_to_process)}")
    
    if not indices_to_process:
        print("  Nothing to process!")
        return
    
    # Create client (shared across threads - Anthropic client is thread-safe)
    client = create_client()
    
    # Progress tracking
    completed_count = 0
    error_count = 0
    start_time = time.time()
    
    # Process in parallel
    print(f"\n[{datetime.now()}] Processing with {num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_idx = {}
        for idx in indices_to_process:
            row = df.iloc[idx]
            future = executor.submit(
                process_single_row,
                client,
                idx,
                row.get("Interventions", ""),
                row.get("Conditions", ""),
                row.get("NCT Number", f"row_{idx}")
            )
            future_to_idx[future] = idx
        
        # Collect results as they complete
        pending_results = []
        
        for future in as_completed(future_to_idx):
            try:
                result = future.result()
                
                # Update DataFrame (thread-safe with lock)
                with results_lock:
                    df.at[result.index, "Target"] = result.target
                    df.at[result.index, "Approval"] = result.approval
                    
                    # Update checkpoint data
                    checkpoint["results"][str(result.index)] = (result.target, result.approval)
                    completed_set.add(result.index)
                    checkpoint["completed_indices"] = list(completed_set)
                
                pending_results.append(result)
                
                if result.success:
                    completed_count += 1
                else:
                    error_count += 1
                
                # Progress output
                elapsed = time.time() - start_time
                rate = completed_count / elapsed if elapsed > 0 else 0
                remaining = len(indices_to_process) - completed_count - error_count
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_hours = eta_seconds / 3600
                
                print(f"  [{completed_count + error_count}/{len(indices_to_process)}] "
                      f"{result.nct_number}: {result.target[:30]}{'...' if len(result.target) > 30 else ''} | {result.approval} "
                      f"({rate:.1f}/s, ETA: {eta_hours:.1f}h)")
                
                # Save checkpoint periodically
                if len(pending_results) >= BATCH_SIZE:
                    save_checkpoint(checkpoint)
                    df.to_csv(output_csv, index=False)
                    print(f"  [Checkpoint saved: {completed_count} completed, {error_count} errors]")
                    pending_results = []
                    
            except Exception as e:
                print(f"  [ERROR] Future failed: {e}")
                error_count += 1
    
    # Final save
    save_checkpoint(checkpoint)
    df.to_csv(output_csv, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n[{datetime.now()}] Processing complete")
    print(f"  Completed: {completed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"  Rate: {completed_count/elapsed:.2f} rows/second")
    print(f"  Output saved to: {output_csv}")


def estimate_time(total_rows: int, num_workers: int = NUM_WORKERS):
    """Estimate processing time."""
    # Assume ~1-2 seconds per API call with simple mode
    seconds_per_call = 1.5
    parallel_factor = num_workers
    
    total_seconds = (total_rows * seconds_per_call) / parallel_factor
    hours = total_seconds / 3600
    days = hours / 24
    
    print(f"Estimated time for {total_rows:,} rows with {num_workers} workers:")
    print(f"  ~{hours:.1f} hours ({days:.1f} days)")
    print(f"\nAdjust NUM_WORKERS based on your API rate limits.")
    print(f"Higher tiers can use 20-50+ workers for faster processing.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract drug targets and FDA approval (PARALLEL VERSION)"
    )
    parser.add_argument("--input", "-i", default=INPUT_CSV, help="Input CSV file")
    parser.add_argument("--output", "-o", default=OUTPUT_CSV, help="Output CSV file")
    parser.add_argument("--workers", "-w", type=int, default=NUM_WORKERS,
                        help=f"Number of parallel workers (default: {NUM_WORKERS})")
    parser.add_argument("--start", type=int, help="Start from this row index")
    parser.add_argument("--limit", type=int, help="Only process N rows")
    parser.add_argument("--estimate", action="store_true", 
                        help="Estimate processing time without running")
    
    args = parser.parse_args()
    
    if args.estimate:
        # Quick row count for estimation
        df = pd.read_csv(args.input)
        estimate_time(len(df), args.workers)
    else:
        process_studies_parallel(
            input_csv=args.input,
            output_csv=args.output,
            num_workers=args.workers,
            start_from=args.start,
            limit=args.limit
        )
