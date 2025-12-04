import pandas as pd
import anthropic

import time
import json
import os
from datetime import datetime
from pathlib import Path




INPUT_CSV = "drug_studies_with_targets.csv"  
OUTPUT_CSV = "drug_studies_with_approvals.csv"
CHECKPOINT_FILE = "checkpoint2.json"
BATCH_SIZE = 10  # Save checkpoint every N rows
MAX_RETRIES = 3
RETRY_DELAY = 60  # seconds to wait on rate limit
REQUEST_DELAY = 1  # seconds between requests to avoid rate limits

# Model configuration
MODEL = "claude-sonnet-4-5-20250929"  # Supports extended thinking
MAX_TOKENS = 16000  # Required minimum for extended thinking
THINKING_BUDGET = 10000  # Token budget for thinking

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
    return {"last_processed_index": -1, "results": {}}


def save_checkpoint(checkpoint):
    """Save checkpoint to file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def get_approval_with_thinking_and_search(client, drug: str, indication: str, use_web_search: bool = True) -> dict:
    """
    Query Claude API with extended thinking and optional web search.
    
    Returns dict with 'target' and 'thinking' keys.
    """
    prompt = f"""Are all of the following drugs FDA-approved for any of the following indications? Say YES if yes, NO if no, and NULL if you do not know. Ignore any reference to "placebo" and just look at names of drugs. If the indication is "healthy", say YES if the drug is FDA-approved for any indication. Do not add any commentary or explanation, just say YES, NO, or NULL.
    Drug: {drug}
    Indication: {indication}"""

    # Build the request parameters
    request_params = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "thinking": {
            "type": "enabled",
            "budget_tokens": THINKING_BUDGET
        },
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    # Add web search tool if enabled
    # Note: Web search via API requires the appropriate API plan
    # If you don't have access, set use_web_search=False
    if use_web_search:
        request_params["tools"] = [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3  # Limit searches per request
            }
        ]
    
    response = client.messages.create(**request_params)
    
    # Extract thinking and response
    thinking_content = ""
    response_text = ""
    
    for block in response.content:
        if block.type == "thinking":
            thinking_content = block.thinking
        elif block.type == "text":
            response_text = block.text
    
    return {
        "target": response_text.strip(),
        "thinking": thinking_content
    }


def get_target_simple(client, drug: str, indication: str) -> dict:
    """
    Fallback: Query Claude API without extended thinking or web search.
    Use this if you encounter issues with extended thinking.
    """
    prompt = f"""Are all of the following drugs FDA-approved for any of the following indications? Say YES if yes, NO if no, and NULL if you do not know. Ignore any reference to "placebo" and just look at names of drugs. If the indication is "healthy", say YES if the drug is FDA-approved for any indication. Do not add any commentary or explanation, just say YES, NO, or NULL.
    Drug: {drug}
    Indication: {indication}"""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return {
        "target": response.content[0].text.strip(),
        "thinking": None
    }



def process_studies(
    input_csv: str = INPUT_CSV,
    output_csv: str = OUTPUT_CSV,
    use_extended_thinking: bool = True,
    use_web_search: bool = True,
    save_thinking: bool = False,
    start_from: int = None,
    limit: int = None
):
    """
    Process drug studies CSV and add Target column.
    
    Args:
        input_csv: Path to input CSV with drug studies
        output_csv: Path to output CSV with targets added
        use_extended_thinking: Enable extended thinking (requires compatible model)
        use_web_search: Enable web search tool (requires API access)
        save_thinking: Whether to save thinking content to separate file
        start_from: Start processing from this row index (overrides checkpoint)
        limit: Only process this many rows (for testing)
    """
    print(f"[{datetime.now()}] Starting drug target extraction")
    print(f"  Input: {input_csv}")
    print(f"  Output: {output_csv}")
    print(f"  Extended thinking: {use_extended_thinking}")
    print(f"  Web search: {use_web_search}")
    
    # Load data
    print(f"\n[{datetime.now()}] Loading data...")
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    print(f"  Loaded {total_rows} rows")
    
   
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    last_idx = checkpoint["last_processed_index"]
    
    if start_from is not None:
        last_idx = start_from - 1
        print(f"  Starting from row {start_from} (override)")
    elif last_idx >= 0:
        print(f"  Resuming from row {last_idx + 1} (checkpoint)")
    
    # Initialize Target column if not exists
    if "Target" not in df.columns:
        df["Target"] = None
    
    # Apply any cached results from checkpoint
    for idx_str, target in checkpoint["results"].items():
        idx = int(idx_str)
        if idx < len(df):
            df.at[idx, "Target"] = target
    
    # Create client
    client = create_client()
    
    # Determine processing range
    start_idx = last_idx + 1
    end_idx = total_rows
    if limit:
        end_idx = min(start_idx + limit, total_rows)
    
    print(f"\n[{datetime.now()}] Processing rows {start_idx} to {end_idx - 1}")
    
    # Optional: file to save thinking content
    thinking_file = None
    if save_thinking and use_extended_thinking:
        thinking_file = open("thinking_log.jsonl", "a")
    
    # Process rows
    processed = 0
    errors = 0
    
    try:
        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]
            drug = row.get("Interventions", "")
            indication = row.get("Conditions", "")
            
            # Skip if no drug
            if pd.isna(drug) or str(drug).strip() == "":
                df.at[idx, "Target"] = "NULL"
                checkpoint["results"][str(idx)] = "NULL"
                print(f"  [{idx}/{total_rows}] Empty drug -> NULL")
                continue
            
            # Try to get target with retries
            for attempt in range(MAX_RETRIES):
                try:
                    if use_extended_thinking:
                        result = get_target_with_thinking_and_search(
                            client, 
                            str(drug),
                            str(indication), 
                            use_web_search=use_web_search
                        )
                    else:
                        result = get_target_simple(client, str(drug), str(indication))
                    
                    target = result["target"]
                    df.at[idx, "Target"] = target
                    checkpoint["results"][str(idx)] = target
                    
                    # Log thinking if enabled
                    if thinking_file and result.get("thinking"):
                        thinking_file.write(json.dumps({
                            "index": idx,
                            "nct_number": row.get("NCT Number", ""),
                            "target": target,
                            "thinking": result["thinking"]
                        }) + "\n")
                        thinking_file.flush()
                    
                    # Progress output
                    nct = row.get("NCT Number", f"row_{idx}")
                    print(f"  [{idx}/{total_rows}] {nct}: {target[:50]}{'...' if len(target) > 50 else ''}")
                    
                    processed += 1
                    break
                    
                except anthropic.RateLimitError as e:
                    print(f"  [{idx}] Rate limit hit, waiting {RETRY_DELAY}s... (attempt {attempt + 1})")
                    time.sleep(RETRY_DELAY)
                    
                except anthropic.APIError as e:
                    if attempt < MAX_RETRIES - 1:
                        print(f"  [{idx}] API error: {e}, retrying...")
                        time.sleep(5)
                    else:
                        print(f"  [{idx}] API error after {MAX_RETRIES} attempts: {e}")
                        df.at[idx, "Target"] = "ERROR"
                        checkpoint["results"][str(idx)] = "ERROR"
                        errors += 1
                        
                except Exception as e:
                    print(f"  [{idx}] Unexpected error: {e}")
                    df.at[idx, "Target"] = "ERROR"
                    checkpoint["results"][str(idx)] = "ERROR"
                    errors += 1
                    break
            
            # Update checkpoint
            checkpoint["last_processed_index"] = idx
            
            # Save checkpoint periodically
            if (idx + 1) % BATCH_SIZE == 0:
                save_checkpoint(checkpoint)
                df.to_csv(output_csv, index=False)
                print(f"  [Checkpoint saved at row {idx}]")
            
            # Delay between requests
            time.sleep(REQUEST_DELAY)
    
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] Interrupted by user")
    
    finally:
        # Final save
        save_checkpoint(checkpoint)
        df.to_csv(output_csv, index=False)
        
        if thinking_file:
            thinking_file.close()
        
        print(f"\n[{datetime.now()}] Processing complete")
        print(f"  Processed: {processed}")
        print(f"  Errors: {errors}")
        print(f"  Output saved to: {output_csv}")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract drug targets from clinical trial data")
    parser.add_argument("--input", "-i", default=INPUT_CSV, help="Input CSV file")
    parser.add_argument("--output", "-o", default=OUTPUT_CSV, help="Output CSV file")
    parser.add_argument("--test", action="store_true", help="Run a single test query")
    parser.add_argument("--no-thinking", action="store_true", help="Disable extended thinking")
    parser.add_argument("--no-search", action="store_true", help="Disable web search")
    parser.add_argument("--save-thinking", action="store_true", help="Save thinking to log file")
    parser.add_argument("--start", type=int, help="Start from this row index")
    parser.add_argument("--limit", type=int, help="Only process N rows")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, help="Delay between requests (seconds)")
    
    args = parser.parse_args()
    
    if args.delay:
        REQUEST_DELAY = args.delay
    
    if args.test:
        test_single_query()
    else:
        process_studies(
            input_csv=args.input,
            output_csv=args.output,
            use_extended_thinking=not args.no_thinking,
            use_web_search=not args.no_search,
            save_thinking=args.save_thinking,
            start_from=args.start,
            limit=args.limit
        )