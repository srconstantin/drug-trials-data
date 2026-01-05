import re
import pandas as pd
import os
import anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

target_results = pd.read_csv('target_results.csv')

successful = target_results[(target_results['Success'] == 'Success')]
successfulpairs = successful[['Target', 'Conditions']].assign(Conditions=successful['Conditions'].str.split('|')).explode('Conditions').drop_duplicates()
successfulsick = successfulpairs[~successfulpairs['Conditions'].str.contains('Healthy', na=False)]

unsuccessful = target_results[(target_results['Success'] == 'No Success')]
unsuccessfulpairs = unsuccessful[['Target', 'Conditions']].assign(Conditions=unsuccessful['Conditions'].str.split('|')).explode('Conditions').drop_duplicates()
unsuccessfulsick = unsuccessfulpairs[~unsuccessfulpairs['Conditions'].str.contains('Healthy', na=False)]




def search_target_validation(target, condition):
    """
    Search for literature supporting target-condition associations.
    Returns tuple of (genetic, animal_gene, animal_drug, in_vitro) responses.
    """
    client = anthropic.Anthropic()
    
    prompts = [
        f"Search for journal articles that find an association between {target} genetics or expression and {condition} in human subjects.",
        f"Search for journal articles that find an effect of {target} on animal models of {condition}, including knockouts, knockdowns, mutant strains, and other genetic modifications.",
        f"Search for journal articles that find an effect of pharmacologically targeting {target} on animal models of {condition}.",
        f"Search for journal articles that find involvement of {target} in {condition} in vitro."
    ]
    
    format_instruction = "Return ONLY a |-separated list of citations with no other text. If no articles are found, return nothing (blank response)."
    
    responses = []
    for prompt in prompts:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10000,
 
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search"
                }
            ],
            messages=[
                {"role": "user", "content": f"{prompt} {format_instruction}"}
            ]
        )
        
        # Extract text response (skip thinking blocks)
        response_text = ""
        for block in message.content:
            if block.type == "text":
                response_text = block.text.strip()
                break
        
        responses.append(response_text)
    
    return tuple(responses)

def countpapers(geneticresponse, animalgeneresponse, animaldrugresponse, invitroresponse):
	genecount = len(geneticresponse.split('|')) if geneticresponse else 0
	animalgenecount = len(animalgeneresponse.split('|')) if animalgeneresponse else 0
	animaldrugcount = len(animaldrugresponse.split('|')) if animaldrugresponse else 0
	invitrocount = len(invitroresponse.split('|')) if invitroresponse else 0
	return (genecount, animalgenecount, animaldrugcount, invitrocount)


# --- Processing loop with parallel execution and checkpointing ---

CHECKPOINT_FILE = 'papercounts_checkpoint.csv'
OUTPUT_FILE = 'papercounts.csv'
MAX_WORKERS = 1  # Adjust based on API rate limits

print_lock = Lock()
file_lock = Lock()

def load_checkpoint():
    """Load previously processed pairs from checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        df = pd.read_csv(CHECKPOINT_FILE)
        processed = set(zip(df['Target'], df['Condition']))
        print(f"Loaded {len(processed)} previously processed pairs from checkpoint.")
        return df, processed
    else:
        df = pd.DataFrame(columns=['Target', 'Condition', 'GeneticCount', 'AnimalGeneCount', 'AnimalDrugCount', 'InVitroCount'])
        return df, set()


def save_checkpoint(df):
    """Save current results to checkpoint file."""
    with file_lock:
        df.to_csv(CHECKPOINT_FILE, index=False)

def process_pair(target, condition):
    """Process a single target-condition pair."""
    try:
        # Run the search
        responses = search_target_validation(target, condition)
        
        # Count papers
        counts = countpapers(*responses)
        
        # Print results
        with print_lock:
            print(f"\n{'='*60}")
            print(f"Target: {target} | Condition: {condition}")
            print(f"Responses:")
            print(f"  Genetic: {responses[0][:100]}..." if len(responses[0]) > 100 else f"  Genetic: {responses[0]}")
            print(f"  Animal Gene: {responses[1][:100]}..." if len(responses[1]) > 100 else f"  Animal Gene: {responses[1]}")
            print(f"  Animal Drug: {responses[2][:100]}..." if len(responses[2]) > 100 else f"  Animal Drug: {responses[2]}")
            print(f"  In Vitro: {responses[3][:100]}..." if len(responses[3]) > 100 else f"  In Vitro: {responses[3]}")
            print(f"Counts: Genetic={counts[0]}, AnimalGene={counts[1]}, AnimalDrug={counts[2]}, InVitro={counts[3]}")
            print(f"{'='*60}")

        return {
            'Target': target,
            'Condition': condition,
            'GeneticCount': counts[0],
            'AnimalGeneCount': counts[1],
            'AnimalDrugCount': counts[2],
            'InVitroCount': counts[3]
        }
    
    except Exception as e:
        with print_lock:
            print(f"ERROR processing {target} - {condition}: {e}")
        return None

def run_validation(pairs_df):
    """
    Run validation on all target-condition pairs with parallel processing.
    pairs_df should have 'Target' and 'Conditions' columns.
    """
    # Load checkpoint
    results_df, processed = load_checkpoint()
    
    # Get list of pairs to process
    pairs = list(zip(pairs_df['Target'], pairs_df['Conditions']))
    pairs_to_process = [(t, c) for t, c in pairs if (t, c) not in processed]
    
    print(f"Total pairs: {len(pairs)}")
    print(f"Already processed: {len(processed)}")
    print(f"Remaining: {len(pairs_to_process)}")
    
    if not pairs_to_process:
        print("All pairs already processed!")
        results_df.to_csv(OUTPUT_FILE, index=False)
        return results_df
    
        # Process in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_pair, t, c): (t, c) for t, c in pairs_to_process}
        print(f"Submitted {len(futures)} futures")  # ADD THIS
        
        for future in as_completed(futures):
        	print("Future completed, processing...")  # ADD THIS
        	target, condition = futures[future]
        	result = future.result()
        	print(f"Got result for {target}-{condition}: {result is not None}")  

        	if result:
        		with file_lock:
        			results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
        			save_checkpoint(results_df)

        		completed += 1
        		with print_lock:
        			print(f"Progress: {completed}/{len(pairs_to_process)} ({100*completed/len(pairs_to_process):.1f}%)")
    
    # Save final output
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone! Results saved to {OUTPUT_FILE}")
    
    return results_df


if __name__ == "__main__":
    # Choose which dataset to process - uncomment the one you want
    # results = run_validation(successfulsick)
    # results = run_validation(unsuccessfulsick)
    
    # Or combine both:
    # all_pairs = pd.concat([successfulsick, unsuccessfulsick]).drop_duplicates()
    # results = run_validation(all_pairs)
    
    # For testing with a small sample:
    sample = successfulsick.head(5)
    results = run_validation(sample)

