import re
import pandas as pd
import os
import argparse
import requests
import time

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
MAX_RETRIES = 5
RETRY_DELAY = 10

def make_request_with_retry(url, params=None):
    """Make a request with retries and exponential backoff"""
    delay = RETRY_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=120)  # longer timeout
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code >= 500:
                print(f"  Server error ({resp.status_code}), retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(delay)
                delay *= 2
            else:
                return None
        except (requests.Timeout, requests.ConnectionError) as e:
            print(f"  Connection issue, retry {attempt + 1}/{MAX_RETRIES}...")
            time.sleep(delay)
            delay *= 2
    print(f"  Failed after {MAX_RETRIES} retries")
    return None

def get_molecule_by_synonym(drug_name):
    url = f"{CHEMBL_BASE}/molecule.json"
    params = {"molecule_synonyms__molecule_synonym__iexact": drug_name}
    data = make_request_with_retry(url, params)
    if data and data.get('molecules'):
        return data['molecules'][0]
    return None

def get_mechanisms(chembl_id):
    url = f"{CHEMBL_BASE}/mechanism.json"
    params = {"molecule_chembl_id": chembl_id}
    data = make_request_with_retry(url, params)
    if data:
        return data.get('mechanisms', [])
    return []

def get_target(target_chembl_id):
    url = f"{CHEMBL_BASE}/target/{target_chembl_id}.json"
    return make_request_with_retry(url)

def extract_year(date_str):
    if pd.isna(date_str) or date_str == '':
        return None
    return int(str(date_str).split('-')[0])

def process_row(idx, dataset):
    drug = dataset['Interventions'][idx]
    startdate_raw = dataset['Start Date'][idx]
    startdate = extract_year(startdate_raw)
    nct = dataset['NCT Number'][idx]

    truedrugs = list(d.split(": ")[1] for d in drug.strip("'").split("|") if "Placebo" not in d)
    truedrugs = set([' '.join(re.sub(r'\b\d+\s*mg\b', '', s, flags=re.IGNORECASE).split()) for s in truedrugs])

    print(f"  Drugs: {truedrugs}")

    target_success = []

    for thisdrug in truedrugs:
        print(f"    Looking up: {thisdrug}")
        mol = get_molecule_by_synonym(thisdrug)
        if mol:
            chembl_id = mol['molecule_chembl_id']
            print(f"    Found: {chembl_id}")
            mechanisms = get_mechanisms(chembl_id)
            approval_date = mol.get('first_approval')

            target_names = []
            for mech in mechanisms:
                target_id = mech.get('target_chembl_id')
                if target_id:
                    target_info = get_target(target_id)
                    if target_info:
                        target_names.append(target_info.get('pref_name', 'N/A'))

            print(f"    Targets: {target_names}, Approval: {approval_date}")

            for target_name in target_names:
                if approval_date is not None and startdate is not None and startdate <= approval_date:
                    target_success.append([nct, startdate, approval_date, target_name, 'Success'])
                elif approval_date is None:
                    target_success.append([nct, startdate, approval_date, target_name, 'No Success'])
                elif approval_date is not None and startdate is not None and startdate > approval_date:
                    target_success.append([nct, startdate, approval_date, target_name, 'Old Approved Drug'])
                elif approval_date is not None and startdate is None:
                    target_success.append([nct, startdate, approval_date, target_name, 'Approved Drug'])
                else:
                    target_success.append([nct, startdate, approval_date, target_name, 'Unknown'])
        else:
            print(f"    Not found in ChEMBL")

    return target_success


def run_all_with_checkpoints(dataset, checkpoint_every=100, output_file="target_results.csv", progress_file="progress.txt"):
    all_results = []
    start_idx = 0

    if os.path.exists(output_file) and os.path.exists(progress_file):
        print("Found existing checkpoint, resuming...")
        existing_df = pd.read_csv(output_file)
        all_results = existing_df.values.tolist()
        with open(progress_file, 'r') as f:
            start_idx = int(f.read().strip()) + 1
        print(f"Loaded {len(all_results)} existing results, resuming from row {start_idx}")

    for idx in range(start_idx, len(dataset)):
        print(f"Processing row {idx + 1}/{len(dataset)}")
        try:
            row_results = process_row(idx, dataset)
            all_results.extend(row_results)
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            continue

        if (idx + 1) % checkpoint_every == 0:
            print(f"Saving checkpoint at row {idx + 1}...")
            temp_df = pd.DataFrame(all_results, columns=["NCT Number", "Start Date", "Approval Date", "Target", "Success"])
            temp_df.to_csv(output_file, index=False)
            with open(progress_file, 'w') as f:
                f.write(str(idx))

    results_df = pd.DataFrame(all_results, columns=["NCT Number", "Start Date", "Approval Date", "Target", "Success"])
    results_df.to_csv(output_file, index=False)

    if os.path.exists(progress_file):
        os.remove(progress_file)

    print(f"Done! Saved {len(results_df)} rows to {output_file}")
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Extract drug targets from clinical trials data using ChEMBL')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('-o', '--output', default='target_results.csv', help='Output CSV file')
    parser.add_argument('-c', '--checkpoint', type=int, default=50, help='Save checkpoint every N rows')
    parser.add_argument('-p', '--progress', default='progress.txt', help='Progress file')

    args = parser.parse_args()

    print(f"Loading data from {args.input_file}...")
    dataset = pd.read_csv(args.input_file)
    print(f"Loaded {len(dataset)} rows")

    run_all_with_checkpoints(
        dataset,
        checkpoint_every=args.checkpoint,
        output_file=args.output,
        progress_file=args.progress
    )


if __name__ == '__main__':
    main()