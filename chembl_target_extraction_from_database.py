import re
import pandas as pd
import os
import argparse
import sqlite3

# Update this path to where your database is

CHEMBL_DB_PATH = "/Users/sarahconstantin/Documents/Documents/clinicaltargets/chembl_35/chembl_35_sqlite/chembl_35.db"

def get_connection():
    return sqlite3.connect(CHEMBL_DB_PATH)

def get_molecule_by_synonym(drug_name, cursor):
    """Search for molecule by synonym"""
    cursor.execute("""
        SELECT DISTINCT md.chembl_id, md.pref_name, md.first_approval
        FROM molecule_dictionary md
        JOIN molecule_synonyms ms ON md.molregno = ms.molregno
        WHERE UPPER(ms.synonyms) = UPPER(?)
        LIMIT 1
    """, (drug_name,))
    row = cursor.fetchone()
    if row:
        return {'molecule_chembl_id': row[0], 'pref_name': row[1], 'first_approval': row[2]}
    return None

def get_targets_for_molecule(chembl_id, cursor):
    """Get all targets for a molecule via mechanism of action"""
    cursor.execute("""
        SELECT DISTINCT td.pref_name
        FROM drug_mechanism dm
        JOIN target_dictionary td ON dm.tid = td.tid
        WHERE dm.molregno = (
            SELECT molregno FROM molecule_dictionary WHERE chembl_id = ?
        )
    """, (chembl_id,))
    return [row[0] for row in cursor.fetchall() if row[0]]

def extract_year(date_str):
    if pd.isna(date_str) or date_str == '':
        return None
    return int(str(date_str).split('-')[0])

def process_row(idx, dataset, cursor):
    drug = dataset['Interventions'][idx]
    startdate_raw = dataset['Start Date'][idx]
    startdate = extract_year(startdate_raw)
    nct = dataset['NCT Number'][idx]
    conditions = dataset['Conditions'][idx]

    truedrugs = list(d.split(": ")[1] for d in drug.strip("'").split("|") if "Placebo" not in d)
    truedrugs = set([' '.join(re.sub(r'\b\d+\s*mg\b', '', s, flags=re.IGNORECASE).split()) for s in truedrugs])

    target_success = []

    for thisdrug in truedrugs:
        mol = get_molecule_by_synonym(thisdrug, cursor)
        if mol:
            chembl_id = mol['molecule_chembl_id']
            approval_date = mol['first_approval']
            target_names = get_targets_for_molecule(chembl_id, cursor)

            for target_name in target_names:
                if approval_date is not None and startdate is not None and startdate <= approval_date:
                    target_success.append([nct, startdate, approval_date, conditions, target_name, 'Success'])
                elif approval_date is None:
                    target_success.append([nct, startdate, approval_date, conditions, target_name, 'No Success'])
                elif approval_date is not None and startdate is not None and startdate > approval_date:
                    target_success.append([nct, startdate, approval_date, conditions, target_name, 'Old Approved Drug'])
                elif approval_date is not None and startdate is None:
                    target_success.append([nct, startdate, approval_date, conditions, target_name, 'Approved Drug'])
                else:
                    target_success.append([nct, startdate, approval_date, conditions, target_name, 'Unknown'])

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

    conn = get_connection()
    cursor = conn.cursor()

    for idx in range(start_idx, len(dataset)):
        if idx % 100 == 0:
            print(f"Processing row {idx + 1}/{len(dataset)}")
        try:
            row_results = process_row(idx, dataset, cursor)
            all_results.extend(row_results)
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            continue

        if (idx + 1) % checkpoint_every == 0:
            print(f"Saving checkpoint at row {idx + 1}...")
            temp_df = pd.DataFrame(all_results, columns=["NCT Number", "Start Date", "Approval Date", "Conditions", "Target", "Success"])
            temp_df.to_csv(output_file, index=False)
            with open(progress_file, 'w') as f:
                f.write(str(idx))

    conn.close()

    results_df = pd.DataFrame(all_results, columns=["NCT Number", "Start Date", "Approval Date", "Conditions", "Target", "Success"])
    results_df.to_csv(output_file, index=False)

    if os.path.exists(progress_file):
        os.remove(progress_file)

    print(f"Done! Saved {len(results_df)} rows to {output_file}")
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Extract drug targets from clinical trials data using local ChEMBL')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('-o', '--output', default='target_results.csv', help='Output CSV file')
    parser.add_argument('-c', '--checkpoint', type=int, default=1000, help='Save checkpoint every N rows')
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