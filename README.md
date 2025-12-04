# Can We Predict Successful Drug Targets?

Step 1: connect clinical studies of drugs from clinicaltrials.gov with drug targets. Use the Claude API to automate assigning a drug target to each drug. (or "NULL" for unknown/undefined targets).

Step 2: define a study as "successful" if that drug is FDA-approved for the disease/condition in the study (or, for studies on healthy subjects, if the drug is FDA-approved for any indication.) Use the Claude API to automate classifying the study as YES (if successful), NO (if unsuccessful), or NULL (if unknown).

Step 3: exploratory data analysis on target trends and target success rates.  What are the most common targets? has this changed over time? What targets are the most successful? 

Step 4: develop an LLM-based research pipeline for scoring the strength of the preclinical evidence base for a target's involvement in a disease. Assign a score or rating (so some targets are rated 'stronger' than others). 

Step 5: identify whether "strong" targets have a significantly higher success rate in the clinic than "weak" targets.
