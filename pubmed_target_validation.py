import re
import pandas as pd
import os


target_results = pd.read_csv('target_results.csv')

successful = target_results[(target_results['Success'] == 'Success')]
successfulpairs = successful[['Target', 'Conditions']].assign(Conditions=successful['Conditions'].str.split('|')).explode('Conditions').drop_duplicates()
successfulsick = successfulpairs[~successfulpairs['Conditions'].str.contains('Healthy', na=False)]

unsuccessful = target_results[(target_results['Success'] == 'No Success')]
unsuccessfulpairs = unsuccessful[['Target', 'Conditions']].assign(Conditions=unsuccessful['Conditions'].str.split('|')).explode('Conditions').drop_duplicates()
unsuccessfulsick = unsuccessfulpairs[~unsuccessfulpairs['Conditions'].str.contains('Healthy', na=False)]
