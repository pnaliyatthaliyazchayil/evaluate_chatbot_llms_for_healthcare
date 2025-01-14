# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:44:38 2024

@author: Parvati, Raajitha
"""
import pandas as pd
import re
import numpy as np

'''
This script demonstrates how to Extract each section from Chatbot llm responses.
The responses we got  from LLM is a paragraph and we had to extract it into a csv using which we can compare the response to the MIMIC-IV sample data.

The response style of every LLM differs, hence we provide the code for each LLM below. 
'''

#Extracting for LLama and ChatGPT responses into csv

# Open the file with error handling and pass it to pandas
with open('prompts_file_llama.csv', 'r', encoding='utf-8', errors='ignore') as file:
    df_llama_responses = pd.read_csv(file)

# Replace newlines with spaces in the 'response' column to preserve separation
df_llama_responses['response'] = df_llama_responses['response'].str.replace('\n', ' ', regex=False)

# Convert all columns to string
df_llama_responses = df_llama_responses.astype(str)

def extract_information(text):
    # Match the content following the label, allowing for possible whitespace and colon after the label
    
    # Use lookahead to stop at the next section's keyword
    primary_diagnosis = re.search(r'primary_diagnosis[\s:]*([^\n]+?)(?=\s*(?:icd9_codes|explaination_of_why_this_diagnosis|readmission_risk|explaination_of_why_readmission_risk|$))', text, re.IGNORECASE)
    icd9_codes = re.search(r'icd9_codes[\s:]*([^\n]+?)(?=\s*(?:explaination_of_why_this_diagnosis|readmission_risk|explaination_of_why_readmission_risk|$))', text, re.IGNORECASE)
    explaination_diagnosis = re.search(r'explaination_of_why_this_diagnosis[\s:]*([^\n]+?)(?=\s*(?:readmission_risk|explaination_of_why_readmission_risk|$))', text, re.IGNORECASE)
    
    # For readmission risk, only capture 'High', 'Medium' or 'Low'
    readmission_risk = re.search(r'readmission_risk[\s:]*\s*(High|Medium|Low)(?=\s*(?:explaination_of_why_readmission_risk|$))', text, re.IGNORECASE)
    
    explaination_readmission = re.search(r'explaination_of_why_readmission_risk[\s:]*([^\n]+)', text, re.IGNORECASE)

    # Process and clean the data
    primary_diagnosis = primary_diagnosis.group(1).strip() if primary_diagnosis else None
    icd9_codes = icd9_codes.group(1).strip() if icd9_codes else None
    explaination_diagnosis = explaination_diagnosis.group(1).strip() if explaination_diagnosis else None
    readmission_risk = readmission_risk.group(1).strip() if readmission_risk else None
    explaination_readmission = explaination_readmission.group(1).strip() if explaination_readmission else None

    # Handle ICD9 codes (if there are multiple codes)
    if icd9_codes:
        # Modified regex to capture the full ICD-9 code (e.g., 123.45, E123, or V45.9)
        codes = re.findall(r'\b([A-Za-z]?\d{2,4}(?:\.\d{1,2})?)\b', icd9_codes)
        icd9_codes = ', '.join(codes)

    return {
        "primary_diagnosis": primary_diagnosis,
        "icd9_codes": icd9_codes,
        "explaination_of_why_this_diagnosis": explaination_diagnosis,
        "readmission_risk": readmission_risk,
        "explaination_of_why_readmission_risk": explaination_readmission
    }

# Apply extraction to each row
df_extracted = df_llama_responses['response'].apply(extract_information).apply(pd.Series)

# Combine with subject_id and print the result
df_final = pd.concat([df_llama_responses[['subject_id', 'response']], df_extracted], axis=1)

# Convert all columns in df_final to string type
df_final = df_final.astype(str)

# Save to a new CSV file
#df_final.to_csv('llama_response_sectioned.csv', index=False)


#Extracting for Gemini response into csv. Gemini output was in json format by default


import pandas as pd
import json
import numpy as np

def extract_json_data(df):
    """Extracts specific fields from JSON data in a DataFrame.

    Args:
        df: A pandas DataFrame containing JSON strings in the 'response' column.

    Returns:
        A pandas DataFrame with extracted fields.
    """
    
    # Parse the JSON strings in the 'response' column
    # If there are any extra quotes, remove them before parsing
    df['response'] = df['response'].apply(lambda x: json.loads(x.strip('"')) if isinstance(x, str) else x)

    # Extract specific fields from the JSON data
    df['primary_diagnosis'] = df['response'].apply(lambda x: x.get('primary_diagnosis', None))
    df['icd9_codes'] = df['response'].apply(lambda x: x.get('icd9_codes', None))
    df['explanation_of_why_this_diagnosis'] = df['response'].apply(lambda x: x.get('explanation_of_why_this_diagnosis', None))
    df['readmission_risk'] = df['response'].apply(lambda x: x.get('readmission_risk', None))
    df['explanation_of_why_readmission_risk'] = df['response'].apply(lambda x: x.get('explanation_of_why_readmission_risk', None))

    # Drop the original 'response' column if not needed
    df = df.drop('response', axis=1)

    return df

# Load your DataFrame from the CSV
file_path = 'GEMINI_INPUT_FILE.csv'
df = pd.read_csv(file_path)

# Apply the extraction function
extracted_df = extract_json_data(df)

extracted_df.to_csv('gemini_response_sectioned.csv', index=False)
