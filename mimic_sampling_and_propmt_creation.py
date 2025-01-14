# -*- coding: utf-8 -*-
"""
Created on Mon Nov 5 21:45:38 2024

@author: Parvati
"""
import pandas as pd
import re

df_discharge =  pd.read_csv('discharge.csv')
df_discharge_detail = pd.read_csv('discharge_detail.csv')
df_diagnosis =  pd.read_csv('diagnoses_icd.csv')
df_diagnosis_detail =  pd.read_csv('d_diagnoses_icd.csv')
df_patient = pd.read_csv('patient.csv')
df_adm = pd.read_csv('admission.csv')

'''
The section of the script helps in creating a sample from MIMIV-IV dataset
'''

print(df_adm.columns)

#getting the 150 subjects that have readmissions > 1 after the smallest seq number note was selected 
# Step 1: Ensure 'admittime' is a datetime object if not already
df_adm['admittime'] = pd.to_datetime(df_adm['admittime'])

# Step 2: Extract only the date part from 'admittime' for easier comparison
df_adm['admit_date'] = df_adm['admittime'].dt.date

# Step 3: Merge df_adm with df_discharge to bring in discharge records
merged_discharge = pd.merge(df_discharge, df_adm[['subject_id', 'hadm_id', 'admit_date']], on=['subject_id', 'hadm_id'], how='inner')

# Step 4: Merge df_diagnosis with merged_discharge on 'subject_id' and 'hadm_id'
merged_discharge_with_diagnosis = pd.merge(
    merged_discharge,
    df_diagnosis[['subject_id', 'hadm_id', 'icd_code', 'icd_version']].astype({'icd_code': 'str'}),
    on=['subject_id', 'hadm_id'],
    how='left'
)

#Step 5: Ensure 'icd_code' values are treated as strings to avoid type issues
merged_discharge_with_diagnosis['icd_code'] = merged_discharge_with_diagnosis['icd_code'].astype(str)
merged_discharge_with_diagnosis['icd_version'] = merged_discharge_with_diagnosis['icd_version'].astype(str)

# Step 6: Group by 'subject_id' and 'hadm_id', aggregating unique 'icd_code' and 'icd_version'
aggregated_diagnosis = merged_discharge_with_diagnosis.groupby(['subject_id', 'hadm_id']).agg({
    'icd_code': lambda x: ','.join(sorted(x.unique())),
    'icd_version': lambda x: ','.join(sorted(x.unique()))
}).reset_index()

# Step 7: Merge the aggregated diagnosis data back into the merged_discharge DataFrame
merged_discharge_with_diagnosis = pd.merge(
    merged_discharge_with_diagnosis.drop(['icd_code', 'icd_version'], axis=1),
    aggregated_diagnosis,
    on=['subject_id', 'hadm_id'],
    how='left'
)

# Step 8: Replace string 'nan' with actual NaN values
merged_discharge_with_diagnosis['icd_code'].replace('nan', pd.NA, inplace=True)
merged_discharge_with_diagnosis['icd_version'].replace('nan', pd.NA, inplace=True)

# Step 9: Drop rows where 'icd_code' or 'icd_version' is null or empty
filtered_discharge_data = merged_discharge_with_diagnosis.dropna(subset=['icd_code', 'icd_version'])

# Step 10: Additionally, remove rows where 'icd_code' or 'icd_version' is an empty string
filtered_discharge_data = filtered_discharge_data[
    (filtered_discharge_data['icd_code'] != '') & 
    (filtered_discharge_data['icd_version'] != '')
]

# Step 11: For each subject, identify the row with the smallest note_seq
idx_min_note_seq = filtered_discharge_data.groupby('subject_id')['note_seq'].idxmin()
smallest_note_seq_data = filtered_discharge_data.loc[idx_min_note_seq, ['subject_id', 'hadm_id', 'note_seq', 'admit_date', 'text','icd_code', 'icd_version']]


# Step 12: Merge smallest_note_seq_data back with df_adm to include all admissions for each subject
merged_with_admissions = pd.merge(
    df_adm, 
    smallest_note_seq_data[['subject_id', 'hadm_id', 'admit_date']], 
    on='subject_id', 
    how='inner', 
    suffixes=('', '_min_note_seq')
)

# Step 13: Filter to only admissions that occur after the smallest `note_seq` admission date
future_admissions = merged_with_admissions[
    merged_with_admissions['admit_date'] > merged_with_admissions['admit_date_min_note_seq']
]

# Step 14: Count future admissions per subject
future_admission_counts = future_admissions.groupby('subject_id').size().reset_index(name='future_admission_count')

# Step 15: Filter for subjects with more than one admission after the smallest note_seq admission
readmitted_subjects = future_admission_counts[future_admission_counts['future_admission_count'] > 1]

# Step 16: Merge back with smallest_note_seq_data to get the required columns and filter down to 150 unique subject IDs
sampled_data_more_than_one = pd.merge(
    smallest_note_seq_data,
    readmitted_subjects[['subject_id', 'future_admission_count']],
    on='subject_id',
    how='inner'
)

# Step 17: Randomly select 150 unique subject_ids
sampled_subjects = sampled_data_more_than_one['subject_id'].drop_duplicates().sample(n=151, random_state=42)
sampled_data_min_note_seq_150 = sampled_data_more_than_one[sampled_data_more_than_one['subject_id'].isin(sampled_subjects)]

# Step 17.a: Keep only the required columns: 'subject_id', 'hadm_id', 'note_seq', 'text'
sampled_data_min_note_seq_150 = sampled_data_min_note_seq_150[['subject_id', 'hadm_id', 'note_seq', 'text', 'future_admission_count', 'icd_code', 'icd_version']]

# Step 18: Save to CSV if needed
#sampled_data_min_note_seq_150.to_csv('test_sampled_min_note_seq_151.csv', index=False)

#getting the next 150 subjects that do not have future admissions

# Step 1: Count the number of unique admissions per subject by 'hadm_id'
single_admission_subjects = (
    df_adm
    .groupby('subject_id')['hadm_id']  # Group by subject_id and hadm_id (to count admissions)
    .nunique()  # Count unique hadm_id values (admissions)
    .reset_index()
)

# Step 2: Filter to keep only subjects with exactly one admission (i.e., 1 unique hadm_id)
single_admission_subjects = single_admission_subjects[single_admission_subjects['hadm_id'] == 1]

# Step 3: Convert to DataFrame for merging
single_admission_subjects_df = single_admission_subjects[['subject_id']]

# Step 4: Merge with df_discharge to get relevant discharge records
merged_single_discharge = pd.merge(df_discharge, single_admission_subjects_df, on='subject_id', how='inner')

# Step 5: Merge df_diagnosis with merged_single_discharge on 'subject_id' and 'hadm_id'
merged_single_discharge_with_diagnosis = pd.merge(
    merged_single_discharge,
    df_diagnosis[['subject_id', 'hadm_id', 'icd_code', 'icd_version']].astype({'icd_code': 'str'}),
    on=['subject_id', 'hadm_id'],
    how='left'
)

# Step 6: Ensure 'icd_code' values are treated as strings to avoid type issues
merged_single_discharge_with_diagnosis['icd_code'] = merged_single_discharge_with_diagnosis['icd_code'].astype(str)
merged_single_discharge_with_diagnosis['icd_version'] = merged_single_discharge_with_diagnosis['icd_version'].astype(str)


# Step 7: Group by 'subject_id' and 'hadm_id', aggregating unique 'icd_code' and 'icd_version'
aggregated_diagnosis_single = merged_single_discharge_with_diagnosis.groupby(['subject_id', 'hadm_id']).agg({
    'icd_code': lambda x: ','.join(sorted(x.unique())),
    'icd_version': lambda x: ','.join(sorted(x.unique()))
}).reset_index()

# Step 8: Merge the aggregated diagnosis data back into the merged_single_discharge DataFrame
merged_single_discharge_with_diagnosis = pd.merge(
    merged_single_discharge_with_diagnosis.drop(['icd_code', 'icd_version'], axis=1),
    aggregated_diagnosis_single,
    on=['subject_id', 'hadm_id'],
    how='left'
)

# Step 9: Replace string 'nan' with actual NaN values in icd_code and icd_version columns
merged_single_discharge_with_diagnosis['icd_code'].replace('nan', pd.NA, inplace=True)
merged_single_discharge_with_diagnosis['icd_version'].replace('nan', pd.NA, inplace=True)

# Step 10: Drop rows where 'icd_code' or 'icd_version' is null or empty
merged_single_discharge_with_diagnosis = merged_single_discharge_with_diagnosis.dropna(subset=['icd_code', 'icd_version'])

# Step 10: Drop rows where either 'icd_code' or 'icd_version' is empty string as well
merged_single_discharge_with_diagnosis = merged_single_discharge_with_diagnosis[
    (merged_single_discharge_with_diagnosis['icd_code'] != '') & 
    (merged_single_discharge_with_diagnosis['icd_version'] != '')
]


# Step 11: Filter for non-null text
merged_single_discharge_with_diagnosis = merged_single_discharge_with_diagnosis[merged_single_discharge_with_diagnosis['text'].notna()]

# Step 12: Randomly sample 150 unique subject IDs
unique_single_subjects = merged_single_discharge_with_diagnosis['subject_id'].unique()
sampled_single_subjects = pd.Series(unique_single_subjects).sample(n=150, random_state=42)

# Step 13: Filter the merged DataFrame to include only the sampled subject IDs and the smallest note_seq per subject
sampled_data_single = merged_single_discharge_with_diagnosis[merged_single_discharge_with_diagnosis['subject_id'].isin(sampled_single_subjects)]

# Step 14: Get the smallest note_seq for each subject in this subset
idx_min_note_seq_single = sampled_data_single.groupby('subject_id')['note_seq'].idxmin()
sampled_data_min_note_seq_single = sampled_data_single.loc[idx_min_note_seq_single, ['subject_id', 'hadm_id', 'note_seq', 'text','icd_code', 'icd_version']]

# Step 15: Add a future_admission_count column with a value of 0 for this subset
sampled_data_min_note_seq_single['future_admission_count'] = 0

# Step 16: Save to CSV if needed
#sampled_data_min_note_seq_single.to_csv('test_sampled_data_min_note_seq_single.csv', index=False)


#get final sample of 300
# Step 1: Union the two samples into a single DataFrame
final_sample_300 = pd.concat([sampled_data_min_note_seq_150, sampled_data_min_note_seq_single], ignore_index=True)

final_sample_300['text'] = final_sample_300['text'].str.replace('\t', ' ', regex=False)
final_sample_300['text'] = final_sample_300['text'].str.replace('\n', ' ', regex=False)
final_sample_300['text'] = final_sample_300['text'].str.replace(',', ';', regex=False)

# Step 1: Save to CSV if needed
#final_sample_300.to_csv('C:/Users/Parvati/Desktop/mimic/final_sample_300.csv', sep=',', index=False,quotechar='"')

'''
The section of the script helps in extraction of relavant sections of notes to create prompt
'''


# Define sections and their phrases in a strict order to enforce extraction sequence
sections = {
    "chief_complaints": ["Chief Complaint"], 
    "procedure": ["Major Surgical or Invasive Procedure", "Major  or Invasive Procedure"],
    "present_illness": ["History of Present Illness"],
    "past_med_hist": ["Past Medical History", "Past"],
    "physical": ["Physical Exam", "Admission Exam", "ADMISSION EXAM"],
    "social": ["Social History"],
    "family": ["Family History"],
    "labs": ["Pertinent Labs", "Pertinent Results", "Admission Labs", "LABS", "NOTABLE LABS"],
    "imaging_section": ["Imaging", "MRI", "MRA head/neck", "Echo", "echo", "ECHO", "CXR", "CTA", "STUDIES", 
                         "Studies", "CTAB", "ECG", "CT Scan", "CT of", "Echocardiogram", "ERCP"],
    "hosp_course": ["Brief Hospital Course"],
    "medications": ["Medications on Admission", "Discharge Medications"],
    "discharge_diag": ["Discharge Diagnosis", "Diagnosis"],
    "discharge_cond": ["Discharge Condition"]
}

# Initialize a list to store extracted DataFrames for each note
extracted_dfs = []

# Iterate over each clinical note in the DataFrame
for index, row in final_sample_300.iterrows():
    clinical_notes = row['text']  # Get the clinical note text
    subject_id = row['subject_id']  # Get subject_id
    hadm_id = row['hadm_id']  # Get hadm_id
    future_admission_count = row['future_admission_count']  # Get future_admission_count

    # Clean up newlines and replace them with spaces
    clinical_notes = clinical_notes.replace("\n", " ")

    # Initialize a dictionary to store extracted data
    extracted_data = {key: "" for key in sections.keys()}

    # Track the end of the last extracted section
    last_end_index = 0

    # Store indices that have been captured for imaging and labs to avoid duplication
    captured_labs_indices = set()
    captured_imaging_indices = set()

    # Iterate over each section to extract relevant data in the defined order
    for section, phrases in sections.items():
        # Compile regex pattern for each phrase to enforce whole-word matching
        section_patterns = [rf'\b{re.escape(phrase.strip())}\b' for phrase in phrases]

        # Track the first occurrence of any keyword for the section
        start_index = None
        earliest_match = None

        # Check for the first occurring keyword in the section
        for pattern in section_patterns:
            match = re.search(pattern, clinical_notes[last_end_index:], flags=re.IGNORECASE)
            if match:
                if not earliest_match or match.start() < earliest_match.start():
                    # Update the earliest match
                    earliest_match = match
                    start_index = last_end_index + earliest_match.start()

        # Proceed if a match was found
        if start_index is not None:
            # Determine where to stop extracting
            end_index = len(clinical_notes)  # Default to the end of the note
            for next_section in sections.keys():
                if next_section != section:
                    # Compile regex pattern for each next section phrase
                    next_section_patterns = [rf'\b{re.escape(next_phrase.strip())}\b' for next_phrase in sections[next_section]]
                    for next_pattern in next_section_patterns:
                        next_match = re.search(next_pattern, clinical_notes[start_index:], flags=re.IGNORECASE)
                        if next_match:
                            next_start_index = start_index + next_match.start()
                            if next_start_index < end_index:
                                end_index = next_start_index

            # Extract the section text between the current section start and the next section start
            extracted_text = clinical_notes[start_index:end_index].strip()

            # Labs special handling: Ensure only admission-related labs and pertinent labs are captured
            if section == "labs":
                # Check if it's part of imaging section keywords
                imaging_keywords = sections["imaging_section"]
                if not any(re.search(rf'\b{re.escape(keyword.strip())}\b', extracted_text, flags=re.IGNORECASE) for keyword in imaging_keywords):
                    # Make sure that if we already captured this part, we don't capture it again
                    if start_index not in captured_labs_indices:
                        extracted_data[section] += extracted_text + " "
                        captured_labs_indices.add(start_index)

            # Imaging special handling: Ensure only imaging-related content is captured
            elif section == "imaging_section":
                # Check if it's part of imaging section keywords
                if any(re.search(rf'\b{re.escape(keyword.strip())}\b', extracted_text, flags=re.IGNORECASE) for keyword in sections["imaging_section"]):
                    # Make sure that if we already captured this part, we don't capture it again
                    if start_index not in captured_imaging_indices:
                        extracted_data[section] += extracted_text + " "
                        captured_imaging_indices.add(start_index)

            else:
                # Default extraction for other sections
                extracted_data[section] += extracted_text + " "

            # Update the last_end_index to the current section's end
            last_end_index = end_index


    # Clean up extracted data to remove unwanted characters
    for key in extracted_data.keys():
        # Remove unwanted characters like "#" and "="
        extracted_data[key] = re.sub(r"[#=]", "", extracted_data[key])
        # Remove empty lines and trim whitespace
        cleaned_text = "\n".join([line.strip() for line in extracted_data[key].splitlines() if line.strip()])
        extracted_data[key] = cleaned_text
    
    # Add subject_id and hadm_id to the extracted data
    extracted_data['subject_id'] = subject_id
    extracted_data['hadm_id'] = hadm_id
    extracted_data['future_admission_count'] = future_admission_count

    # Convert the extracted data for this note to a DataFrame and add to the list
    extracted_dfs.append(pd.DataFrame([extracted_data]))

# Step 3: Concatenate all extracted DataFrames
df_with_sections = pd.concat(extracted_dfs, ignore_index=True)

# Step 4: Drop specified columns from the final DataFrame
columns_to_drop = ['present_illness', 'physical', 'social', 'family', 'hosp_course', 'medications', 'discharge_cond']
df_with_sections.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Step 5: Define phrases to remove per column in a dictionary to clean th data as much as possible. you can awalys add to this if you find more.
phrases_to_remove_per_column = {
    "chief_complaints": ["Chief Complaint:", "___","Complaint:"],
    "procedure": ["Major Surgical or Invasive Procedure:", "NONE", "None", "none", "N/A", "___", ":", "=", "=-"],
    "past_med_hist": ["Past Medical History:", "PAST MEDICAL HISTORY:", "___","=","=-"],
    "labs": ["Pertinent Results:", "ADMISSION Labs: ", "___", "ADMISSION LABS: ", "Admission Labs", "Admission labs", "Pertinent labs",
             "Admission:", "ADMISSION/IMPORTANT LABS:", "Laboratory results:", "LABS: ", "ADMISSION LABS --------------", 
             "ADMISSION LABS", "Admit labs:", "admission:", "ADMISSION:", "admission labs"],
    "discharge_diag": ["Discharge Diagnosis:", "___"]
    # Add other columns and their specific phrases here
}

# Step 6: Iterate through each column and its phrases
for col, phrases in phrases_to_remove_per_column.items():
    if col in df_with_sections.columns:
        for phrase in phrases:
            # Replace each phrase with an empty string and strip whitespace
            df_with_sections[col] = df_with_sections[col].str.replace(phrase, "", regex=False).str.strip()

# Step 7: Save the resulting DataFrame to a CSV file with pipe separator-- paste this into excel fr validation of sections. if any further fine tuning of code is needed
#df_with_sections.to_csv('df_with_sections_cleaned.csv', sep='|', index=False, quoting=1)

#create a propt through concatination

# Step 1: Generate the prompt without new lines between sections
df_with_sections['prompt'] = (
    "Here is my Patient Information "
    "Chief Complaints: {chief_complaints} "
    "Procedure: {procedure} "
    "Past Medical History: {past_med_hist} "
    "Labs: {labs} "
    "Imaging: {imaging_section} "
    "Based on the above patient information, please provide the following in a structured format as indicated below: "
    "primary_diagnosis: Identify the primary diagnosis for the patient based on the provided sections. "
    "icd9_codes: associated with the diagnosis. "
    "explaination_of_why_this_diagnosis: explain why you think. "
    "readmission_risk: choose one of these low/medium/high. "
    "explaination_of_why_redamission_risk: explain why you think. "
    "Ensure that each section is clearly labeled, as shown above, to facilitate extraction using structured formatting. Donot provide any further explaination beyond thats asked."
).format(
    subject_id="{subject_id}",
    hadm_id="{hadm_id}",
    chief_complaints="{chief_complaints}",
    procedure="{procedure}",
    past_med_hist="{past_med_hist}",
    labs="{labs}",
    imaging_section="{imaging_section}"
)

# Step 2: Now format the 'prompt' column with each row's data
df_with_sections['prompt'] = df_with_sections.apply(lambda row: row['prompt'].format(
    subject_id=row['subject_id'],
    hadm_id=row['hadm_id'],
    chief_complaints=row['chief_complaints'] if pd.notna(row['chief_complaints']) else "Not provided",
    procedure=row['procedure'] if pd.notna(row['procedure']) else "Not provided",
    past_med_hist=row['past_med_hist'] if pd.notna(row['past_med_hist']) else "Not provided",
    labs=row['labs'] if pd.notna(row['labs']) else "Not provided",
    imaging_section=row['imaging_section'] if pd.notna(row['imaging_section']) else "Not provided"
), axis=1)

# Step 3: Select the desired columns for the new DataFrame
df_prompt = df_with_sections[['subject_id', 'hadm_id', 'prompt', 'future_admission_count']]

# Step 4: Save the resulting DataFrame to a CSV file 
#df_prompt.to_csv('df_prompt.csv', sep=',', index=False, quoting=1)
