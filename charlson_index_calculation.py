##CHARLSON CATEGORY DEFINITIONS AND WEIGHTS
# Define the ICD-9 categories and associated codes (first three digits)
icd9_categories = {
    "Myocardial_infarction": ["410"],
    "Congestive_heart_failure": ["428"],
    "Peripheral_vascular_disease": ["443"],
    "Cerebrovascular_disease": ["430", "431", "432", "433", "434", "435", "436", "437", "438"],
    "Dementia": ["290"],
    "Chronic_obstructive_pulmonary_disease": ["491", "492", "496"],
    "Rheumatologic_disease": ["710"],
    "Peptic_ulcer_disease": ["531", "532", "533", "534"],
    "Liver_disease_mild_moderate": ["571"],
    "Diabetes_mellitus": ["250"],
    "Hemiplegia_paraplegia": ["342", "344"],
    "Renal_disease": ["585"],
    "Cancer": [  # Malignant and benign cancers
        "140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", 
        "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", 
        "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", 
        "173", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", 
        "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", 
        "195", "196", "197", "198", "199",  # Malignant neoplasms (up to 199)
        "210", "211", "212", "213", "214", "215", "216", "217", "218", "219", "220", 
        "221", "222", "223", "224", "225", "226", "227", "228", "229"  # Benign neoplasms (210-229)
    ],
    "Leukemia":["204", "205", "206", "207", "208"],
    "Lymphoma":["200", "202", "203"],
    "Moderate_severe_liver_disease": ["571"],
    "Metastatic_cancer": ["196", "197", "198", "199"],
    "AIDS": ["042"]
}

# Define the weights for each condition
charlson_weights = {
    "Myocardial_infarction": 1,
    "Congestive_heart_failure": 1,
    "Peripheral_vascular_disease": 1,
    "Cerebrovascular_disease": 1,
    "Dementia": 1,
    "Chronic_obstructive_pulmonary_disease": 1,
    "Rheumatologic_disease": 1,
    "Peptic_ulcer_disease": 1,
    "Liver_disease_mild_moderate": 1,
    "Diabetes_mellitus": 1,
    "Hemiplegia_paraplegia": 2,
    "Renal_disease": 2,
    "Cancer": 2,
    "Leukemia": 2,
    "Lymphoma": 2,
    "Moderate_severe_liver_disease": 3,
    "Metastatic_cancer": 6,
    "AIDS": 6
}
print("complete")

###############Charleston comorbidity index  Llama
import pandas as pd
# Load the dataset
df = pd.read_csv('llama_file.csv')

#Function to calculate cci Score
def calculate_charlson_comorbidity_index(icd9_codes):
    # If icd9_codes is None or not a string, return 0 (no comorbidities)
    if not isinstance(icd9_codes, str) or not icd9_codes:
        return 0
    
    total_cci = 0
    
    # Split the comma-separated string into a list, strip spaces, and take the first 3 digits
    icd9_codes = [code.strip()[:3] for code in icd9_codes.split(",")]
    
    # Loop through each category in the icd9_categories
    for category, codes in icd9_categories.items():
        # Check if any of the ICD-9 codes match the category codes
        if any(code in codes for code in icd9_codes):
            total_cci += charlson_weights[category]
    
    return total_cci

# Apply the function to the 'icd9_codes' column to calculate CCI score for each subject
df['cci_score'] = df['icd9_codes'].apply(calculate_charlson_comorbidity_index)

# Functon to clasify the scores
df[df['cci_score'] != 0].head(5)
def classify_comorbidity(cci_score):
    if cci_score == 0:
        return "Absence of Comorbidity"
    elif 1 <= cci_score <= 2:
        return "Mild Comorbidity"
    elif 3 <= cci_score <= 4:
        return "Moderate Comorbidity"
    else:
        return "High Comorbidity"
# Apply the function to
df['comorbidity_category'] = df['cci_score'].apply(classify_comorbidity)

#count the patients in ecah category
comorbidity_counts = df['comorbidity_category'].value_counts()
print(comorbidity_counts)


##################Charleston comorbidity index  Chatgpt
import pandas as pd
# Load the dataset
df = pd.read_csv('chatgpt_file.csv')

#Function to calculate cci Score
def calculate_charlson_comorbidity_index(icd9_codes):
    # If icd9_codes is None or not a string, return 0 (no comorbidities)
    if not isinstance(icd9_codes, str) or not icd9_codes:
        return 0
    
    total_cci = 0
    
    # Split the comma-separated string into a list, strip spaces, and take the first 3 digits
    icd9_codes = [code.strip()[:3] for code in icd9_codes.split(",")]
    
    # Loop through each category in the icd9_categories
    for category, codes in icd9_categories.items():
        # Check if any of the ICD-9 codes match the category codes
        if any(code in codes for code in icd9_codes):
            total_cci += charlson_weights[category]
    
    return total_cci

# Apply the function to the 'icd9_codes' column to calculate CCI score for each subject
df['cci_score'] = df['icd9_codes'].apply(calculate_charlson_comorbidity_index)

# Functon to clasify the scores
df[df['cci_score'] != 0].head(5)
def classify_comorbidity(cci_score):
    if cci_score == 0:
        return "Absence of Comorbidity"
    elif 1 <= cci_score <= 2:
        return "Mild Comorbidity"
    elif 3 <= cci_score <= 4:
        return "Moderate Comorbidity"
    else:
        return "High Comorbidity"
# Apply the function to
df['comorbidity_category'] = df['cci_score'].apply(classify_comorbidity)


###########Charleston comorbidity index  gemini
import pandas as pd
# Load the dataset
df = pd.read_csv('gemini_file.csv')

#Function to calculate cci Score
def calculate_charlson_comorbidity_index(icd9_codes):
    # If icd9_codes is None or not a string, return 0 (no comorbidities)
    if not isinstance(icd9_codes, str) or not icd9_codes:
        return 0
    
    total_cci = 0
    
    # Split the comma-separated string into a list, strip spaces, and take the first 3 digits
    icd9_codes = [code.strip()[:3] for code in icd9_codes.split(",")]
    
    # Loop through each category in the icd9_categories
    for category, codes in icd9_categories.items():
        # Check if any of the ICD-9 codes match the category codes
        if any(code in codes for code in icd9_codes):
            total_cci += charlson_weights[category]
    
    return total_cci

# Apply the function to the 'icd9_codes' column to calculate CCI score for each subject
df['cci_score'] = df['icd9_codes'].apply(calculate_charlson_comorbidity_index)

# Functon to clasify the scores
df[df['cci_score'] != 0].head(5)
def classify_comorbidity(cci_score):
    if cci_score == 0:
        return "Absence of Comorbidity"
    elif 1 <= cci_score <= 2:
        return "Mild Comorbidity"
    elif 3 <= cci_score <= 4:
        return "Moderate Comorbidity"
    else:
        return "High Comorbidity"
# Apply the function to
df['comorbidity_category'] = df['cci_score'].apply(classify_comorbidity)

#count the patients in ecah category
comorbidity_counts = df['comorbidity_category'].value_counts()
print(comorbidity_counts)

#count the patients in ecah category
comorbidity_counts = df['comorbidity_category'].value_counts()
print(comorbidity_counts)

######################Charleston comorbidity index  MIMIC-IV sample
import pandas as pd
# Load the dataset
df = pd.read_csv('mimic_file.csv')

#Function to calculate cci Score
def calculate_charlson_comorbidity_index(converted_icd_code):
    # If converted_icd_code is None or not a string, return 0 (no comorbidities)
    if not isinstance(converted_icd_code, str) or not converted_icd_code:
        return 0
    
    total_cci = 0
    
    # Split the comma-separated string into a list, strip spaces, and take the first 3 digits
    converted_icd_code = [code.strip()[:3] for code in converted_icd_code.split(",")]
    
    # Loop through each category in the icd9_categories
    for category, codes in icd9_categories.items():
        # Check if any of the ICD-9 codes match the category codes
        if any(code in codes for code in converted_icd_code):
            total_cci += charlson_weights[category]
    
    return total_cci

# Apply the function to the 'converted_icd_code' column to calculate CCI score for each subject
df['cci_score'] = df['converted_icd_code'].apply(calculate_charlson_comorbidity_index)

# Functon to clasify the scores
df[df['cci_score'] != 0].head(5)
def classify_comorbidity(cci_score):
    if cci_score == 0:
        return "Absence of Comorbidity"
    elif 1 <= cci_score <= 2:
        return "Mild Comorbidity"
    elif 3 <= cci_score <= 4:
        return "Moderate Comorbidity"
    else:
        return "High Comorbidity"
# Apply the function to
df['comorbidity_category'] = df['cci_score'].apply(classify_comorbidity)

#count the patients in ecah category
comorbidity_counts = df['comorbidity_category'].value_counts()
print(comorbidity_counts)
