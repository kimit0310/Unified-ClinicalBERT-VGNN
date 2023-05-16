import pandas as pd
import re
import string
from tqdm import tqdm

def preprocess1(x):
    if isinstance(x, str):
        y = re.sub('\\[(.*?)\\]', '', x)  # remove de-identified brackets
        y = re.sub('[0-9]+\\.', '', y)  # remove 1.2. since the segmenter segments based on this
        y = re.sub('dr\\.', 'doctor', y)
        y = re.sub('m\\.d\\.', 'md', y)
        y = re.sub('admission date:', '', y)
        y = re.sub('discharge date:', '', y)
        y = re.sub('--|__|==', '', y)

        # remove digits and spaces
        y = y.translate(str.maketrans("", "", string.digits))
        y = " ".join(y.split())
    else:
        y = ''
    return y

def preprocessing(df_notes):
    df_notes['TEXT'] = df_notes['TEXT'].fillna('')  # fill missing values in the TEXT column
    df_notes['TEXT'] = df_notes['TEXT'].apply(str)  # convert all values to strings

    df_notes['TEXT'] = df_notes['TEXT'].str.replace('\n', ' ')
    df_notes['TEXT'] = df_notes['TEXT'].str.replace('\r', ' ')
    df_notes['TEXT'] = df_notes['TEXT'].apply(str.strip)
    df_notes['TEXT'] = df_notes['TEXT'].str.lower()

    df_notes['TEXT'] = df_notes['TEXT'].apply(lambda x: preprocess1(x))
    
    # Group by HADM_ID within the chunk and concatenate the TEXT values
    df_notes = df_notes.groupby('HADM_ID').agg({'ROW_ID': 'first', 'SUBJECT_ID': 'first', 'TEXT': lambda x: ' '.join(x)}).reset_index()

    return df_notes


def process_chunks(chunk_iterator):
    chunk_list = []
    chunk_counter = 0

    for chunk in tqdm(chunk_iterator, desc="Processing chunks"):
        preprocessed_chunk = preprocessing(chunk)
        chunk_list.append(preprocessed_chunk)
        chunk_counter += 1

    # Concatenate all preprocessed chunks
    concatenated_df = pd.concat(chunk_list)

    # Further group by HADM_ID across chunks and concatenate the TEXT values
    final_df = concatenated_df.groupby('HADM_ID').agg({'ROW_ID': 'first', 'SUBJECT_ID': 'first', 'TEXT': lambda x: ' '.join(x)}).reset_index()

    return final_df

# Read the NOTEEVENTS.csv file in chunks
chunksize = 10000
chunk_iterator = pd.read_csv('/home/ik798/proj/preprocess/NOTEEVENTS.csv', dtype={4: 'str', 5: 'str'}, chunksize=chunksize)

# Process chunks and concatenate the preprocessed chunks
preprocessed_noteevents_df = process_chunks(chunk_iterator)

# Save the preprocessed DataFrame to a CSV file
preprocessed_noteevents_df.to_csv('preprocessed_NOTEEVENTS.csv', index=False)