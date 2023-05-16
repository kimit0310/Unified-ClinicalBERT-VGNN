import pandas as pd

# Read the preprocessed_NOTEEVENTS.csv file
grouped_noteevents_df = pd.read_csv('preprocessed_NOTEEVENTS.csv')

# Load the train, validation, and test pickle files
train_df = pd.read_pickle('/home/ik798/proj/NYU_NLU_BERT_GNN_COMBO/data/mimic_GNN/post_data/train_csr.pkl')
val_df = pd.read_pickle('/home/ik798/proj/NYU_NLU_BERT_GNN_COMBO/data/mimic_GNN/post_data/validation_csr.pkl')
test_df = pd.read_pickle('/home/ik798/proj/NYU_NLU_BERT_GNN_COMBO/data/mimic_GNN/post_data/test_csr.pkl')

# Extract HADM_IDs and labels from the train, validation, and test data
train_hadm_ids = set(train_df[2])
val_hadm_ids = set(val_df[2])
test_hadm_ids = set(test_df[2])

train_labels = dict(zip(train_df[2], train_df[1]))
val_labels = dict(zip(val_df[2], val_df[1]))
test_labels = dict(zip(test_df[2], test_df[1]))

# Filter the grouped_NOTEEVENTS.csv file based on the HADM_IDs from the train, validation, and test data
train_matched_df = grouped_noteevents_df[grouped_noteevents_df['HADM_ID'].isin(train_hadm_ids)][['SUBJECT_ID', 'HADM_ID', 'TEXT']]
val_matched_df = grouped_noteevents_df[grouped_noteevents_df['HADM_ID'].isin(val_hadm_ids)][['SUBJECT_ID', 'HADM_ID', 'TEXT']]
test_matched_df = grouped_noteevents_df[grouped_noteevents_df['HADM_ID'].isin(test_hadm_ids)][['SUBJECT_ID', 'HADM_ID', 'TEXT']]

# Add the labels to the matched DataFrames
train_matched_df['LABEL'] = train_matched_df['HADM_ID'].map(train_labels)
val_matched_df['LABEL'] = val_matched_df['HADM_ID'].map(val_labels)
test_matched_df['LABEL'] = test_matched_df['HADM_ID'].map(test_labels)

# Save the filtered DataFrames to separate CSV files
train_matched_df.to_csv('train_matched.csv', index=False)
val_matched_df.to_csv('val_matched.csv', index=False)
test_matched_df.to_csv('test_matched.csv', index=False)