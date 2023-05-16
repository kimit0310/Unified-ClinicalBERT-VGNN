import pandas as pd

# Read the preprocessed_ADMISSIONS.csv file
preprocessed_admissions_df = pd.read_csv('preprocessed_ADMISSIONS.csv', dtype={4: 'str', 5: 'str'})

# Read the grouped_NOTEEVENTS.csv file
grouped_noteevents_df = pd.read_csv('preprocessed_NOTEEVENTS.csv')

# Merge the preprocessed_ADMISSIONS.csv file with the grouped_NOTEEVENTS.csv file
merged_df = preprocessed_admissions_df.merge(grouped_noteevents_df[['HADM_ID', 'TEXT']], on='HADM_ID', how='left')

# Get the number of unique values for each column in the merged DataFrame
merged_unique_counts = merged_df.nunique()

# Find rows from grouped_NOTEEVENTS.csv that do not have a match in preprocessed_ADMISSIONS.csv
no_match_df = grouped_noteevents_df[~grouped_noteevents_df['HADM_ID'].isin(preprocessed_admissions_df['HADM_ID'])][['SUBJECT_ID', 'HADM_ID', 'TEXT']]

# Get the number of unique values for each column in the no_match DataFrame
no_match_unique_counts = no_match_df.nunique()

# Print the unique counts for merged_df and no_match_df
print("Unique counts for merged_df:")
print(merged_unique_counts)
print("\nUnique counts for no_match_df:")
print(no_match_unique_counts)

# Save the merged DataFrame to a CSV file
merged_df.to_csv('merged_ADMISSIONS_NOTEEVENTS.csv', index=False)

# Save the no_match DataFrame to a CSV file
no_match_df.to_csv('no_match_NOTEEVENTS.csv', index=False)
