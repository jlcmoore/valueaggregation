import os
import pandas as pd

# Directory containing the files
directory = 'data/results/mturk'

# Columns to be deleted
columns_to_delete = ['HITId', 'HITTypeId', 'RequesterAnnotation', 'AssignmentId']

# Dictionary to map original WorkerId to new unique integer ID
worker_id_map = {}
current_id = 1

# Function to process each file
def process_file(filepath):
    global current_id
    df = pd.read_csv(filepath)
    
    # Replace WorkerId with new unique identifier
    if 'WorkerId' in df.columns:
        new_worker_ids = []
        for worker_id in df['WorkerId']:
            if worker_id not in worker_id_map:
                worker_id_map[worker_id] = current_id
                current_id += 1
            new_worker_ids.append(worker_id_map[worker_id])
        df['WorkerId'] = new_worker_ids
    
    # Drop the specified columns
    df.drop(columns=columns_to_delete, inplace=True, errors='ignore')
    
    # Save the modified dataframe back to the original file
    df.to_csv(filepath, index=False)

# Walk through the directory and process each file
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.csv'):
            filepath = os.path.join(root, file)
            process_file(filepath)

print("Processing complete.")